import pathlib
import sys

import numpy as np
import psycopg2
import io
import pandas as pd
import cv2
from pfs.utils.coordinates import CoordTransp
from pfs.utils.coordinates import DistortionCoefficients

import opscore.protocols.keys as keys
import opscore.protocols.types as types

from opscore.utility.qstr import qstr

from ics.fpsActor import fpsState
from ics.fpsActor import najaVenator

class FpsCmd(object):
    def __init__(self, actor):
        # This lets us access the rest of the actor.
        self.actor = actor
        
        self.nv = najaVenator.NajaVenator()

        self.tranMatrix = None
        # Declare the commands we implement. When the actor is started
        # these are registered with the parser, which will call the
        # associated methods when matched. The callbacks will be
        # passed a single argument, the parsed and typed command.
        #
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('loadDesign', '<id>', self.loadDesign),
            ('moveToDesign', '', self.moveToDesign),
            ('getAEfromFF', '', self.getAEfromFF),
            ('testCamera', '[<cnt>] [<expTime>] [@noCentroids]', self.testCamera),
            ('testLoop', '[<cnt>] [<expTime>] [<visit>]', self.testLoop),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("visit", types.Int(), help="PFS visit to use"),
                                        keys.Key("id", types.Long(),
                                                 help=("fpsDesignId for the field, "
                                                       "which defines the fiber positions")),
                                        keys.Key("expTime", types.Float(), 
                                                 help="Seconds for exposure"))

    def ping(self, cmd):
        """Query the actor for liveness/happiness."""

        cmd.finish("text='Present and (probably) well'")

    def status(self, cmd):
        """Report status and version; obtain and send current data"""

        self.actor.sendVersionKey(cmd)

        keyStrings = ['text="FPS Actor status report"']
        keyMsg = '; '.join(keyStrings)

        cmd.inform(keyMsg)
        cmd.diag(sys.path)
        cmd.diag('text="FPS ready to go."')
        cmd.finish()

    def _loadPfsDesign(self, cmd, designId):
        """ Return the pfsDesign for the given pfsDesignId. """

        cmd.warn(f'text="have a pfsDesignId={designId:#016x}, but do not know how to fetch it yet."')

        return None

    def loadDesign(self, cmd):
        """ Load our design from the given pfsDesignId. """

        designId = cmd.cmd.keywords['id'].values[0]

        try:
            design = self._loadPfsDesign(cmd, designId)
        except Exception as e:
            cmd.fail(f'text="Failed to load pfsDesign for pfsSDesignId={designId}: {e}"')
            return

        fpsState.fpsState.setDesign(designId, design)
        cmd.finish(f'pfsDesignId={designId:#016x}')

    def moveToDesign(self,cmd):
        """ Move cobras to the pfsDesign. """

        raise NotImplementedError('moveToDesign')
        cmd.finish()

    def _mcsExpose(self, cmd, frameId, expTime=None, doCentroid=True):
        """ Request a single MCS exposure, with centroids by default.

        Args
        ----
        cmd : `actorcore.Command`
          What we report back to.
        frameId : `int`
          The full 8-digit frame number for the MCS
        expTime : `float`
          1.0s by default.
        doCentroid : bool
          Whether to measure centroids

        Returns
        -------
        frameId : `int`
          The frameId of the image. Should match `mcsData.frameId`

        """

        if expTime is None:
            expTime = 1.0

        cmdString = "expose object frameId=%d expTime=%0.1f %s" % (frameId, expTime,
                                                                   'doCentroid' if doCentroid else '')
        cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                      forUserCmd=cmd, timeLim=expTime+30)
        if cmdVar.didFail:
            cmd.warn('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))
            return None

        filekey= self.actor.models['mcs'].keyVarDict['filename'][0]
        filename = pathlib.Path(filekey)
        frameId = int(filename.stem[4:], base=10)
        cmd.inform(f'frameId={frameId}')

        return frameId

    def _findHomes(baseData,poolData):
    
        """
        Do nearest neighbour matching on a set of centroids (ie, home position case).

        """

        orix=[]
        oriy=[]
        px = []
        py = []
        fiberid = []
        for index, row in baseData.iterrows():
            dist=np.sqrt((row['x']-poolData['pfix'])**2 + (row['y']-poolData['pfiy'])**2)
            ind=pd.Series.idxmin(dist)

            orix.append(row['x'])
            oriy.append(row['y'])
            fiberid.append(row['fiberID'])
            if(min(dist) < 10):
                px.append(poolData['pfix'][ind])
                py.append(poolData['pfiy'][ind])
                #otherwise set values to NaN
            else:
                px.append(np.nan)
                py.append(np.nan)
            
        d = {'fiberID': fiberid, 'orix': orix, 'oriy': oriy, 'pfix': px, 'pfiy': py}
        match = pd.DataFrame(data=d)

        return match

    def getAEfromFF(self, cmd, frameId):
        """ Checking distortion with fidicial fibers.  """
        
        #frameId= frameID
        moveId = 1

        offset=[0,-85]
        rotCent=[[4471],[2873]]

        telInform = self.nv.readTelescopeInform(frameId)
        za = 90-telInform['azi']
        inr = telInform['instrot']

        inr=inr-180
        if(inr < 0):
            inr=inr+360

        mcsData = self.nv.readCentroid(frameId, moveId)
        ffData = self.nv.readFFConfig()
        #sfData = self.nv.readCobraConfig()


        ffData['x']-=offset[0]
        ffData['y']-=offset[1]

        #correect input format
        xyin=np.array([mcsData['centroidx'],mcsData['centroidy']])

        #call the routine
        xyout=CoordTransp.CoordinateTransform(xyin,za,'mcs_pfi',inr=inr,cent=rotCent)

        d = {'ffID': np.arange(len(xyout[0])), 'pfix': xyout[0], 'pfiy': xyout[1]}
        transPos=pd.DataFrame(data=d) 
        
        match = self._findHomes(ffData, transPos)
        
        pts1=np.zeros((1,len(match['orix']),2))
        pts2=np.zeros((1,len(match['orix']),2))

        pts1[0,:,0]=match['orix']
        pts1[0,:,1]=match['oriy']

        pts2[0,:,0]=match['pfix']
        pts2[0,:,1]=match['pfiy']


        afCoeff,inlier=cv2.estimateAffinePartial2D(pts2, pts1)

        mat={}
        mat['affineCoeff'] = afCoeff
        mat['xTrans']=afCoeff[0,2]
        mat['yTrans']=afCoeff[1,2]
        mat['xScale']=np.sqrt(afCoeff[0,0]**2+afCoeff[0,1]**2)
        mat['yScale']=np.sqrt(afCoeff[1,0]**2+afCoeff[1,1]**2)
        mat['angle']=np.arctan2(afCoeff[1,0]/np.sqrt(afCoeff[0,0]**2+afCoeff[0,1]**2),
                                    afCoeff[1,1]/np.sqrt(afCoeff[1,0]**2+afCoeff[1,1]**2))


        self.tranMatrix = mat
    
    def applyAEonCobra(self, cmd, frameId):
        
        #frameId= 16493
        moveId = 1

        offset=[0,-85]

        telInform = self.nv.readTelescopeInform(frameId)
        za = 90-telInform['azi']
        inr = telInform['instrot']
        inr=inr-180
        if(inr < 0):
            inr=inr+360

        mcsData = nv.readCentroid(frameId, moveId)
        sfData = nv.readCobraConfig()

        sfData['x']-=offset[0]
        sfData['y']-=offset[1]

        


        #correect input format
        xyin=np.array([mcsData['centroidx'],mcsData['centroidy']])

        #call the routine
        xyout=CoordTransp.CoordinateTransform(xyin,za,'mcs_pfi',inr=inr,cent=rotCent)

        d = {'ffID': np.arange(len(xyout[0])), 'pfix': xyout[0], 'pfiy': xyout[1]}
        transPos=pd.DataFrame(data=d) 

        pts2=np.zeros((1,len(transPos['pfix']),2))

        pts2[0,:,0]=transPos['pfix']
        pts2[0,:,1]=transPos['pfiy']


        afCor=cv2.transform(pts2,self.tranMatrix['affineCoeff'])
        #xx=afCor[0,:,0]
        #yy=afCor[0,:,1]

        d = {'ffID': np.arange(len(afCor[0,:,0])), 'mcsx': afCor[0,:,0], 'mcsy': afCor[0,:,1]}
        mcs=pd.DataFrame(data=d) 
        match = _findHomes(sfData, mcs)

        match['dx'] = match['orix'] - match['pfix']
        match['dy'] = match['oriy'] - match['pfiy']
        
        return match

    def testCamera(self, cmd):
        """ Camera Loop Test. """

        cmdKeys = cmd.cmd.keywords
        cnt = cmdKeys["cnt"].values[0] \
              if 'cnt' in cmdKeys \
                 else 1
        expTime = cmdKeys["expTime"].values[0] \
                  if "expTime" in cmdKeys \
                     else 1.0
        doCentroid = 'noCentroids' not in cmdKeys

        for i in range(cnt):
            cmd.inform(f'text="taking exposure loop {i+1}/{cnt}"')
            visit = self._mcsExpose(cmd, expTime=expTime, doCentroid=doCentroid)
            if not visit:
                cmd.fail('text="exposure failed"')
                return

        cmd.finish()

    def testLoop(self, cmd):
        """ Run the expose-move loop a few times. For development. """

        cmdKeys = cmd.cmd.keywords
        visit = cmdKeys["visit"].values[0] \
            if 'visit' in cmdKeys \
            else None
        cnt = cmdKeys["cnt"].values[0] \
            if 'cnt' in cmdKeys \
               else 7
        expTime = cmdKeys["expTime"].values[0] \
            if "expTime" in cmdKeys \
            else None

        for i in range(cnt):
            frameId = 100*visit + i
            cmd.inform(f'text="taking exposure loop {i+1}/{cnt}"')
            frameId = self._mcsExpose(cmd, frameId, expTime=expTime, doCentroid=True)
            if not frameId:
                cmd.fail('text="exposure failed"')
                return

            cmd.inform('text="Exposure finished." ')

            if (i == 0):
                cmd.inform('text="Apply distrotion correction and getting Affine coefficients." ')
                self.getAEfromFF(cmd, frameId)

        # Can look up frameId == visit in mcsData....

        cmd.finish()

#            rawCentroids = self.actor.models['mcs'].keyVarDict['centroidsChunk'][0]
#         expTime = cmd.cmd.keywords["expTime"].values[0] \
#           if "expTime" in cmd.cmd.keywords \
#           else 0.0
# 
# 
#         times = numpy.zeros((cnt, 4), dtype='f8')
#         
#         targetPos = self.targetPositions("some field ID")
#         for i in range(cnt):
#             times[i,0] = time.time()
# 
#             # Fetch measured centroid from the camera actor
#             cmdString = "centroid expTime=%0.1f" % (expTime)
#             cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
#                                           forUserCmd=cmd, timeLim=expTime+5.0)
#             if cmdVar.didFail:
#                 cmd.fail('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))
#             #    return
#             # Encoding will be encapsulated.
#             rawCentroids = self.actor.models['mcs'].keyVarDict['centroidsChunk'][0]
#             centroids = numpy.fromstring(base64.b64decode(rawCentroids), dtype='f4').reshape(2400,2)
#             times[i,1] = time.time()
# 
#             # Command the actuators to move.
#             cmdString = 'moveTo chunk=%s' % (base64.b64encode(targetPos.tostring()))
#             cmdVar = self.actor.cmdr.call(actor='mps', cmdStr=cmdString,
#                                           forUserCmd=cmd, timeLim=5.0)
#             if cmdVar.didFail:
#                 cmd.fail('text=%s' % (qstr('Failed to move with %s' % (cmdString))))
#                 return
#             times[i,2] = time.time()
# 
#             cmdVar = self.actor.cmdr.call(actor='mps', cmdStr="ping",
#                                           forUserCmd=cmd, timeLim=5.0)
#             if cmdVar.didFail:
#                 cmd.fail('text=%s' % (qstr('Failed to ping')))
#                 return
#             times[i,3] = time.time()
# 
#         for i, itimes in enumerate(times):
#             cmd.inform('text="dt[%d]=%0.4f, %0.4f, %0.4f"' % (i+1, 
#                                                               itimes[1]-itimes[0],
#                                                               itimes[2]-itimes[1],
#                                                               itimes[3]-itimes[2],
#                                                               ))
#         cmd.inform('text="dt[mean]=%0.4f, %0.4f, %0.4f"' % ((times[:,1]-times[:,0]).sum()/cnt,
#                                                             (times[:,2]-times[:,1]).sum()/cnt,
#                                                             (times[:,3]-times[:,2]).sum()/cnt))
#         cmd.inform('text="dt[max]=%0.4f, %0.4f, %0.4f"' % ((times[:,1]-times[:,0]).max(),
#                                                            (times[:,2]-times[:,1]).max(),
#                                                            (times[:,3]-times[:,2]).max()))
        cmd.finish("text='Testing loop finished.'")

