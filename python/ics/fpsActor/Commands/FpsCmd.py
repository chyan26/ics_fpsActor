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
from ics.fpsActor import fpsFunction as fpstool
#import mcsActor.Visualization.mcsRoutines as mcs
#import mcsActor.Visualization.fpsRoutines as fps



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
            ('calculateBoresight', '[<startFrame>] [<endFrame>]', self.calculateBoresight),
            ('testCamera', '[<cnt>] [<expTime>] [@noCentroids]', self.testCamera),
            ('testLoop', '[<cnt>] [<expTime>] [<visit>]', self.testLoop),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("startFrame", types.Int(), help="starting frame for "
                                                        "boresight calculating"),
                                       keys.Key("endFrame", types.Int(), help="ending frame for "
                                                        "boresight calculating"),
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

    def _findHomes(self, baseData,poolData):
    
        """
        Do nearest neighbour matching on a set of centroids (ie, home position case).

        """
        dist_thres = 5.0
        orix=[]
        oriy=[]
        px = []
        py = []
        cx = []
        cy = []
        fiberid = []
        mcsid = []
        
        for index, row in baseData.iterrows():
            dist=np.sqrt((row['x']-poolData['pfix'])**2 + (row['y']-poolData['pfiy'])**2)
            ind=pd.Series.idxmin(dist)
            
            fiberid.append(row['fiberID'])
            orix.append(row['x'])
            oriy.append(row['y'])
            if(min(dist) < dist_thres):
                px.append(poolData['pfix'][ind])
                py.append(poolData['pfiy'][ind])
                cx.append(poolData['centroidx'][ind])
                cy.append(poolData['centroidy'][ind])
                mcsid.append(poolData['mcsId'][ind])
                #otherwise set values to NaN
            else:
                px.append(np.nan)
                py.append(np.nan)
                cx.append(np.nan)
                cy.append(np.nan)
                mcsid.append(np.nan)
        d = {'fiberId': np.array(fiberid).astype('int32'), 'mcsId': np.array(mcsid).astype('int32'), 
            'orix': np.array(orix).astype('float'), 'oriy': np.array(oriy).astype('float'), 
            'pfix': np.array(px).astype('float'), 'pfiy': np.array(py).astype('float'), 
                'centroidx': np.array(cx).astype('float'), 'centroidy': np.array(cy).astype('float')}
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

        mcsData['pfix'] = xyout[0]
        mcsData['pfiy'] = xyout[1]


        #d = {'ffID': np.arange(len(xyout[0])), 'pfix': xyout[0], 'pfiy': xyout[1]}
        #transPos=pd.DataFrame(data=d) 

        match = self._findHomes(ffData, mcsData)

        
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
        rotCent=[[4471],[2873]]

        telInform = self.nv.readTelescopeInform(frameId)
        za = 90-telInform['azi']
        inr = telInform['instrot']
        inr=inr-180
        if(inr < 0):
            inr=inr+360

        mcsData = self.nv.readCentroid(frameId, moveId)
        sfData = self.nv.readCobraConfig()

        sfData['x']-=offset[0]
        sfData['y']-=offset[1]

        
        #correect input format
        xyin=np.array([mcsData['centroidx'],mcsData['centroidy']])

        #call the routine
        xyout=CoordTransp.CoordinateTransform(xyin,za,'mcs_pfi',inr=inr,cent=rotCent)

        mcsData['pfix'] = xyout[0]
        mcsData['pfiy'] = xyout[1]

        pts2=np.zeros((1,len(xyout[1]),2))

        pts2[0,:,0]=xyout[0]
        pts2[0,:,1]=xyout[1]


        afCor=cv2.transform(pts2,self.tranMatrix['affineCoeff'])

        mcsData['pfix'] = afCor[0,:,0]
        mcsData['pfiy'] = afCor[0,:,1]

        match = self._findHomes(sfData, mcsData)

        match['dx'] = match['orix'] - match['pfix']
        match['dy'] = match['oriy'] - match['pfiy']

        return match

    def calculateBoresight(self, cmd):
        """ Function for calculating the rotation center """
        cmdKeys = cmd.cmd.keywords

        startFrame = cmdKeys["startFrame"].values[0] 
        endFrame = cmdKeys["endFrame"].values[0]

        nframes=  endFrame - startFrame +1
        frameid = (np.arange(nframes)+startFrame)

        xCorner=[]
        yCorner=[]

        for i in frameid:
            mcsData = self.nv.readCentroid(i, 1)
            x=mcsData['centroidx']
            y=mcsData['centroidy']  
            
            x0,x1,y0,y1=fpstool.getCorners(x,y)
            xCorner.append(x0)
            yCorner.append(y0)

        xCorner=np.array(xCorner)
        yCorner=np.array(yCorner)

        #fig,ax=plt.subplots()
        #ax.plot(xCorner,yCorner,'dg')

        coords=[xCorner,yCorner]
        xc,yc,r,_= fpstool.least_squares_circle(xCorner,yCorner)
    
        cmd.finish(f'mcsBoresight={x:0.4f},{y:0.4f}')


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
                cmd.inform('text="Getting Affine coefficients from fiducial fibers." ')
                self.getAEfromFF(cmd, frameId)
            
            cmd.inform('text="Apply Affine coefficients to science fibers." ')
            match = self.applyAEonCobra(cmd, frameId)
            self.nv.writeCobraConfig(match,frameId)


        cmd.finish("text='Testing loop finished.'")

