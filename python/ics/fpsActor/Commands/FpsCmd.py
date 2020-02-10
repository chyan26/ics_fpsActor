import pathlib
import sys

import numpy as np
import psycopg2
import io
import pandas as pd
import cv2
from pfs.utils.coordinates import CoordTransp
from pfs.utils.coordinates import DistortionCoefficients

import logging

import opscore.protocols.keys as keys
import opscore.protocols.types as types

from opscore.utility.qstr import qstr

from ics.fpsActor import fpsState
from ics.fpsActor import najaVenator
from ics.fpsActor import fpsFunction as fpstool
#import mcsActor.Visualization.mcsRoutines as mcs
#import mcsActor.Visualization.fpsRoutines as fps

from ics.cobraCharmer.procedures.moduleTest import calculation
from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer.utils import butler
from ics.cobraCharmer.fpgaState import fpgaState
from ics.cobraCharmer import cobraState



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
            ('makeMotorMap','[<arm>] [<stepsize>]',self.makMotorMap)
            ('calculateBoresight', '[<startFrame>] [<endFrame>]', self.calculateBoresight),
            ('testCamera', '[<cnt>] [<expTime>] [@noCentroids]', self.testCamera),
            ('testLoop', '[<cnt>] [<expTime>] [<visit>]', self.testLoop),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("arm", types.String(), help="corbra arm"),
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

        self.logger = logging.getLogger('fps')
        self.logger.setLevel(logging.INFO)

        """ Init module 1 cobras """

        # NO, not 1!! Pass in moduleName, etc. -- CPL
        reload(pfiControl)
        self.allCobras = np.array(pfiControl.PFI.allocateCobraModule(1))
        self.fpgaHost = fpgaHost
        self.xml = xml
        self.brokens = brokens
        #self.camSplit = camSplit

        # partition module 1 cobras into odd and even sets
        moduleCobras = {}
        for group in 1, 2:
            cm = range(group, 58, 2)
            mod = [1]*len(cm)
            moduleCobras[group] = pfiControl.PFI.allocateCobraList(zip(mod, cm))
        self.oddCobras = moduleCobras[1]
        self.evenCobras = moduleCobras[2]

        self.pfi = None

        self.thetaCenter = None
        self.thetaCCWHome = None
        self.thetaCWHome = None
        self.phiCenter = None
        self.phiCCWHome = None
        self.phiCWHome = None

        self.setBrokenCobras(self.brokens)

    def _connect(self):
        # Initializing COBRA module
        self.pfi = pfiControl.PFI(fpgaHost=self.fpgaHost,
                                  doLoadModel=False,
                                  logDir=self.runManager.logDir)
        self.pfi.loadModel(self.xml)
        self.pfi.setFreq()

        # initialize cameras
        #self.cam = camera.cameraFactory(name='rmod',doClear=True, runManager=self.runManager)

        # init calculation library
        self.cal = calculation.Calculation(self.xml, None, None)

        # define the broken/good cobras
        self.setBrokenCobras(self.brokens)

    def setBrokenCobras(self, brokens=None):
        """ define the broken/good cobras """
        if brokens is None:
            brokens = []
        visibles = [e for e in range(1, 58) if e not in brokens]
        self.badIdx = np.array(brokens) - 1
        self.goodIdx = np.array(visibles) - 1
        self.badCobras = np.array(self.getCobras(self.badIdx))
        self.goodCobras = np.array(self.getCobras(self.goodIdx))

        if hasattr(self, 'cal'):
            self.cal.setBrokenCobras(brokens)

    def exposeAndExtractPositions(self, name=None, guess=None, tolerance=None):
        """ Take an exposure, measure centroids, match to cobras, save info.

        Args
        ----
        name : `str`
           Additional name for saved image file. File _always_ gets PFS-compliant name.
        guess : `ndarray` of complex coordinates
           Where to center searches. By default uses the cobra center.
        tolerance : `float`
           Additional factor to scale search region by. 1 = cobra radius (phi+theta)

        Returns
        -------
        positions : `ndarray` of complex
           The measured positions of our cobras.
           If no matching spot found, return the cobra center.

        Note
        ----
        Not at all convinced that we should return anything if no matching spot found.

        """
        cmd.inform(f'text="executing exposeAndExtractPositions"')

        frameId = self._mcsExpose(cmd, frameId, expTime=expTime, doCentroid=True)
        
        centroids, filename, bkgd = self.cam.expose(name)
        positions, indexMap = self.cal.matchPositions(centroids, guess=guess, tolerance=tolerance)
        self._saveMoveTable(filename.stem, positions, indexMap)

        return positions        
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

    def makeMotorMap(self,cmd,arm):
        """ Making motor map. """

        cmd.finish(f'Motor map sequence finished')


    def _makePhiMotorMap(self, cmd, newXml, repeat=3, steps=100,
            totalSteps=5000, fast=False, phiOnTime=None, updateGeometry=False,
            limitOnTime=0.08, resetScaling=True, delta=np.deg2rad(5.0), fromHome=False
        ):
        """ generate phi motor maps, it accepts custom phiOnTIme parameter.
            it assumes that theta arms have been move to up/down positions to avoid collision
            if phiOnTime is not None, fast parameter is ignored. Otherwise use fast/slow ontime

            Example:
                makePhiMotorMap(xml, path, fast=True)             // update fast motor maps
                makePhiMotorMap(xml, path, fast=False)            // update slow motor maps
                makePhiMotorMap(xml, path, phiOnTime=0.06)        // motor maps for on-time=0.06
        """
        self._connect()
        defaultOnTimeFast = deepcopy([self.pfi.calibModel.motorOntimeFwd2,
                                      self.pfi.calibModel.motorOntimeRev2])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd2,
                                      self.pfi.calibModel.motorOntimeSlowRev2])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        fastOnTime = [np.full(57, limitOnTime)] * 2
        if phiOnTime is not None:
            if np.isscalar(phiOnTime):
                slowOnTime = [np.full(57, phiOnTime)] * 2
            else:
                slowOnTime = phiOnTime
        elif fast:
            slowOnTime = defaultOnTimeFast
        else:
            slowOnTime = defaultOnTimeSlow

        # update ontimes for test
        self.pfi.calibModel.updateOntimes(phiFwd=fastOnTime[0], phiRev=fastOnTime[1], fast=True)
        self.pfi.calibModel.updateOntimes(phiFwd=slowOnTime[0], phiRev=slowOnTime[1], fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        phiFW = np.zeros((57, repeat, iteration+1), dtype=complex)
        phiRV = np.zeros((57, repeat, iteration+1), dtype=complex)

        if resetScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='phi')

        # record the phi movements
        dataPath = self.runManager.dataDir
        self.logger.info(f'phi home {-totalSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, 0, -totalSteps)  # default is fast
        for n in range(repeat):
            self.cam.resetStack(f'phiForwardStack{n}.fits')

            # forward phi motor maps
            phiFW[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'phiBegin{n}.fits')

            notdoneMask = np.zeros(len(phiFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi forward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, (k+1)*steps, phiFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, steps, phiFast=False)
                phiFW[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'phiForward{n}N{k}.fits',
                                                                             guess=phiFW[self.goodIdx, n, k])
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -(k+1)*steps)

                doneMask, lastAngles = self.phiFWDone(phiFW, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    phiFW[self.goodIdx, n, k+2:] = phiFW[self.goodIdx, n, k+1][:,None]
                    break
            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not finish:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    self.logger.warn(f'  {str(c)}: {np.round(d, 2)}')

            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} phi forward {totalSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, totalSteps)  # fast to limit

            # reverse phi motor maps
            self.cam.resetStack(f'phiReverseStack{n}.fits')
            phiRV[self.goodIdx, n, 0] = self.exposeAndExtractPositions(f'phiEnd{n}.fits',
                                                                       guess=phiFW[self.goodIdx, n, iteration])
            notdoneMask = np.zeros(len(phiRV), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi backward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -(k+1)*steps, phiFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, -steps, phiFast=False)
                phiRV[self.goodIdx, n, k+1] = self.exposeAndExtractPositions(f'phiReverse{n}N{k}.fits',
                                                                             guess=phiRV[self.goodIdx, n, k])
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], 0, (k+1)*steps)
                doneMask, lastAngles = self.phiRVDone(phiRV, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    phiRV[self.goodIdx, n, k+2:] = phiRV[self.goodIdx, n, k+1][:,None]
                    break

            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not finish:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    self.logger.warn(f'  {str(c)}: {np.round(d, 2)}')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} phi reverse {-totalSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, -totalSteps)  # fast to limit
        self.cam.resetStack()

        # save calculation result
        np.save(dataPath / 'phiFW', phiFW)
        np.save(dataPath / 'phiRV', phiRV)

        # calculate centers and phi angles
        phiCenter, phiRadius, phiAngFW, phiAngRV, badRange = self.cal.phiCenterAngles(phiFW, phiRV)
        np.save(dataPath / 'phiCenter', phiCenter)
        np.save(dataPath / 'phiRadius', phiRadius)
        np.save(dataPath / 'phiAngFW', phiAngFW)
        np.save(dataPath / 'phiAngRV', phiAngRV)
        np.save(dataPath / 'badRange', badRange)

        # calculate average speeds
        phiSpeedFW, phiSpeedRV = self.cal.speed(phiAngFW, phiAngRV, steps, delta)
        np.save(dataPath / 'phiSpeedFW', phiSpeedFW)
        np.save(dataPath / 'phiSpeedRV', phiSpeedRV)

        # calculate motor maps by Johannes weighting
        if fromHome:
            phiMMFW, phiMMRV, bad = self.cal.motorMaps2(phiAngFW, phiAngRV, steps, delta)
        else:
            phiMMFW, phiMMRV, bad = self.cal.motorMaps(phiAngFW, phiAngRV, steps, delta)
        bad[badRange] = True
        np.save(dataPath / 'phiMMFW', phiMMFW)
        np.save(dataPath / 'phiMMRV', phiMMRV)
        np.save(dataPath / 'bad', np.where(bad)[0])

        # calculate motor maps by average speeds
        #phiMMFW2, phiMMRV2, bad2 = self.cal.motorMaps2(phiAngFW, phiAngRV, steps, delta)
        #bad2[badRange] = True
        #np.save(dataPath / 'phiMMFW2', phiMMFW2)
        #np.save(dataPath / 'phiMMRV2', phiMMRV2)
        #np.save(dataPath / 'bad2', np.where(bad2)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updatePhiMotorMaps(phiMMFW, phiMMRV, bad, slow)
        if phiOnTime is not None:
            if np.isscalar(phiOnTime):
                onTime = np.full(57, phiOnTime)
                self.cal.calibModel.updateOntimes(phiFwd=onTime, phiRev=onTime, fast=fast)
            else:
                self.cal.calibModel.updateOntimes(phiFwd=phiOnTime[0], phiRev=phiOnTime[1], fast=fast)
        if updateGeometry:
            self.cal.calibModel.updateGeometry(centers=phiCenter, phiArms=phiRadius)
        self.cal.calibModel.createCalibrationFile(self.runManager.outputDir / newXml, name='phiModel')

        # restore default setting ( really? why? CPL )
        # self.cal.restoreConfig()
        # self.pfi.loadModel(self.xml)

        self.setPhiGeometryFromRun(self.runManager.runDir, onlyIfClear=True)
        return self.runManager.runDir

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

        coords=[xCorner,yCorner]
        xc,yc,r,_= fpstool.least_squares_circle(xCorner,yCorner)
    
        data = {'visitid':startFrame, 'xc':xc, 'yc': yc}
        
        self.nv.writeBoresightTable(data)
        
        cmd.finish(f'mcsBoresight={xc:0.4f},{yc:0.4f}')


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

