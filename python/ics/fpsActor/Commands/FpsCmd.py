import pathlib
import sys
from importlib import reload

from astropy.io import fits

import numpy as np
import psycopg2
import io
import pandas as pd
import cv2
from pfs.utils.coordinates import CoordTransp
from pfs.utils.coordinates import DistortionCoefficients
import pathlib

from copy import deepcopy


import logging
import time
import opscore.protocols.keys as keys
import opscore.protocols.types as types

from opscore.utility.qstr import qstr

from ics.fpsActor import fpsState
from ics.fpsActor import najaVenator
from ics.fpsActor import fpsFunction as fpstool
from ics.fpsActor.utils import butler
#import mcsActor.Visualization.mcsRoutines as mcs
#import mcsActor.Visualization.fpsRoutines as fps

from procedures.moduleTest import calculation
reload(calculation)
from procedures.moduleTest.speedModel import SpeedModel

from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer import pfiDesign
#from ics.cobraCharmer.utils import butler
from ics.cobraCharmer.fpgaState import fpgaState
from ics.cobraCharmer import cobraState

from procedures.moduleTest import cobraCoach
from procedures.moduleTest import engineer as eng

reload(pfiControl)
reload(cobraCoach)
    
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
            ('reset', '[<mask>]', self.reset),
            ('power', '[<mask>]', self.power),
            ('powerOn', '', self.powerOn),
            ('powerOff', '', self.powerOff),
            ('diag', '', self.diag),
            ('connect', '', self.connect),
            ('loadDesign', '<id>', self.loadDesign),
            ('loadModel', '<xml>', self.loadModel),
            ('movePhiToAngle', '<angle> <iteration>', self.movePhiToAngle),
            ('movePhiToHome','', self.movePhiToHome),
            ('moveToHome','@(phi|theta|all)', self.moveToHome),
            ('setGeometry', '@(phi|theta) <runDir>', self.setGeometry),            
            ('moveToDesign', '', self.moveToDesign),
            ('gotoSafeFromPhi60','',self.gotoSafeFromPhi60),
            ('gotoVerticalFromPhi60','',self.gotoVerticalFromPhi60),
            ('makeMotorMap','@(phi|theta) <stepsize> <repeat> [@slowOnly]',self.makeMotorMap),
            ('angleConverge','@(phi|theta) <angleTargets> [@doGeometry]',self.angleConverge),
            ('targetConverge','@(ontime|speed) <totalTargets> <maxsteps>',self.targetConverge),
            ('motorOntimeSearch','@(phi|theta)',self.motorOntimeSearch),
            ('calculateBoresight', '[<startFrame>] [<endFrame>]', self.calculateBoresight),
            ('testCamera', '[<cnt>] [<expTime>] [@noCentroids]', self.testCamera),
            ('testLoop', '[<cnt>] [<expTime>] [<visit>]', self.testLoop),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("angle", types.Int(), help="arm angle"),
                                        keys.Key("stepsize", types.Int(), help="step size of motor"),
                                        keys.Key("repeat", types.Int(), help="number of iteration for motor "
                                                     "map generation"),
                                        keys.Key("angleTargets", types.Int(), 
                                                        help="Target number for angle convergence"),
                                        keys.Key("totalTargets", types.Int(), 
                                                        help="Target number for 2D convergence"),
                                        keys.Key("maxsteps", types.Int(), 
                                                        help="Maximum step number for 2D convergence test"),                
                                        keys.Key("xml", types.String(), help="XML filename"),
                                        keys.Key("runDir", types.String(), help="Directory of run data"),
                                        keys.Key("startFrame", types.Int(), help="starting frame for "
                                                        "boresight calculating"),
                                        keys.Key("endFrame", types.Int(), help="ending frame for "
                                                        "boresight calculating"),
                                        keys.Key("visit", types.Int(), help="PFS visit to use"),
                                        keys.Key("iteration", types.Int(), help="Interation number"),
                                        keys.Key("id", types.Long(),
                                                 help=("fpsDesignId for the field, "
                                                       "which defines the fiber positions")),
                                        keys.Key("mask", types.Int(), help="mask for power and/or reset"),
                                        keys.Key("expTime", types.Float(), 
                                                 help="Seconds for exposure"))

        self.logger = logging.getLogger('fps')
        self.logger.setLevel(logging.INFO)


        self.cc = None
        

    def loadModel(self, cmd):
        
        xml = cmd.cmd.keywords['xml'].values[0]
        self.logger.info(f'Input XML file = {xml}')
        self.xml = pathlib.Path(xml)

        mod = 'ALL'
        self.cc = cobraCoach.CobraCoach('fpga', loadModel=False, actor=self.actor, cmd=cmd)
        self.cc.loadModel(file=pathlib.Path(self.xml))
        #self.cc.connect()
        eng.setCobraCoach(self.cc)

        #self.cc = cc
        #eng.setPhiMode()
        cmd.finish(f"text='Loading model = {self.xml}'")


    def _fpsInit(self):
        """ Init module 1 cobras """

        self.xml = None

        # partition module 1 cobras into odd and even sets
        #moduleCobras = {}
        #for group in 1, 2:
        #    cm = range(group, 58, 2)
        #    mod = [1]*len(cm)
        #    moduleCobras[group] = pfiControl.PFI.allocateCobraList(zip(mod, cm))
        #self.oddCobras = moduleCobras[1]
        #self.evenCobras = moduleCobras[2]


    def _simpleConnect(self):
        self.runManager.newRun()
        reload(pfiControl)
        # Initializing COBRA module
        self.pfi = pfiControl.PFI(fpgaHost=self.fpgaHost,
                                  doLoadModel=False,
                                  logDir=self.runManager.logDir)
        
        self.modules = [f'SC{m:02d}' for m in range(1,43)]
        self.modFiles = [butler.mapPathForModule(mn, version='final') for mn in self.modules]
        self.pfi.loadModel(self.modFiles)

        if self.xml is None:
            newModel = pfiDesign.PFIDesign(pathlib.Path('/home/pfs/mhs/devel/ics_cobraCharmer/procedures/moduleTest/allModule.xml'))
        else:
            newModel = pfiDesign.PFIDesign(self.xml)
        self.pfi.calibModel = newModel
        
        self.allCobras = np.array(self.pfi.getAllDefinedCobras())
        self.nCobras = len(self.allCobras)
        
    def _connect(self):
        self.runManager.newRun()
        reload(pfiControl)
        # Initializing COBRA module
        self.pfi = pfiControl.PFI(fpgaHost=self.fpgaHost,
                                  doLoadModel=False,
                                  logDir=self.runManager.logDir)

        # It takes > 10s to power down.
        # Power on is pretty much instant, but requires a reset as well.
        # Two resets is the same a power on and a reset.
        if True:
            self.pfi.diag()
            self.pfi.power(0)
            time.sleep(1)
            self.pfi.reset()
            time.sleep(1)
            self.pfi.diag()
        else:
            self.pfi.power(0x3f)
            time.sleep(1)
            self.pfi.power(0x3f)
            time.sleep(1)
            self.pfi.reset()
            time.sleep(1)
            self.pfi.diag()
            time.sleep(1)
            self.pfi.power(0)
            time.sleep(1)
            self.pfi.power(0)
            time.sleep(1)
            self.pfi.reset()
            time.sleep(1)
            self.pfi.diag()
            
        self.modules = [f'SC{m:02d}' for m in range(1,43)]
        self.modFiles = [butler.mapPathForModule(mn, version='final') for mn in self.modules]

        #self.pfi.loadModel(self.modFiles)
        
        
        if self.xml is None:
            newModel = pfiDesign.PFIDesign(pathlib.Path('/home/pfs/mhs/devel/ics_cobraCharmer/procedures/moduleTest/allModule.xml'))
        else:
            newModel = pfiDesign.PFIDesign(self.xml)
        
        self.pfi.calibModel = newModel
        self.pfi.calibModel.fixModuleIds()
        
        self.logger.info(f'Loading XML from {self.xml}')

        self.allCobras = np.array(self.pfi.getAllDefinedCobras())
        self.nCobras = len(self.allCobras)

        self.pfi.setFreq(self.allCobras)

        # initialize cameras
        #self.cam = camera.cameraFactory(name='rmod',doClear=True, runManager=self.runManager)

        # init calculation library
        self.cal = calculation.Calculation(self.pfi.calibModel, None, None)

        # define the broken/good cobras

        self.setBrokenCobras(brokens=self.brokens)
        self.logger.info(f'Setting broken fibers =  {self.brokens}')

    def setBrokenCobras(self, brokens=None):
        """ define the broken/good cobras """
        if brokens is None:
            brokens = []
            brokens = [self.pfi.calibModel.findCobraByModuleAndPositioner(3,25)+1,
                        self.pfi.calibModel.findCobraByModuleAndPositioner(15,1)+1,
                        self.pfi.calibModel.findCobraByModuleAndPositioner(15,23)+1,
                        self.pfi.calibModel.findCobraByModuleAndPositioner(15,55)+1,
                        self.pfi.calibModel.findCobraByModuleAndPositioner(17,37)+1,
                        self.pfi.calibModel.findCobraByModuleAndPositioner(29,57)+1,
                        #self.pfi.calibModel..findCobraByModuleAndPositioner(30,1)+1,
                        self.pfi.calibModel.findCobraByModuleAndPositioner(31,14)+1,
                        self.pfi.calibModel.findCobraByModuleAndPositioner(34,1)+1]
        else:
            brokens = brokens
        #brokens = [self.pfi.calibModel.findCobraByModuleAndPositioner(3,25)+1,
        #            self.pfi.calibModel.findCobraByModuleAndPositioner(15,1)+1,
        #            self.pfi.calibModel.findCobraByModuleAndPositioner(15,23)+1,
        #            self.pfi.calibModel.findCobraByModuleAndPositioner(15,55)+1,
        #            self.pfi.calibModel.findCobraByModuleAndPositioner(17,37)+1,
        #            self.pfi.calibModel.findCobraByModuleAndPositioner(29,57)+1,
                    #self.pfi.calibModel..findCobraByModuleAndPositioner(30,1)+1,
        #            self.pfi.calibModel.findCobraByModuleAndPositioner(31,14)+1]
                    #self.pfi.calibModel.findCobraByModuleAndPositioner(34,1)+1]

        brokens = [self.pfi.calibModel.findCobraByModuleAndPositioner(1,47)+1,
                    #self.pfi.calibModel.findCobraByModuleAndPositioner(3,25)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(4,22)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(7,19)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(7,5)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(14,13)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,23)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,55)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(17,37)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(21,10)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(22,11)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(22,13)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(27,38)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(28,41)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(29,57)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(31,14)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(34,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(34,22)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(29,41)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(33,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(33,12)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(37,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(42,15)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(42,43)+1,
                    ]
        
        #brokens.append([189,290,447,559,589,607,809,1171,1252,1321,2033,2120,2136,2149,2163])
        self.brokens = brokens   
        visibles = [e for e in range(1, self.nCobras+1) if e not in brokens]
        # brokens = np.array([])
        # for f in [3,4,7,8,11,12,13,14,17,18,21,22,23,24,25,26,31,32,35,36,39,40,41,42]:
        #     brokens = np.append(brokens,self.pfi.calibModel.findCobraByModule(f))
        #     brokens = (brokens+1).astype('int').tolist()
        # visibles = [e for e in range(1, self.nCobras+1) if e not in brokens]
        self.brokens = brokens 
        
        self.badIdx = np.array(brokens) - 1
        
        self.goodIdx = np.array(visibles) - 1
        
        
        #self.goodIdx = np.arange(1197,1596)
        #self.badIdx = [e for e in range(0,self.nCobras) if e not in self.goodIdx]
        
        #self.badCobras = np.array(self.getCobras(self.badIdx))
        #self.goodCobras = np.array(self.getCobras(self.goodIdx))
        if len(self.badIdx) is not 0:
            self.badCobras = np.array(self.allCobras)[self.badIdx]
        self.goodCobras = np.array(self.allCobras)[self.goodIdx]

        if hasattr(self, 'cal'):
            self.cal.setBrokenCobras(brokens)

    def getPositionsForFrame(self, frameId):
        mcsData = self.nv.readCentroid(frameId)
        self.logger.info(f'mcs data {mcsData.shape[0]}')
        centroids = {'x':mcsData['centroidx'].values.astype('float'),
                     'y':mcsData['centroidy'].values.astype('float')}
        return centroids

    def exposeAndExtractPositions(self, cmd, name=None, guess=None, tolerance=None):
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
        ret = self.actor.cmdr.call(actor='gen2', cmdStr='getVisit', timeLim=10.0)
        visit = self.actor.models['gen2'].keyVarDict['visit'].valueList[0]
        frameId = visit * 100
        
        datapath, filename, frameId = self._mcsExpose(cmd, expTime=0.5, doCentroid=True)
        self.logger.info(f'path = {datapath} filename = {filename}, frame ID = {frameId}')

        centroids =  self.getPositionsForFrame(frameId)

        positions, indexMap = self.cal.matchPositions(centroids, guess=guess, tolerance=tolerance)
        
        self.logger.info(f'Matched positions = {len(positions)}')
        #print(positions)
        #self._saveMoveTable(filename.stem, positions, indexMap)
        #positions = np.zeros(len(self.goodIdx))
        self.dataPath = datapath
        return positions, filename

    @staticmethod
    def dPhiAngle(target, source, doWrap=False, doAbs=False):
        d = np.atleast_1d(target - source)

        if doAbs:
            d[d<0] += 2*np.pi
            d[d>=2*np.pi] -= 2*np.pi

            return d

        if doWrap:
            lim = np.pi
        else:
            lim = 2*np.pi

        # d[d > lim] -= 2*np.pi
        d[d < -lim] += 2*np.pi

        return d
    @staticmethod
    def _fullAngle(toPos, fromPos=None):
        """ Return ang of vector, 0..2pi """
        if fromPos is None:
            fromPos = 0+0j
        a = np.angle(toPos - fromPos)
        if np.isscalar(a):
            if a < 0:
                a += 2*np.pi
            if a >= 2*np.pi:
                a -= 2*np.pi
        else:
            a[a<0] += 2*np.pi
            a[a>=2*np.pi] -= 2*np.pi

        return a

    def _saveMoveTable(self, expId, positions, indexMap):
        """ Save cobra move and spot information to a file.

        Args
        ----
        expId : `str`
          An exposure identifier. We want "PFxxNNNNNNNN".
        positions : `ndarray` of complex coordinates.
          What the matcher thinks is the cobra position.
        indexMap : `ndarray` of `int`
          For each of our cobras, the index of measured spot

        """
        moveTable = np.zeros(len(positions), dtype=self.movesDtype)
        moveTable['expId'][:] = expId
        if len(positions) != len(self.goodCobras):
            raise RuntimeError("Craig is confused about cobra lists")

        for pos_i, pos in enumerate(positions):
            cobraInfo = self.goodCobras[pos_i]
            cobraNum = self.pfi.calibModel.findCobraByModuleAndPositioner(cobraInfo.module,
                                                                          cobraInfo.cobraNum)
            moveInfo = fpgaState.cobraLastMove(cobraInfo)

            phiMotorId = cobraState.mapId(cobraNum, 'phi', 'ccw' if moveInfo['phiSteps'] < 0 else 'cw')
            thetaMotorId = cobraState.mapId(cobraNum, 'theta', 'ccw' if moveInfo['thetaSteps'] < 0 else 'cw')
            phiScale = self.pfi.ontimeScales.get(phiMotorId, 1.0)
            thetaScale = self.pfi.ontimeScales.get(thetaMotorId, 1.0)
            moveTable['spotId'][pos_i] = indexMap[pos_i]
            moveTable['module'][pos_i] = cobraInfo.module
            moveTable['cobra'][pos_i] = cobraInfo.cobraNum
            for field in ('phiSteps', 'phiOntime',
                          'thetaSteps', 'thetaOntime'):
                moveTable[field][pos_i] = moveInfo[field]
            moveTable['thetaOntimeScale'] = thetaScale
            moveTable['phiOntimeScale'] = phiScale

        movesPath = self.runManager.outputDir / "moves.npz"
        self.logger.debug(f'saving {len(moveTable)} moves to {movesPath}')
        if movesPath.exists():
            with open(movesPath, 'rb') as f:
                oldMoves = np.load(f)['moves']
            allMoves = np.concatenate([oldMoves, moveTable])
        else:
            allMoves = moveTable

        with open(movesPath, 'wb') as f:
            np.savez_compressed(f, moves=allMoves)


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
    
    def XXloadModel(self, cmd):
        
        xml = cmd.cmd.keywords['xml'].values[0]
        self.logger.info(f'Input XML file = {xml}')
        self.xml = pathlib.Path(xml)

        cmd.finish(f"text='Loading model = {self.xml}'")

    def _loadPfsDesign(self, cmd, designId):
        """ Return the pfsDesign for the given pfsDesignId. """

        cmd.warn(f'text="have a pfsDesignId={designId:#016x}, but do not know how to fetch it yet."')

        return None

    def reset(self, cmd):
        """Send the FPGA POWer command with a reset mask. """

        cmdKeys = cmd.cmd.keywords
        resetMask = cmdKeys['mask'].values[0] if 'mask' in cmdKeys else 0x3f
        
        self.pfi.reset(resetMask)
        time.sleep(1)
        res = self.pfi.diag()
        cmd.finish(f'text="diag = {res}"')

    def power(self, cmd):
        """Send the FPGA POWer command with a sector mask. """

        cmdKeys = cmd.cmd.keywords
        powerMask = cmdKeys['mask'].values[0] if 'mask' in cmdKeys else 0x0
        
        self.pfi.power(powerMask)
        time.sleep(1)
        res = self.pfi.diag()
        cmd.finish(f'text="diag = {res}"')

    def powerOn(self, cmd):
        """Do what is required to power on all PFI sectors. """

        cmdKeys = cmd.cmd.keywords
        
        self.pfi.power(0x0)
        time.sleep(1)
        self.pfi.reset()
        time.sleep(1)
        res = self.pfi.diag()
        cmd.finish(f'text="diag = {res}"')

    def powerOff(self, cmd):
        """Do what is required to power off all PFI sectors """

        cmdKeys = cmd.cmd.keywords
        
        self.pfi.power(0x23f)
        time.sleep(10)
        res = self.pfi.diag()
        cmd.finish(f'text="diag = {res}"')

    def diag(self, cmd):
        """Read the FPGA sector inventory"""

        cmdKeys = cmd.cmd.keywords
        
        res = self.pfi.diag()
        cmd.finish(f'text="diag = {res}"')

    def connect(self, cmd):
        """Connect to the FPGA and set up output tree. """

        cmdKeys = cmd.cmd.keywords
        
        self._simpleConnect()
        time.sleep(2)
        
        res = self.pfi.diag()
        cmd.finish(f'text="diag = {res}"')
        
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

    def getCobras(self, cobs):
        # cobs is 0-indexed list
        if cobs is None:
            cobs = np.arange(len(self.allCobras))

        # assumes module == 1 XXX
        return np.array(pfiControl.PFI.allocateCobraList(zip(np.full(len(cobs), 1), np.array(cobs) + 1)))
    
    def _mapDone(self, centers, points, limits, n, k,
                 needAtEnd=4, closeEnough=np.deg2rad(1),
                 limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the axis limit.

        See thetaFWDone.
        """

        if centers is None or limits is None or k+1 < needAtEnd:
            return None, None

        lastAngles = np.angle(points[:,n,k-needAtEnd+1:k+1] - centers[:,None])
        atEnd = np.abs(lastAngles[:,-1] - limits) <= limitTolerance
        endDiff = np.abs(np.diff(lastAngles, axis=1))
        stable = np.all(endDiff <= closeEnough, axis=1)

        # Diagnostic: return the needAtEnd distances from the limit.
        anglesFromEnd = lastAngles - limits[:,None]

        return atEnd & stable, anglesFromEnd

    def _thetaFWDone(self, thetas, n, k, needAtEnd=4,
                    closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the FW theta limit.

        Args
        ----
        thetas : `np.array` of `complex`
          2 or 3d array of measured positions.
          0th axis is cobra, last axis is iteration
        n, k : integer
          the iteration we just made.
        needAtEnd : integer
          how many iterations we require to be at the same position
        closeEnough : radians
          how close the last needAtEnd point must be to each other.
        limitTolerance : radians
          how close to the known FW limit the last (kth) point must be.

        Returns
        -------
        doneMask : array of `bool`
          True for the cobras which are at the FW limit.
        endDiffs : array of radians
          The last `needAtEnd` angles to the limit
        """

        return self._mapDone(self.thetaCenter, thetas, self.thetaCWHome, n, k,
                             needAtEnd=needAtEnd, closeEnough=closeEnough,
                             limitTolerance=limitTolerance)

    def _thetaRVDone(self, thetas, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the RV theta limit.

        See `thetaFWDone`
        """
        return self._mapDone(self.thetaCenter, thetas, self.thetaCCWHome, n, k,
                             needAtEnd=needAtEnd, closeEnough=closeEnough,
                             limitTolerance=limitTolerance)

    def _phiFWDone(self, phis, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the FW phi limit.

        See `thetaFWDone`
        """
        return self._mapDone(self.phiCenter, phis, self.phiCWHome, n, k,
                             needAtEnd=needAtEnd, closeEnough=closeEnough,
                             limitTolerance=limitTolerance)

    def _phiRVDone(self, phis, n, k, needAtEnd=4, closeEnough=np.deg2rad(1), limitTolerance=np.deg2rad(2)):
        """ Return a mask of the cobras which we deem at the RV phi limit.

        See `thetaFWDone`
        """
        return self._mapDone(self.phiCenter, phis, self.phiCCWHome, n, k,
                             needAtEnd=needAtEnd, closeEnough=closeEnough,
                             limitTolerance=limitTolerance)

    def _setPhiCentersFromRun(self, geometryRun):
        self.phiCenter = np.load(geometryRun / 'data'/ 'phiCenter.npy')

    def _setPhiGeometryFromRun(self, geometryRun, onlyIfClear=True):
        if (onlyIfClear and (self.phiCenter is not None
                             and self.phiCWHome is not None
                             and self.phiCCWHome is not None)):
            return
        self._setPhiCentersFromRun(geometryRun)

        FW = np.load(geometryRun / 'data'/ 'phiFW.npy')
        RV = np.load(geometryRun / 'data'/ 'phiRV.npy')
        self.phiCCWHome = np.angle(FW[:,0,0] - self.phiCenter)
        self.phiCWHome = np.angle(RV[:,0,0] - self.phiCenter)
        dAng = self.phiCWHome - self.phiCCWHome
        dAng[dAng<0] += 2*np.pi
        stopped = np.where(dAng < np.deg2rad(182.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"phi ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(self.phiCWHome[stopped])} "
                              f"CCW={np.rad2deg(self.phiCCWHome[stopped])}")
            self.logger.error(f"     {np.round(np.rad2deg(dAng[stopped]), 2)}")

        self.logger.info(f'PHI runDir = {geometryRun}')
        self.phiRunDir = geometryRun

    def _setThetaCentersFromRun(self, geometryRun):
        self.thetaCenter = np.load(geometryRun / 'data' / 'thetaCenter.npy')

    def _setThetaGeometryFromRun(self, geometryRun, onlyIfClear=True):
        if (onlyIfClear and (self.thetaCenter is not None
                             and self.thetaCWHome is not None
                             and self.thetaCCWHome is not None)):
            return

        self._setThetaCentersFromRun(geometryRun)

        thetaFW = np.load(geometryRun / 'data' / 'thetaFW.npy')
        thetaRV = np.load(geometryRun / 'data' / 'thetaRV.npy')
        self.thetaCCWHome = np.angle(thetaFW[:,0,0] - self.thetaCenter[:])
        self.thetaCWHome = np.angle(thetaRV[:,0,0] - self.thetaCenter[:])

        dAng = (self.thetaCWHome - self.thetaCCWHome + np.pi) % (np.pi*2) + np.pi
        stopped = np.where(dAng < np.deg2rad(370.0))[0]
        if len(stopped) > 0:
            self.logger.error(f"theta ranges for cobras {stopped+1} are too small: "
                              f"CW={np.rad2deg(self.thetaCWHome[stopped])} "
                              f"CCW={np.rad2deg(self.thetaCCWHome[stopped])}")
            self.logger.error(f"     {np.round(np.rad2deg(dAng[stopped]), 2)}")

    def setGeometry(self, cmd):

        cmdKeys = cmd.cmd.keywords
        runDir = pathlib.Path(cmd.cmd.keywords['runDir'].values[0])

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        if phi is True:
            self._setPhiGeometryFromRun(runDir)
            self.logger.info(f'Using PHI geometry from {runDir}')
        else:
            self._setThetaGeometryFromRun(runDir)
            self.logger.info(f'Using THETA geometry from {runDir}')
        cmd.finish(f"text='Setting geometry is finished'")

    def makeMotorMap(self,cmd):
        """ Making motor map. """
        cmdKeys = cmd.cmd.keywords

        #self._connect()
        repeat = cmd.cmd.keywords['repeat'].values[0]
        stepsize = cmd.cmd.keywords['stepsize'].values[0]


        slowOnlyArg = 'slowOnly' in cmdKeys
        if slowOnlyArg is True:
            slowOnly = True
        else:
            slowOnly = False

        #limitOnTime=0.08
        
        delta=0.1

        # Switch from default no centroids to default do centroids
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        #print(self.goodIdx)
        if phi is True:
            eng.setPhiMode()
            steps = stepsize
            #repeat = 3
            day = time.strftime('%Y-%m-%d')
            
            self.logger.info(f'Running PHI SLOW motor map.')
            newXml= f'{day}-phi-slow.xml'
            #runDir, bad = self._makePhiMotorMap(cmd, newXml, repeat=repeat,steps=steps,delta=delta, fast=False)
            runDir, bad =eng.makePhiMotorMaps(newXml, steps=steps, totalSteps=6000, repeat=repeat, fast=False)

            self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
            self.pfi.loadModel([self.xml])
            
            if slowOnly is False:
                self.logger.info(f'Running PHI Fast motor map.')
                newXml= f'{day}-phi-final.xml'
                runDir, bad = eng.makePhiMotorMaps(newXml, steps=steps, totalSteps=6000, repeat=repeat, fast=True)

        else:
            eng.setThetaMode()
            steps = stepsize
            #repeat = 3
            day = time.strftime('%Y-%m-%d')
            
            
            self.logger.info(f'Running THETA SLOW motor map.')
            newXml= f'{day}-theta-slow.xml'
            #runDir, bad = self._makeThetaMotorMap(cmd, newXml, repeat=repeat,steps=steps,delta=delta, fast=False)
            runDir, bad = eng.makeThetaMotorMaps(newXml, totalSteps=10000, repeat=repeat,steps=steps,delta=delta, fast=False)

            self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
            self.pfi.loadModel([self.xml])

            if slowOnly is False:
                self.logger.info(f'Running THETA FAST motor map.')
                newXml= f'{day}-theta-final.xml'
                #runDir, bad = self._makeThetaMotorMap(cmd, newXml, repeat=repeat,steps=steps,delta=delta, fast=True)
                runDir, bad = eng.makeThetaMotorMaps(cmd, newXml, repeat=repeat,steps=steps,delta=delta, fast=True)
            
        cmd.finish(f'Motor map sequence finished')
    
    def moveToHome(self, cmd):
        cmdKeys = cmd.cmd.keywords

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        all = 'all' in cmdKeys
        
        # move to home position
        self.cc.moveToHome(self.cc.goodCobras, thetaEnable=True, phiEnable=True, thetaCCW=False)

        cmd.finish(f'Move all arms back to home')


    def movePhiToHome(self, cmd):
        cmdKeys = cmd.cmd.keywords

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys


        self._connect()
        totalSteps = 6000
        self.logger.info(f'phi home {-totalSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, 0, -totalSteps)

        cmd.finish(f'Move Phi back to home')

    def _makePhiMotorMap(self, cmd, newXml, repeat=3,
            steps=100,
            totalSteps=5000,
            fast=False,
            phiOnTime=None,
            updateGeometry=True,
            limitOnTime=0.08,
            limitSteps=5000,
            resetScaling=True,
            delta=0.1,
            fromHome=False
        ):
        
        totalSteps =  6000
        self._connect()
        self.logger.info(f'Connect to FPGA without problem.')

        defaultOnTimeFast = deepcopy([self.pfi.calibModel.motorOntimeFwd2,
                                    self.pfi.calibModel.motorOntimeRev2])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd2,
                                    self.pfi.calibModel.motorOntimeSlowRev2])
       
        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        fastOnTime = [np.full(self.nCobras, limitOnTime)] * 2
        if phiOnTime is not None:
            self.logger.info(f'Using phi on-time {phiOnTime}')

            if np.isscalar(phiOnTime):
                slowOnTime = [np.full(self.nCobras, phiOnTime)] * 2
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
        phiFW = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        phiRV = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        
        self.logger.info(f'number of good cobra = {len(self.goodIdx)}')
        if resetScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='phi')

        dataPath = self.runManager.dataDir
        self.logger.info(f'phi home {-totalSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, 0, -totalSteps)

        
        for n in range(repeat):
            fwlist = []
            rvlist = []
            
            # This is the beginning point
            phiFW[self.goodIdx, n, 0], filename = self.exposeAndExtractPositions(cmd)
            notdoneMask = np.zeros(len(phiFW), 'bool')
            notdoneMask[self.goodIdx] = True
            fwlist.append(filename)
            
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi forward to {(k+1)*steps}')
                self.pfi.moveAllSteps(self.allCobras[np.where(notdoneMask)], 0, steps, phiFast=False)
                phiFW[self.goodIdx, n, k+1], filename = self.exposeAndExtractPositions(cmd,
                                                        guess=phiFW[self.goodIdx, n, k])

                #print(phiFW)
                fwlist.append(filename)
                doneMask, lastAngles = self._phiFWDone(phiFW, n, k)
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
                #self.exposeAndExtractPositions(cmd)
            self.logger.info(f'Making phi forward stacked image.')
            self._makeStackImage(fwlist, f'{self.runManager.dataDir}/phiForwardStack{n}.fits')

            self.logger.info(f'{n+1}/{repeat} phi forward {totalSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, totalSteps)

            phiRV[self.goodIdx, n, 0], filename = self.exposeAndExtractPositions(cmd,
                                                        guess=phiFW[self.goodIdx, n, iteration])
            rvlist.append(filename)
            notdoneMask = np.zeros(len(phiRV), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} phi backward to {-totalSteps+(k+1)*steps}')
                self.pfi.moveAllSteps(self.allCobras[np.where(notdoneMask)], 0, -steps, phiFast=False)
                phiRV[self.goodIdx, n, k+1], filename = self.exposeAndExtractPositions(cmd,
                                                guess=phiRV[self.goodIdx, n, k])

                rvlist.append(filename)
                doneMask, lastAngles = self._phiRVDone(phiRV, n, k)
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

            self.logger.info(f'Making phi reverse stacked image.')
            self._makeStackImage(rvlist, f'{self.runManager.dataDir}/phiReverseStack{n}.fits')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} phi reverse {-totalSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, 0, -totalSteps)

        # restore ontimes after test
        self.pfi.calibModel.updateOntimes(phiFwd=defaultOnTimeFast[0], phiRev=defaultOnTimeFast[1], fast=True)
        self.pfi.calibModel.updateOntimes(phiFwd=defaultOnTimeSlow[0], phiRev=defaultOnTimeSlow[1], fast=False)

        dataPath = self.runManager.dataDir

        np.save(dataPath / 'phiFW', phiFW)
        np.save(dataPath / 'phiRV', phiRV)

        # calculate centers and phi angles
        phiCenter, phiRadius, phiAngFW, phiAngRV, badRange = self.cal.phiCenterAngles(phiFW, phiRV)
        for short in badRange:
            if short in self.badIdx:
                self.logger.warn(f"phi range for {short+1:-2d} is short, but that was expected")
            else:
                self.logger.warn(f'phi range for {short+1:-2d} is short: '
                                 f'out={np.rad2deg(phiAngRV[short,0,0]):-6.2f} '
                                 f'back={np.rad2deg(phiAngRV[short,0,-1]):-6.2f}')
        
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

        slow = not fast
        
        self.cal.updatePhiMotorMaps(phiMMFW, phiMMRV, bad, slow)

        if phiOnTime is not None:
            if np.isscalar(phiOnTime):
                onTime = np.full(self.nCobras, phiOnTime)
                self.logger.info(f'Updating on-time {onTime}.')
                self.cal.calibModel.updateOntimes(phiFwd=onTime, phiRev=onTime, fast=fast)
            else:
                self.logger.info(f'Updating on-time {phiOnTime}.')
                self.cal.calibModel.updateOntimes(phiFwd=phiOnTime[0], phiRev=phiOnTime[1], fast=fast)
        if updateGeometry is True:
            self.logger.info(f'Updating geometry in XML file = {newXml}.')
            self.cal.calibModel.updateGeometry(centers=phiCenter, phiArms=phiRadius)
        
        
        self.cal.calibModel.createCalibrationFile(self.runManager.outputDir / newXml, name='phiModel')

        self._setPhiGeometryFromRun(self.runManager.runDir, onlyIfClear=True)
        
        if len(self.badIdx)>0:
            bad[self.badIdx] = False
        return self.runManager.runDir, np.where(bad)[0]


    def _measureAngles(self, cmd, centers, homes):
        """ measure positions and angles for good cobras """

        curPos, _ = self.exposeAndExtractPositions(cmd,guess=centers)
        angles = (np.angle(curPos - centers) - homes) % (np.pi*2)
        
        return angles, curPos

    def _makeStackImage(self, filelist, output):
        
        i=0
        for f in filelist:
            hdu = fits.open(f)
            if i == 0:
                stack = deepcopy(hdu[1].data)
                i=i+1
            else:
                stack = hdu[1].data+stack

        hdu = fits.PrimaryHDU(stack/len(filelist))
        
        hdu.writeto(output,overwrite=True)

    def _acquireThetaMotorMap(self, cmd,
                             steps=100,
                             repeat=1,
                             totalSteps=10000,
                             fast=False,
                             thetaOnTime=None,
                             limitOnTime=0.08,
                             limitSteps=10000,
                             resetScaling=True,
                             fromHome=False):
        """ """
        self._connect()

        brokens = [self.pfi.calibModel.findCobraByModuleAndPositioner(1,47)+1,
                    #self.pfi.calibModel.findCobraByModuleAndPositioner(3,25)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(4,22)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(7,19)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(7,5)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(14,13)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,23)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,55)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(17,37)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(21,10)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(22,13)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(27,38)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(28,41)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(29,57)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(31,14)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(34,22)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(29,41)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(33,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(33,12)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(34,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(37,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(42,15)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(42,43)+1
                    ]

        self.setBrokenCobras(brokens=brokens)
        self.logger.info(f'broken fibers={brokens} ')
        totalSteps = 12000
        defaultOnTimeFast = deepcopy([self.pfi.calibModel.motorOntimeFwd1,
                                      self.pfi.calibModel.motorOntimeRev1])
        defaultOnTimeSlow = deepcopy([self.pfi.calibModel.motorOntimeSlowFwd1,
                                      self.pfi.calibModel.motorOntimeSlowRev1])

        # set fast on-time to a large value so it can move over whole range, set slow on-time to the test value.
        fastOnTime = [np.full(self.nCobras, limitOnTime)] * 2
        if thetaOnTime is not None:
            if np.isscalar(thetaOnTime):
                slowOnTime = [np.full(self.nCobras, thetaOnTime)] * 2
            else:
                slowOnTime = thetaOnTime
        elif fast:
            slowOnTime = defaultOnTimeFast
        else:
            slowOnTime = defaultOnTimeSlow

        # update ontimes for test
        self.pfi.calibModel.updateOntimes(thetaFwd=fastOnTime[0], thetaRev=fastOnTime[1], fast=True)
        self.pfi.calibModel.updateOntimes(thetaFwd=slowOnTime[0], thetaRev=slowOnTime[1], fast=False)

        # variable declaration for position measurement
        iteration = totalSteps // steps
        self.logger.info(f'nCobra = {self.nCobras}, repeat={repeat} iteration={iteration+1}')
        thetaFW = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)
        thetaRV = np.zeros((self.nCobras, repeat, iteration+1), dtype=complex)

        if resetScaling:
            self.pfi.resetMotorScaling(cobras=None, motor='theta')

        # record the theta movements
        self.logger.info(f'theta home {-limitSteps} steps')
        self.pfi.moveAllSteps(self.goodCobras, -limitSteps, 0)

        

        for n in range(repeat):
            fwlist = []
            rvlist = []
            #self.cam.resetStack(f'thetaForwardStack{n}.fits')

            # forward theta motor maps
            thetaFW[self.goodIdx, n, 0], filename = self.exposeAndExtractPositions(cmd)
            fwlist.append(filename)

            notdoneMask = np.zeros(len(thetaFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta forward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], (k+1)*steps, 0, thetaFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], steps, 0, thetaFast=False)
                thetaFW[self.goodIdx, n, k+1], filename = self.exposeAndExtractPositions(cmd)
                fwlist.append(filename)

                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -(k+1)*steps, 0)

                doneMask, lastAngles = self._thetaFWDone(thetaFW, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    thetaFW[self.goodIdx, n, k+2:] = thetaFW[self.goodIdx, n, k+1][:,None]
                    fwlist.append(filename)
                    break

            self.logger.info(f'Making stacked image.')
            self._makeStackImage(fwlist, f'{self.runManager.dataDir}/thetaForwardStack{n}.fits')

            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not reach theta CW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # make sure it goes to the limit
            self.logger.info(f'{n+1}/{repeat} theta forward {limitSteps} to limit')
            self.pfi.moveAllSteps(self.goodCobras, limitSteps, 0)

            # reverse theta motor maps
            #self.cam.resetStack(f'thetaReverseStack{n}.fits')
            thetaRV[self.goodIdx, n, 0], filename = self.exposeAndExtractPositions(cmd)
            rvlist.append(filename)

            notdoneMask = np.zeros(len(thetaFW), 'bool')
            notdoneMask[self.goodIdx] = True
            for k in range(iteration):
                self.logger.info(f'{n+1}/{repeat} theta backward to {(k+1)*steps}')
                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -(k+1)*steps, 0, thetaFast=False)
                else:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], -steps, 0, thetaFast=False)
                thetaRV[self.goodIdx, n, k+1], filename = self.exposeAndExtractPositions(cmd)
                rvlist.append(filename)

                if fromHome:
                    self.pfi.moveAllSteps(self.allCobras[notdoneMask], (k+1)*steps, 0)

                doneMask, lastAngles = self._thetaRVDone(thetaRV, n, k)
                if doneMask is not None:
                    newlyDone = doneMask & notdoneMask
                    if np.any(newlyDone):
                        notdoneMask &= ~doneMask
                        self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                if not np.any(notdoneMask):
                    thetaRV[self.goodIdx, n, k+1:] = thetaRV[self.goodIdx, n, k+1][:,None]
                    rvlist.append(filename)
                    break

            self._makeStackImage(rvlist, f'{self.runManager.dataDir}/thetaReverseStack{n}.fits')
            if doneMask is not None and np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} did not reach theta CCW limit:')
                for c_i in np.where(notdoneMask)[0]:
                    c = self.allCobras[c_i]
                    d = np.rad2deg(lastAngles[c_i])
                    with np.printoptions(precision=2, suppress=True):
                        self.logger.warn(f'  {str(c)}: {d}')

            # At the end, make sure the cobra back to the hard stop
            self.logger.info(f'{n+1}/{repeat} theta reverse {-limitSteps} steps to limit')
            self.pfi.moveAllSteps(self.goodCobras, -limitSteps, 0)
        #self.cam.resetStack()

        # restore ontimes after test
        self.pfi.calibModel.updateOntimes(thetaFwd=defaultOnTimeFast[0], thetaRev=defaultOnTimeFast[1], fast=True)
        self.pfi.calibModel.updateOntimes(thetaFwd=defaultOnTimeSlow[0], thetaRev=defaultOnTimeSlow[1], fast=False)

        # save calculation result
        dataPath = self.runManager.dataDir
        np.save(dataPath / 'thetaFW', thetaFW)
        np.save(dataPath / 'thetaRV', thetaRV)

        return self.runManager.runDir, thetaFW, thetaRV



    def _reduceThetaMotorMap(self, cmd, newXml, runDir, steps,
                            thetaOnTime=None,
                            delta=None, fast=False,
                            phiRunDir=None,
                            updateGeometry=False,
                            fromHome=False):
        dataPath = runDir / 'data'

        # load calculation result
        thetaFW = np.load(dataPath / 'thetaFW.npy')
        thetaRV = np.load(dataPath / 'thetaRV.npy')

        # calculate centers and theta angles
        thetaCenter, thetaRadius, thetaAngFW, thetaAngRV, badRange = self.cal.thetaCenterAngles(thetaFW,
                                                                                                thetaRV)
        for short in badRange:
            self.logger.warn(f'theta range for {short+1:-2d} is short: '
                             f'out={np.rad2deg(thetaAngRV[short,0,0]):-6.2f} '
                             f'back={np.rad2deg(thetaAngRV[short,0,-1]):-6.2f}')
        np.save(dataPath / 'thetaCenter', thetaCenter)
        np.save(dataPath / 'thetaRadius', thetaRadius)
        np.save(dataPath / 'thetaAngFW', thetaAngFW)
        np.save(dataPath / 'thetaAngRV', thetaAngRV)
        np.save(dataPath / 'badRange', badRange)

        self.thetaCenter = thetaCenter
        self.thetaCCWHome = thetaAngFW[:,0,0]
        self.thetaCCWHome = thetaAngRV[:,0,0]

        # calculate average speeds
        thetaSpeedFW, thetaSpeedRV = self.cal.speed(thetaAngFW, thetaAngRV, steps, delta)
        np.save(dataPath / 'thetaSpeedFW', thetaSpeedFW)
        np.save(dataPath / 'thetaSpeedRV', thetaSpeedRV)

        # calculate motor maps in Johannes weighting
        if fromHome:
            thetaMMFW, thetaMMRV, bad = self.cal.motorMaps2(thetaAngFW, thetaAngRV, steps, delta)
        else:
            thetaMMFW, thetaMMRV, bad = self.cal.motorMaps(thetaAngFW, thetaAngRV, steps, delta)
        for bad_i in np.where(bad)[0]:
            self.logger.warn(f'theta map for {bad_i+1} is bad')
        bad[badRange] = True
        np.save(dataPath / 'thetaMMFW', thetaMMFW)
        np.save(dataPath / 'thetaMMRV', thetaMMRV)
        np.save(dataPath / 'bad', np.where(bad)[0])

        # update XML file, using Johannes weighting
        slow = not fast
        self.cal.updateThetaMotorMaps(thetaMMFW, thetaMMRV, bad, slow)
        if thetaOnTime is not None:
            if np.isscalar(thetaOnTime):
                onTime = np.full(self.nCobras, thetaOnTime)
                self.logger.info(f'Updating theta-ontime: {onTime}')
                self.pfi.calibModel.updateOntimes(thetaFwd=onTime, thetaRev=onTime, fast=fast)
            else:
                self.logger.info(f'Updating theta-ontime: {thetaOnTime}')
                self.pfi.calibModel.updateOntimes(thetaFwd=thetaOnTime[0], thetaRev=thetaOnTime[1], fast=fast)
        if updateGeometry:
            self.logger.info(f'Updating geometry!')
            
            if phiRunDir is None:
                phiRunDir = self.phiRunDir
            
            phiCenter = np.load(phiRunDir / 'data' / 'phiCenter.npy')
            phiRadius = np.load(phiRunDir / 'data' / 'phiRadius.npy')
            phiFW = np.load(phiRunDir / 'data' / 'phiFW.npy')
            phiRV = np.load(phiRunDir / 'data' / 'phiRV.npy')

            thetaL, phiL, thetaCCW, thetaCW, phiCCW, phiCW = self.cal.geometry(thetaCenter, thetaRadius,
                                                                               thetaFW, thetaRV,
                                                                               phiCenter, phiRadius,
                                                                               phiFW, phiRV)
            self.pfi.calibModel.updateGeometry(thetaCenter, thetaL, phiL)
            self.pfi.calibModel.updateThetaHardStops(thetaCCW, thetaCW)
            self.pfi.calibModel.updatePhiHardStops(phiCCW, phiCW)

            self._setThetaGeometryFromRun(runDir)

        self.pfi.calibModel.createCalibrationFile(self.runManager.outputDir / newXml)

        if len(self.badIdx)>0:
            bad[self.badIdx] = False
        return self.runManager.runDir, np.where(bad)[0]

    def _makeThetaMotorMap(self, cmd, newXml,
                          repeat=3,
                          steps=100,
                          totalSteps=10000,
                          fast=False,
                          thetaOnTime=None,
                          updateGeometry=False,
                          phiRunDir=None,
                          limitOnTime=0.08,
                          limitSteps=10000,
                          resetScaling=True,
                          delta=np.deg2rad(5.0),
                          fromHome=False):

        if updateGeometry and phiRunDir is None:
            raise RuntimeError('To write geometry, need to be told the phiRunDir')

        runDir, thetaFW, thetaRV = self._acquireThetaMotorMap(cmd, steps=steps, repeat=repeat, totalSteps=totalSteps,
                                                             fast=fast, thetaOnTime=thetaOnTime,
                                                             limitOnTime=limitOnTime, limitSteps=limitSteps,
                                                             resetScaling=resetScaling, fromHome=fromHome)
        runDir, duds = self._reduceThetaMotorMap(cmd, newXml, runDir, steps,
                                                thetaOnTime=thetaOnTime,
                                                delta=delta, fast=fast,
                                                phiRunDir=phiRunDir,
                                                updateGeometry=False,
                                                fromHome=fromHome)
        return runDir, duds

    def targetConverge(self, cmd):
        """ Making target convergence test. """
        cmdKeys = cmd.cmd.keywords
        runs = cmd.cmd.keywords['totalTargets'].values[0]
        maxsteps = cmd.cmd.keywords['maxsteps'].values[0]
        ontime = 'ontime' in cmdKeys
        speed = 'speed' in cmdKeys
        
        eng.setNormalMode()
        self.cc.moveToHome(self.cc.goodCobras, thetaEnable=True, phiEnable=True, thetaCCW=False)
        if ontime is True:
            
            self.logger.info(f'Run convergence test of {runs} targets with constant on-time') 
            self.logger.info(f'Setting max step = {maxsteps}')
            eng.setConstantOntimeMode(maxSteps=1000)
            
            targets, moves = eng.convergenceTest2(self.cc.goodIdx, runs=runs, thetaMargin=np.deg2rad(15.0), 
                                phiMargin=np.deg2rad(15.0), thetaOffset=0, 
                                phiAngle=(np.pi*5/6, np.pi/3, np.pi/4), 
                                tries=16, tolerance=0.2, threshold=20.0, 
                                newDir=True, twoSteps=False)
        
        cmd.finish(f'target convergece is finished')


    def angleConverge(self, cmd):
        """ Making comvergence test for a specific arm. """
        cmdKeys = cmd.cmd.keywords
        runs = cmd.cmd.keywords['angleTargets'].values[0]

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        doGeometryArg = 'doGeometry' in cmdKeys

        if phi is True:
            
            if doGeometryArg is True:
                # Do geometry before every convergence test
                self.phiCenter = None
                self.phiCWHome = None 
                self.phiCCWHome = None 

            self._phiConvergenceTest(cmd, margin=15.0, runs=runs, tries=8,tolerance=0.2, scaleFactor=2.0)
            cmd.finish(f'angleConverge of phi arm is finished')
        else:
            if doGeometryArg is True:
                # Do geometry before every convergence test
                self.thetaCenter = None
                self.thetaCWHome = None 
                self.thetaCCWHome = None 
            self._thetaConvergenceTest(cmd, margin=15.0, runs=runs, tries=8,tolerance=0.2, scaleFactor=2.0)
            cmd.finish(f'angleConverge of theta arm is finished')

    def _phiConvergenceTest(self, cmd, margin=15.0, runs=50, tries=8, fast=False, 
                finalAngle=None, tolerance=0.2, scaleFactor=1.0):

        self._connect()
        dataPath = self.runManager.dataDir

        if (self.phiCenter is None or self.phiCWHome is None or self.phiCCWHome is None):
            self.logger.info('Get phi grometry first!!!')

            # variable declaration for center measurement
            steps = 200
            iteration = 4000 // steps
            phiFW = np.zeros((self.nCobras, iteration+1), dtype=complex)
            phiRV = np.zeros((self.nCobras, iteration+1), dtype=complex)

            #record the phi movements
            #self.cam.resetStack('phiForwardStack.fits')
            self.pfi.resetMotorScaling(self.goodCobras, 'phi')
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)
            phiFW[self.goodIdx, 0], _ = self.exposeAndExtractPositions(cmd)

            for k in range(iteration):
                self.logger.info(f'Phi forward to {k*steps}!!!')
                self.pfi.moveAllSteps(self.goodCobras, 0, steps, phiFast=False)
                phiFW[self.goodIdx, k+1], _ = self.exposeAndExtractPositions(cmd,guess=phiFW[self.goodIdx, k])

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 0, 5000, phiFast=True)

            # reverse phi motors
            #self.cam.resetStack('phiReverseStack.fits')
            phiRV[self.goodIdx, 0], _ = self.exposeAndExtractPositions(cmd,guess=phiFW[self.goodIdx, iteration])

            for k in range(iteration):
                self.logger.info(f'Phi backward to {-5000+k*steps}!!!')
                self.pfi.moveAllSteps(self.goodCobras, 0, -steps, phiFast=False)
                phiRV[self.goodIdx, k+1], _ = self.exposeAndExtractPositions(cmd,guess=phiRV[self.goodIdx, k])
            #self.cam.resetStack()

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)

            #dataPath = pathlib.Path(self.dataPath)
            # save calculation result
            np.save(dataPath / 'phiFW', phiFW)
            np.save(dataPath / 'phiRV', phiRV)

            # variable declaration
            phiCenter = np.zeros(self.nCobras, dtype=complex)
            phiRadius = np.zeros(self.nCobras, dtype=float)
            phiCCWHome = np.zeros(self.nCobras, dtype=float)
            phiCWHome = np.zeros(self.nCobras, dtype=float)

            # measure centers
            for c in self.goodIdx:
                data = np.concatenate((phiFW[c].flatten(), phiRV[c].flatten()))
                x, y, r = calculation.circle_fitting(data)
                phiCenter[c] = x + y*(1j)
                phiRadius[c] = r

            # measure phi hard stops
            for c in self.goodIdx:
                phiCCWHome[c] = np.angle(phiFW[c, 0] - phiCenter[c])
                phiCWHome[c] = np.angle(phiRV[c, 0] - phiCenter[c])

            # save calculation result
            np.save(dataPath / 'phiCenter', phiCenter)
            np.save(dataPath / 'phiRadius', phiRadius)
            np.save(dataPath / 'phiCCWHome', phiCCWHome)
            np.save(dataPath / 'phiCWHome', phiCWHome)

            self.logger.info('Save phi geometry setting')
            centers = phiCenter[self.goodIdx]
            homes = phiCCWHome[self.goodIdx]
            self.phiCenter = phiCenter
            self.phiCCWHome = phiCCWHome
            self.phiCWHome = phiCWHome

        else:
            self.logger.info('Use current phi geometry setting!!!')
            centers = self.phiCenter[self.goodIdx]
            homes = self.phiCCWHome[self.goodIdx]

        # convergence test
        phiData = np.zeros((self.nCobras, runs, tries, 4))
        zeros = np.zeros(len(self.goodIdx))
        notdoneMask = np.zeros(self.nCobras, 'bool')
        nowDone = np.zeros(self.nCobras, 'bool')
        tolerance = np.deg2rad(tolerance)

        for i in range(runs):
            self.logger.info('Running phi convergence now!!!')
            #self.cam.resetStack(f'phiConvergenceTest{i}.fits')
            if runs > 1:
                angle = np.deg2rad(margin + (180 - 2 * margin) * i / (runs - 1))
            else:
                angle = np.deg2rad(90)
            notdoneMask[self.goodIdx] = True
            self.logger.info(f'Run {i+1}: angle={np.rad2deg(angle):.2f} degree')
            self.pfi.resetMotorScaling(self.goodCobras, 'phi')
            self.pfi.moveThetaPhi(self.goodCobras, zeros, zeros + angle, phiFast=fast)
            cAngles, cPositions = self._measureAngles(cmd, centers, homes)
            phiData[self.goodIdx, i, 0, 0] = cAngles
            phiData[self.goodIdx, i, 0, 1] = np.real(cPositions)
            phiData[self.goodIdx, i, 0, 2] = np.imag(cPositions)
            phiData[self.goodIdx, i, 0, 3] = 1.0

            scale = np.full(len(self.goodIdx), 1.0)
            for j in range(tries - 1):
                nm = notdoneMask[self.goodIdx]
                self.pfi.moveThetaPhi(self.allCobras[notdoneMask], zeros[nm], (angle - cAngles)[nm],
                                      phiFroms=cAngles[nm], phiFast=fast)
                lastAngle = cAngles
                cAngles, cPositions = self._measureAngles(cmd, centers, homes)
                cAngles[cAngles>np.pi*(3/2)] -= np.pi*2
                nowDone[:] = False
                nowDone[self.goodIdx[abs(cAngles - angle) < tolerance]] = True
                newlyDone = nowDone & notdoneMask
                if np.any(newlyDone):
                    notdoneMask &= ~newlyDone
                    self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                for k in range(len(self.goodIdx)):
                    if abs(cAngles[k] - lastAngle[k]) > self.minScalingAngle:
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        engageScale = (rawScale - 1) / scaleFactor + 1
                        direction = 'ccw' if angle < lastAngle[k] else 'cw'
                        scale[k] = self.pfi.scaleMotorOntimeBySpeed(self.goodCobras[k], 'phi', direction, fast, engageScale)

                phiData[self.goodIdx, i, j+1, 0] = cAngles
                phiData[self.goodIdx, i, j+1, 1] = np.real(cPositions)
                phiData[self.goodIdx, i, j+1, 2] = np.imag(cPositions)
                phiData[self.goodIdx, i, j+1, 3] = scale
                self.logger.debug(f'Scaling factor: {np.round(scale, 2)}')
                if not np.any(notdoneMask):
                    phiData[self.goodIdx, i, j+2:, 0] = cAngles[..., np.newaxis]
                    phiData[self.goodIdx, i, j+2:, 1] = np.real(cPositions)[..., np.newaxis]
                    phiData[self.goodIdx, i, j+2:, 2] = np.imag(cPositions)[..., np.newaxis]
                    phiData[self.goodIdx, i, j+2:, 3] = scale[..., np.newaxis]
                    break

            if np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not finish: '
                                 f'{np.where(notdoneMask)[0]}, '
                                 f'{np.round(np.rad2deg(cAngles)[notdoneMask[self.goodIdx]], 2)}')

            # home phi
            self.pfi.moveAllSteps(self.goodCobras, 0, -5000, phiFast=True)
            #self.cam.resetStack()

        # save calculation result
        np.save(dataPath / 'phiData', phiData)
        self.pfi.resetMotorScaling(self.goodCobras, 'phi')

        if finalAngle is not None:
            angle = np.deg2rad(finalAngle)
            self.pfi.moveThetaPhi(self.goodCobras, zeros, zeros + angle, phiFast=fast)
            cAngles, cPositions = self._measureAngles(cmd, centers, homes)

            for j in range(tries - 1):
                self.pfi.moveThetaPhi(self.goodCobras, zeros, angle - cAngles, phiFroms=cAngles, phiFast=fast)
                lastAngle = cAngles
                cAngles, cPositions = self._measureAngles(cmd, centers, homes)
                cAngles[cAngles>np.pi*(3/2)] -= np.pi*2
                for k in range(len(self.goodIdx)):
                    if abs(angle - lastAngle[k]) > self.minScalingAngle:
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        if angle < lastAngle[k]:
                            scale[k] = 1 + (rawScale - 1) / (scaleFactor * ratioRv[k])
                            self.pfi.scaleMotorOntime(self.goodCobras[k], 'phi', 'ccw', scale[k])
                        else:
                            scale[k] = 1 + (rawScale - 1) / (scaleFactor * ratioFw[k])
                            self.pfi.scaleMotorOntime(self.goodCobras[k], 'phi', 'cw', scale[k])
            self.logger.info(f'Final angles: {np.round(np.rad2deg(cAngles), 2)}')
            self.pfi.resetMotorScaling(self.goodCobras, 'phi')
        #return self.runManager.runDir
    
    def _thetaConvergenceTest(self, cmd, margin=15.0, runs=50, tries=8, fast=False, tolerance=0.2, scaleFactor=1.0):
        self._connect()
        
        brokens = [self.pfi.calibModel.findCobraByModuleAndPositioner(1,47)+1,
                    #self.pfi.calibModel.findCobraByModuleAndPositioner(3,25)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(4,22)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(7,19)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(7,5)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(14,13)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,23)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,55)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(17,37)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(21,10)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(22,13)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(27,38)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(28,41)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(29,57)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(31,14)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(34,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(34,22)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(29,41)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(33,12)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(37,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(42,15)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(42,43)+1
                    ]
        self.setBrokenCobras(brokens=brokens)

        dataPath = self.runManager.dataDir

        if (self.thetaCenter is None or self.thetaCWHome is None or self.thetaCCWHome is None):
            self.logger.info('Get theta grometry first!!!')

            # variable declaration for center measurement
            steps = 300
            iteration = 6000 // steps
            thetaFW = np.zeros((self.nCobras, iteration+1), dtype=complex)
            thetaRV = np.zeros((self.nCobras, iteration+1), dtype=complex)

            #record the theta movements
            #self.cam.resetStack('thetaForwardStack.fits')
            self.pfi.resetMotorScaling(self.goodCobras, 'theta')
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)
            thetaFW[self.goodIdx, 0], _ = self.exposeAndExtractPositions(cmd)

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, steps, 0, thetaFast=False)
                thetaFW[self.goodIdx, 0], _ = self.exposeAndExtractPositions(cmd)

            # make sure it goes to the limit
            self.pfi.moveAllSteps(self.goodCobras, 10000, 0, thetaFast=True)

            # reverse theta motors
            #self.cam.resetStack('thetaReverseStack.fits')
            thetaRV[self.goodIdx, 0], _ = self.exposeAndExtractPositions(cmd)

            for k in range(iteration):
                self.pfi.moveAllSteps(self.goodCobras, -steps, 0, thetaFast=False)
                thetaRV[self.goodIdx, k+1], _ = self.exposeAndExtractPositions(cmd)
            #self.cam.resetStack()

            # At the end, make sure the cobra back to the hard stop
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)

            # save calculation result
            np.save(dataPath / 'thetaFW', thetaFW)
            np.save(dataPath / 'thetaRV', thetaRV)

            # variable declaration
            thetaCenter = np.zeros(self.nCobras, dtype=complex)
            thetaRadius = np.zeros(self.nCobras, dtype=float)
            thetaCCWHome = np.zeros(self.nCobras, dtype=float)
            thetaCWHome = np.zeros(self.nCobras, dtype=float)

            # measure centers
            for c in self.goodIdx:
                data = np.concatenate((thetaFW[c].flatten(), thetaRV[c].flatten()))
                x, y, r = calculation.circle_fitting(data)
                thetaCenter[c] = x + y*(1j)
                thetaRadius[c] = r

            # measure theta hard stops
            for c in self.goodIdx:
                thetaCCWHome[c] = np.angle(thetaFW[c, 0] - thetaCenter[c])
                thetaCWHome[c] = np.angle(thetaRV[c, 0] - thetaCenter[c])

            # save calculation result
            np.save(dataPath / 'thetaCenter', thetaCenter)
            np.save(dataPath / 'thetaRadius', thetaRadius)
            np.save(dataPath / 'thetaCCWHome', thetaCCWHome)
            np.save(dataPath / 'thetaCWHome', thetaCWHome)

            self.logger.info('Save theta geometry setting')
            centers = thetaCenter[self.goodIdx]
            homes = thetaCCWHome[self.goodIdx]
            self.thetaCenter = thetaCenter
            self.thetaCCWHome = thetaCCWHome
            self.thetaCWHome = thetaCWHome

        else:
            self.logger.info('Use current theta geometry setting!!!')
            centers = self.thetaCenter[self.goodIdx]
            homes = self.thetaCCWHome[self.goodIdx]

        # convergence test
        thetaData = np.zeros((self.nCobras, runs, tries, 4))
        zeros = np.zeros(len(self.goodIdx))
        tGaps = (((self.pfi.calibModel.tht1 - self.pfi.calibModel.tht0+np.pi) % (np.pi*2))-np.pi)[self.goodIdx]
        notdoneMask = np.zeros(self.nCobras, 'bool')
        nowDone = np.zeros(self.nCobras, 'bool')
        tolerance = np.deg2rad(tolerance)

        for i in range(runs):
            #self.cam.resetStack(f'thetaConvergenceTest{i}.fits')
            if runs > 1:
                angle = np.deg2rad(margin + (360 - 2*margin) * i / (runs - 1))
            else:
                angle = np.deg2rad(180)
            notdoneMask[self.goodIdx] = True
            self.logger.info(f'Run {i+1}: angle={np.rad2deg(angle):.2f} degree')
            self.pfi.resetMotorScaling(self.goodCobras, 'theta')
            self.pfi.moveThetaPhi(self.goodCobras, zeros + angle, zeros, thetaFast=fast)
            cAngles, cPositions = self._measureAngles(cmd, centers, homes)
            for k in range(len(self.goodIdx)):
                if angle > np.pi and cAngles[k] < tGaps[k] + 0.1:
                    cAngles[k] += np.pi*2
            thetaData[self.goodIdx, i, 0, 0] = cAngles
            thetaData[self.goodIdx, i, 0, 1] = np.real(cPositions)
            thetaData[self.goodIdx, i, 0, 2] = np.imag(cPositions)
            thetaData[self.goodIdx, i, 0, 3] = 1.0

            scale = np.full(len(self.goodIdx), 1.0)
            for j in range(tries - 1):
                dirs = angle > cAngles
                lastAngle = cAngles
                nm = notdoneMask[self.goodIdx]
                self.pfi.moveThetaPhi(self.allCobras[notdoneMask], (angle - cAngles)[nm],
                                      zeros[nm], thetaFroms=cAngles[nm], thetaFast=fast)
                cAngles, cPositions = self._measureAngles(cmd, centers, homes)
                nowDone[:] = False
                nowDone[self.goodIdx[abs((cAngles - angle + np.pi) % (np.pi*2) - np.pi) < tolerance]] = True
                newlyDone = nowDone & notdoneMask
                if np.any(newlyDone):
                    notdoneMask &= ~newlyDone
                    self.logger.info(f'done: {np.where(newlyDone)[0]}, {(notdoneMask == True).sum()} left')
                for k in range(len(self.goodIdx)):
                    if angle > np.pi and cAngles[k] < tGaps[k] + 0.1:
                        cAngles[k] += np.pi*2
                    elif angle < np.pi and cAngles[k] > np.pi*2 - 0.1:
                        cAngles[k] -= np.pi*2
                    if abs(cAngles[k] - lastAngle[k]) > self.minScalingAngle:
                        rawScale = abs((angle - lastAngle[k]) / (cAngles[k] - lastAngle[k]))
                        engageScale = (rawScale - 1) / scaleFactor + 1
                        direction = 'cw' if dirs[k] else 'ccw'
                        scale[k] = self.pfi.scaleMotorOntimeBySpeed(self.goodCobras[k], 'theta', direction, fast, engageScale)

                thetaData[self.goodIdx, i, j+1, 0] = cAngles
                thetaData[self.goodIdx, i, j+1, 1] = np.real(cPositions)
                thetaData[self.goodIdx, i, j+1, 2] = np.imag(cPositions)
                thetaData[self.goodIdx, i, j+1, 3] = scale
                self.logger.debug(f'Scaling factor: {np.round(scale, 2)}')
                if not np.any(notdoneMask):
                    thetaData[self.goodIdx, i, j+2:, 0] = cAngles[..., np.newaxis]
                    thetaData[self.goodIdx, i, j+2:, 1] = np.real(cPositions)[..., np.newaxis]
                    thetaData[self.goodIdx, i, j+2:, 2] = np.imag(cPositions)[..., np.newaxis]
                    thetaData[self.goodIdx, i, j+2:, 3] = scale[..., np.newaxis]
                    break

            if np.any(notdoneMask):
                self.logger.warn(f'{(notdoneMask == True).sum()} cobras did not finish: '
                                 f'{np.where(notdoneMask)[0]}, '
                                 f'{np.round(np.rad2deg(cAngles)[notdoneMask[self.goodIdx]], 2)}')

            # home theta
            self.pfi.moveAllSteps(self.goodCobras, -10000, 0, thetaFast=True)
            #self.cam.resetStack()

        # save calculation result
        np.save(dataPath / 'thetaData', thetaData)
        self.pfi.resetMotorScaling(self.goodCobras, 'theta')
        return self.runManager.runDir
    
    def movePhiToAngle(self, cmd):
        """ Making comvergence test for a specific arm. """
        cmdKeys = cmd.cmd.keywords
        angle = cmd.cmd.keywords['angle'].values[0]
        
        
        itr = cmd.cmd.keywords['iteration'].values[0]

        if itr == 0:
            itr = 8

        self.logger.info(f'Move phi to angle = {angle}')
            
        datapath = self._moveToPhiAngle(cmd, angle=angle, maxTries=itr, keepExistingPosition=False)

        self.logger.info(f'Data path : {datapath}')    
        cmd.finish(f'PHI is now at {angle} degress!!')
        
    def _getIndexInGoodCobras(self, idx=None):
        # map an index for all cobras to an index for only the visible cobras
        if idx is None:
            return np.arange(len(self.goodCobras))
        else:
            if len(set(idx) & set(self.badIdx)) > 0:
                raise RuntimeError('should not include invisible cobras')
            _idx = np.zeros(self.nCobras, 'bool')
            _idx[idx] = True
            return np.where(_idx[self.goodIdx])[0]

    def _moveToPhiAngle(self, cmd, idx=None, angle=60.0,
                       keepExistingPosition=False,
                       tolerance=np.rad2deg(0.005), maxTries=10,
                       scaleFactor=5.0,
                       doFast=False):
        """
        Robustly move to a given phi angle.

        This uses only the angle between the phi center and the
        measured spot to determine where the phi motor is, and only
        the phi motor is moved. The significant drawback is that it
        requires the location of the phi center, which is not always
        known. But for the initial, post-phiMap move, we do.

        EXPECTS TO BE AT PHI HOME if keepExistingPosition is False.

        Args
        ----
        idx : index or index array
          Which cobras to limit the move to.
        angle : `float`
          Degrees we want to move to from the CCW limit.
        keepExistingPosition : bool
          Do not reset the phi home position to where we are.
        tolerance : `float`
          How close we want to get, in degrees.
        maxTries: `int`
          How many moves to attempt.
        scaleFactor: `float`
          What fraction of the motion error to apply to the motor scale. 1/scalefactor
        doFast : bool
          For the first move, use the fast map?
        """

        dtype = np.dtype(dict(names=['iteration', 'cobra', 'target', 'position', 'left', 'steps', 'done'],
                              formats=['i2', 'i2', 'f4', 'f4', 'f4', 'i4', 'i1']))

        # We do want a new stack of these images.
        self._connect()
        #self.cam.resetStack(doStack=True)

        if idx is None:
            idx = self.goodIdx
        _idx = self._getIndexInGoodCobras(idx)
        cobras = np.array(self.allCobras[idx])
        moveList = []
        moves0 = np.zeros(len(cobras), dtype=dtype)

        if np.isscalar(angle):
            angle = np.full(len(cobras), angle)
        elif len(angle) == self.nCobras:
            angle = angle[idx]

        if self.phiCenter is not None:
            phiCenters = self.phiCenter
        else:
            raise RuntimeError("moduleTest needs to have been to told the phi Centers")
        phiCenters = phiCenters[idx]

        tolerance = np.deg2rad(tolerance)

        # extract sources and fiber identification
        allPos, _ = self.exposeAndExtractPositions(cmd,tolerance=0.2)
        curPos = allPos[_idx]
        if keepExistingPosition and hasattr(self, 'phiHomes'):
            homeAngles = self.phiHomes[idx]
            curAngles = self._fullAngle(curPos, phiCenters)
            lastAngles = self.dPhiAngle(curAngles, homeAngles, doAbs=True)
        else:
            homeAngles = self._fullAngle(curPos, phiCenters)
            curAngles = homeAngles
            lastAngles = np.zeros(len(homeAngles))
            if not hasattr(self, 'phiHomes'):
                self.phiHomes = np.zeros(self.nCobras)
            self.phiHomes[idx] = homeAngles

        targetAngles = np.full(len(homeAngles), np.deg2rad(angle))
        thetaAngles = targetAngles * 0
        ntries = 1
        notDone = targetAngles != 0
        left = self.dPhiAngle(targetAngles, lastAngles, doWrap=True)

        moves = moves0.copy()
        moveList.append(moves)
        for i in range(len(cobras)):
            cobraNum = cobras[i].cobraNum
            moves['iteration'][i] = 0
            moves['cobra'][i] = cobraNum
            moves['target'][i] = targetAngles[i]
            moves['position'][i] = lastAngles[i]
            moves['left'][i] = left[i]
            moves['done'][i] = not notDone[i]

        with np.printoptions(precision=2, suppress=True):
            self.logger.info("to: %s", np.rad2deg(targetAngles)[notDone])
            self.logger.info("at: %s", np.rad2deg(lastAngles)[notDone])
        while True:
            with np.printoptions(precision=2, suppress=True):
                self.logger.debug("to: %s", np.rad2deg(targetAngles)[notDone])
                self.logger.debug("at: %s", np.rad2deg(lastAngles)[notDone])
                self.logger.debug("try %d/%d, %d/%d cobras left: %s",
                                  ntries, maxTries,
                                  notDone.sum(), len(cobras),
                                  np.rad2deg(left)[notDone])
                self.logger.info("try %d/%d, %d/%d cobras left",
                                 ntries, maxTries,
                                 notDone.sum(), len(cobras))
            _, phiSteps = self.pfi.moveThetaPhi(cobras[notDone],
                                                thetaAngles[notDone],
                                                left[notDone],
                                                phiFroms=lastAngles[notDone],
                                                phiFast=(doFast and ntries==1))
            allPhiSteps = np.zeros(len(cobras), dtype='i4')
            allPhiSteps[notDone] = phiSteps

            # extract sources and fiber identification
            allPos, _ = self.exposeAndExtractPositions(cmd, tolerance=0.2)
            curPos = allPos[_idx]
            a1 = self._fullAngle(curPos, phiCenters)
            atAngles = self.dPhiAngle(a1, homeAngles, doAbs=True)
            left = self.dPhiAngle(targetAngles, atAngles, doWrap=True)

            # Any cobras which were 0 steps away on the last move are done.
            lastNotDone = notDone.copy()
            tooCloseToMove = (allPhiSteps == 0)
            notDone[tooCloseToMove] = False

            # check position errors
            closeEnough = np.abs(left) <= tolerance
            notDone[closeEnough] = False

            moves = moves0.copy()
            for i in range(len(cobras)):
                cobraNum = cobras[i].cobraNum
                moves['iteration'][i] = ntries
                moves['cobra'][i] = cobraNum
                moves['target'][i] = targetAngles[i]
                moves['position'][i] = atAngles[i]
                moves['left'][i] = left[i]
                moves['steps'][i] = allPhiSteps[i]
                moves['done'][i] = not notDone[i]
            moveList[-1]['steps'][lastNotDone] = phiSteps
            moveList.append(moves)

            if not np.any(notDone):
                self.logger.info(f'Convergence sequence done after {ntries} iterations')
                break

            for c_i in np.where(notDone)[0]:

                tryDist = self.dPhiAngle(targetAngles[c_i], lastAngles[c_i], doWrap=True)[0]
                gotDist = self.dPhiAngle(atAngles[c_i], lastAngles[c_i], doWrap=True)[0]
                rawScale = abs(tryDist/gotDist)
                if abs(tryDist) > np.deg2rad(2) and (rawScale < 0.9 or rawScale > 1.1):
                    direction = 'ccw' if tryDist < 0 else 'cw'

                    if rawScale > 1:
                        scale = 1 + (rawScale - 1)/scaleFactor
                    else:
                        scale = 1/(1 + (1/rawScale - 1)/scaleFactor)

                    if scale <= 0.75 or scale >= 1.25:
                        logCall = self.logger.info
                    else:
                        logCall = self.logger.debug

                    logCall(f'{c_i} at={np.rad2deg(atAngles[c_i]):0.2f} '
                            f'try={np.rad2deg(tryDist):0.2f} '
                            f'got={np.rad2deg(gotDist):0.2f} '
                            f'rawScale={rawScale:0.2f} scale={scale:0.2f}')
                    self.pfi.scaleMotorOntime(cobras[c_i], 'phi', direction, scale)

            lastAngles = atAngles
            if ntries >= maxTries:
                self.logger.warn(f'Reached max {maxTries} tries, {notDone.sum()} cobras left')
                #self.logger.warn(f'   cobras: {[c.cobraNum for c in cobras[np.where(notDone)]]}')
                self.logger.warn(f'   cobras: {[str(c) for c in cobras[np.where(notDone)]]}')
                self.logger.warn(f'   position: {np.round(np.rad2deg(atAngles)[notDone], 2)}')
                self.logger.warn(f'   left: {np.round(np.rad2deg(left)[notDone], 2)}')

                _, phiSteps = self.pfi.moveThetaPhi(cobras[notDone],
                                                    thetaAngles[notDone],
                                                    left[notDone],
                                                    phiFroms=lastAngles[notDone],
                                                    phiFast=(doFast and ntries==1),
                                                    doRun=False)
                self.logger.warn(f'   steps: {phiSteps}')

                break
            ntries += 1

        moves = np.concatenate(moveList)
        movesPath = self.runManager.outputDir / "phiConvergence.npy"
        np.save(movesPath, moves)

        return self.runManager.runDir
   
    def _moveToThetaAngle(self, cmd, idx=None, angle=60.0,
                         keepExistingPosition=False,
                         tolerance=1.0, maxTries=12, scaleFactor=5.0,
                         globalAngles=False,
                         doFast=False):
        """
        Robustly move to a given theta angle.

        This uses only the angle between the theta center and the
        measured spot to determine where the theta motor is, and only
        the theta motor is moved.

        Args
        ----
        idx : index or index array
          Which cobras to limit the move to.
        angle : `float`
          Degrees we want to move to from the CCW limit.
        globalAngle : `bool`
          Whether to use limit-based or module-based angles.
        tolerance : `float`
          How close we want to get, in degrees.
        maxTries: `int`
          How many moves to attempt.
        doFast : bool
          For the first move, use the fast map?
        """

        dtype = np.dtype(dict(names=['iteration', 'cobra', 'target', 'position', 'left', 'steps', 'done'],
                              formats=['i2', 'i2', 'f4', 'f4', 'f4', 'i4', 'i1']))

        # We do want a new stack of these images.
        self._connect()
        #self.cam.resetStack(doStack=True)

        if idx is None:
            idx = self.goodIdx
        _idx = self._getIndexInGoodCobras(idx)
        cobras = np.array(self.allCobras[idx])
        moveList = []
        moves0 = np.zeros(len(cobras), dtype=dtype)

        if np.isscalar(angle):
            angle = np.full(len(cobras), angle)
        elif len(angle) == self.nCobras:
            angle = angle[idx]

        if self.thetaCenter is not None:
            thetaCenters = self.thetaCenter
        else:
            thetaCenters = self.pfi.calibModel.centers
        thetaCenters =  thetaCenters[idx]

        tolerance = np.deg2rad(tolerance)

        if not keepExistingPosition or not hasattr(self, 'thetaHomes'):
            # extract sources and fiber identification
            self.logger.info(f'theta backward -10000 steps to limit')
            self.pfi.moveAllSteps(cobras, -10000, 0)
            allCurPos, _ = self.exposeAndExtractPositions(cmd, tolerance=0.2)
            homeAngles = self._fullAngle(allCurPos, thetaCenters)[_idx]
            if not hasattr(self, 'thetaHomes'):
                self.thetaHomes = np.zeros(self.nCobras)
                self.thetaAngles = np.zeros(self.nCobras)
            self.thetaHomes[idx] = homeAngles
            self.thetaAngles[idx] = 0
        homeAngles = self.thetaHomes[idx]
        lastAngles = self.thetaAngles[idx]

        targetAngles = np.deg2rad(angle)
        if globalAngles:
            targetAngles = (targetAngles - homeAngles) % (np.pi*2)

        phiAngles = targetAngles*0
        ntries = 1
        notDone = targetAngles != 0
        left = targetAngles - lastAngles

        moves = moves0.copy()
        moveList.append(moves)
        for i in range(len(cobras)):
            cobraNum = cobras[i].cobraNum
            moves['iteration'][i] = 0
            moves['cobra'][i] = cobraNum
            moves['target'][i] = targetAngles[i]
            moves['position'][i] = lastAngles[i]
            moves['left'][i] = left[i]
            moves['done'][i] = not notDone[i]

        with np.printoptions(precision=2, suppress=True):
            self.logger.info("to: %s", np.rad2deg(targetAngles)[notDone])
            self.logger.info("at: %s", np.rad2deg(lastAngles)[notDone])
        while True:
            with np.printoptions(precision=2, suppress=True):
                self.logger.debug("to: %s", np.rad2deg(targetAngles)[notDone])
                self.logger.debug("at: %s", np.rad2deg(lastAngles)[notDone])
                self.logger.debug("left try %d/%d, %d/%d: %s",
                                  ntries, maxTries,
                                  notDone.sum(), len(cobras),
                                  np.rad2deg(left)[notDone])
                self.logger.info("left try %d/%d, %d/%d",
                                 ntries, maxTries,
                                 notDone.sum(), len(cobras))
            thetaSteps, _ = self.pfi.moveThetaPhi(cobras[notDone],
                                                  left[notDone],
                                                  phiAngles[notDone],
                                                  thetaFroms=lastAngles[notDone],
                                                  thetaFast=(doFast and ntries==1))
            allThetaSteps = np.zeros(len(cobras), dtype='i4')
            allThetaSteps[notDone] = thetaSteps

            # extract sources and fiber identification
            allPos, _ = self.exposeAndExtractPositions(cmd, tolerance=0.2)

            curPos = allPos[_idx]
            # Get our angle w.r.t. home.
            atAngles = unwrappedPosition(curPos, thetaCenters, homeAngles,
                                         lastAngles, targetAngles)
            left = targetAngles - atAngles

            lastNotDone = notDone.copy()
            tooCloseToMove = (allThetaSteps == 0)
            notDone[tooCloseToMove] = False

            # check position errors
            closeEnough = np.abs(left) <= tolerance
            notDone[closeEnough] = False

            moves = moves0.copy()
            for i in range(len(cobras)):
                cobraNum = cobras[i].cobraNum
                moves['iteration'][i] = ntries
                moves['cobra'][i] = cobraNum
                moves['target'][i] = targetAngles[i]
                moves['position'][i] = atAngles[i]
                moves['left'][i] = left[i]
                moves['done'][i] = not notDone[i]
            moveList[-1]['steps'][lastNotDone] = thetaSteps
            moveList.append(moves)

            if not np.any(notDone):
                self.logger.info(f'Convergence sequence done after {ntries} iterations')
                break

            tryDist = targetAngles - lastAngles
            gotDist = atAngles - lastAngles
            for c_i in np.where(notDone)[0]:
                rawScale = np.abs(tryDist[c_i]/gotDist[c_i])
                if abs(tryDist[c_i]) > np.deg2rad(2) and (rawScale < 0.9 or rawScale > 1.1):
                    direction = 'ccw' if tryDist[c_i] < 0 else 'cw'

                    if rawScale > 1:
                        scale = 1 + (rawScale - 1)/scaleFactor
                    else:
                        scale = 1/(1 + (1/rawScale - 1)/scaleFactor)

                    if scale <= 0.75 or scale >= 1.25:
                        logCall = self.logger.info
                    else:
                        logCall = self.logger.debug

                    logCall(f'{c_i+1} at={np.rad2deg(atAngles[c_i]):0.2f} '
                            f'try={np.rad2deg(tryDist[c_i]):0.2f} '
                            f'got={np.rad2deg(gotDist[c_i]):0.2f} '
                            f'rawScale={rawScale:0.2f} scale={scale:0.2f}')
                    self.pfi.scaleMotorOntime(cobras[c_i], 'theta', direction, scale)

            lastAngles = atAngles
            self.thetaAngles[idx] = atAngles
            if ntries >= maxTries:
                self.logger.warn(f'Reached max {maxTries} tries, {notDone.sum()} cobras left')
                self.logger.warn(f'   cobras: {[str(c) for c in cobras[np.where(notDone)]]}')
                #self.logger.warn(f'   cobras: {[c.cobraNum for c in cobras[np.where(notDone)]]}')
                self.logger.warn(f'   position: {np.round(np.rad2deg(atAngles)[notDone], 2)}')
                self.logger.warn(f'   left: {np.round(np.rad2deg(left)[notDone], 2)}')

                thetaSteps, _ = self.pfi.moveThetaPhi(cobras[notDone],
                                                      left[notDone],
                                                      phiAngles[notDone],
                                                      thetaFroms=lastAngles[notDone],
                                                      thetaFast=(doFast and ntries==1),
                                                      doRun=False)
                self.logger.warn(f'   steps: {thetaSteps}')
                break
            ntries += 1

        moves = np.concatenate(moveList)
        movesPath = self.runManager.outputDir / 'thetaConvergence.npy'
        np.save(movesPath, moves)

        return self.runManager.runDir
    
    def gotoSafeFromPhi60(self, cmd):
        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.

        """
        phiAngle=60.0
        tolerance=np.rad2deg(0.05)
        angle = (180.0 - phiAngle) / 2.0
        thetaAngles = np.full(len(self.allCobras), -angle, dtype='f4')
        thetaAngles[np.arange(0,self.nCobras,2)] += 0
        thetaAngles[np.arange(1,self.nCobras,2)] += 180

        if not hasattr(self, 'thetaHomes'):
            keepExisting = False
        else:
            keepExisting = True

        run = self._moveToThetaAngle(cmd, None, angle=thetaAngles, tolerance=tolerance,
                                    keepExistingPosition=keepExisting, globalAngles=True)
        
        cmd.finish(f'gotoSafeFromPhi60 is finished')

    def gotoVerticalFromPhi60(self, cmd):
        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.

        """
        self._connect()
        
        broken = [self.pfi.calibModel.findCobraByModuleAndPositioner(1,47)+1,
                    #self.pfi.calibModel.findCobraByModuleAndPositioner(3,25)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(4,22)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(7,19)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(7,5)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(14,13)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,23)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(15,55)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(17,37)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(21,10)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(22,13)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(27,38)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(28,41)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(29,57)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(31,14)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(34,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(34,22)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(29,41)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(33,12)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(37,1)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(42,15)+1,
                    self.pfi.calibModel.findCobraByModuleAndPositioner(42,43)+1
                    ]
        #broken.append([189,290,447,559,589,607,809,1171,1252,1321,2033,2120,2136,2149,2163])
        self.setBrokenCobras(brokens=broken)

        phiAngle=60.0
        tolerance=np.rad2deg(0.05)
        angle = (180.0 - phiAngle) / 2.0
        thetaAngles = np.full(self.nCobras, -angle, dtype='f4')
        
        thetaAngles[np.arange(0,799)] += 270        
        thetaAngles[np.arange(799,1596)] += 150
        thetaAngles[np.arange(1596,2394)] += 30

        #thetaAngles[855] += 30


        if not hasattr(self, 'thetaHomes'):
            keepExisting = False
        else:
            keepExisting = True

        run = self._moveToThetaAngle(cmd, None, angle=thetaAngles, tolerance=tolerance,
                                    keepExistingPosition=keepExisting, globalAngles=True)
        
        cmd.finish(f'gotoVerticalFromPhi60 is finished')

    def motorOntimeSearch(self, cmd):
        """ FPS interface of searching the on time parameters for a specified motor speed """
        cmdKeys = cmd.cmd.keywords

        #self._connect()

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi is True:
            day = time.strftime('%Y-%m-%d')
            newXml = f'{day}-phi_opt.xml'        

            xml=self._phiOnTimeSearch(cmd, newXml, speeds=(0.06,0.12), steps=(500,250), iteration=3, repeat=1, b=0.07)
            
            cmd.finish(f'motorOntimeSearch of phi arm is finished')
        else:
            day = time.strftime('%Y-%m-%d')
            newXml = f'{day}-theta_opt.xml'  
            xml=self._thetaOnTimeSearch(cmd, newXml, speeds=[0.06,0.12], steps=[1000,500], iteration=3, repeat=1, b=0.088)
            self.logger.info(f'Theta on-time optimal XML = {xml}')
            cmd.finish(f'motorOntimeSearch of theta arm is finished')

    def _thetaOnTimeSearch(self, cmd, newXml, speeds=[0.06,0.12], steps=[1000,500], iteration=3, repeat=1, b=0.088):
        """ search the on time parameters for a specified motor speed """
        onTimeHigh = 0.08
        onTimeLow = 0.015
        onTimeHighSteps = 200

        if iteration < 3:
            self.logger.warn(f'Change iteration parameter from {iteration} to 3!')
            iteration = 3
        if np.isscalar(speeds) or len(speeds) != 2:
            raise ValueError(f'speeds parameter should be a two value tuples: {speeds}')
        if speeds[0] > speeds[1]:
            speeds = speeds[1], speeds[0]
        speeds = np.deg2rad(speeds)
        if np.isscalar(steps) or len(steps) != 2:
            raise ValueError(f'steps parameter should be a two value tuples: {steps}')
        if steps[0] < steps[1]:
            steps = steps[1], steps[0]

        # Getting current XML for total cobras
        des = pfiDesign.PFIDesign(self.xml)
        self.nCobras = len(des.centers)


        slopeF = np.zeros(self.nCobras)
        slopeR = np.zeros(self.nCobras)
        ontF = np.zeros(self.nCobras)
        ontR = np.zeros(self.nCobras)
        _ontF = []
        _ontR = []
        _spdF = []
        _spdR = []

        # get the average speeds for onTimeHigh, small step size since it's fast
        self.logger.info(f'Initial run, onTime = {onTimeHigh}')
        runDir, duds = self._makeThetaMotorMap(cmd, newXml, repeat=repeat, steps=onTimeHighSteps, thetaOnTime=onTimeHigh, fast=True)
        spdF = np.load(runDir / 'data' / 'thetaSpeedFW.npy')
        spdR = np.load(runDir / 'data' / 'thetaSpeedRV.npy')

        # assume a typical value for bad cobras, sticky??
        limitSpeed = np.deg2rad(0.02)
        spdF[spdF<limitSpeed] = limitSpeed
        spdR[spdR<limitSpeed] = limitSpeed

        _ontF.append(np.full(self.nCobras, onTimeHigh))
        _ontR.append(np.full(self.nCobras, onTimeHigh))
        _spdF.append(spdF.copy())
        _spdR.append(spdR.copy())

        # rough estimation for on time
        for (fast, speed, step) in zip([False, True], speeds, steps):
            # calculate on time
            for c_i in self.goodIdx:
                ontF[c_i] = self.thetaModel.getOntimeFromData(speed, _spdF[0][c_i], onTimeHigh)
                ontR[c_i] = self.thetaModel.getOntimeFromData(speed, _spdR[0][c_i], onTimeHigh)

            for n in range(iteration):
                ontF[ontF>self.pfi.maxThetaOntime] = self.pfi.maxThetaOntime
                ontR[ontR>self.pfi.maxThetaOntime] = self.pfi.maxThetaOntime
                ontF[ontF<onTimeLow] = onTimeLow
                ontR[ontR<onTimeLow] = onTimeLow
                self.logger.info(f'Run for {fast} on-time {n+1}/{iteration}, onTime = {np.round([ontF, ontR],4)}')
                runDir, duds = self._makeThetaMotorMap(cmd, newXml, repeat=repeat, steps=step, thetaOnTime=[ontF, ontR], fast=fast)
                spdF = np.load(runDir / 'data' / 'thetaSpeedFW.npy')
                spdR = np.load(runDir / 'data' / 'thetaSpeedRV.npy')
                _ontF.append(ontF.copy())
                _ontR.append(ontR.copy())
                _spdF.append(spdF.copy())
                _spdR.append(spdR.copy())

                # try the same on-time again for bad measuement
                spdF[spdF<=0.0] = speed
                spdR[spdR<=0.0] = speed

                # calculate on time
                for c_i in self.goodIdx:
                    ontF[c_i] = self.thetaModel.getOntimeFromData(speed, spdF[c_i], ontF[c_i])
                    ontR[c_i] = self.thetaModel.getOntimeFromData(speed, spdR[c_i], ontR[c_i])

        # try to find best on time, maybe.....
        ontF = self._searchOnTime(speeds[0], np.array(_spdF), np.array(_ontF))
        ontR = self._searchOnTime(speeds[0], np.array(_spdR), np.array(_ontR))
        ontF[ontF>onTimeHigh] = onTimeHigh
        ontR[ontR>onTimeHigh] = onTimeHigh

        # build SLOW motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        runDir, duds = self._makeThetaMotorMap(cmd, newXml, repeat=3, steps=250, thetaOnTime=[ontF, ontR], fast=False)
        self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
        self.pfi.loadModel([self.xml])

        # for fast on time
        ontF = self._searchOnTime(speeds[1], np.array(_spdF), np.array(_ontF))
        ontR = self._searchOnTime(speeds[1], np.array(_spdR), np.array(_ontR))
        ontF[ontF>onTimeHigh] = onTimeHigh
        ontR[ontR>onTimeHigh] = onTimeHigh

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        runDir, duds = self._makeThetaMotorMap(cmd, newXml, repeat=3, steps=125, thetaOnTime=[ontF, ontR], fast=True)
        self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
        self.pfi.loadModel([self.xml])

        return self.xml

    def _phiOnTimeSearch(self, cmd, newXml, speeds=(0.06,0.12), steps=(500,250), iteration=3, repeat=1, b=0.07):
        """ search the on time parameters for a specified motor speed """
        onTimeHigh = 0.08
        onTimeLow = 0.01
        onTimeHighSteps = 100

        if iteration < 3:
            self.logger.warn(f'Change iteration parameter from {iteration} to 3!')
            iteration = 3
        if np.isscalar(speeds) or len(speeds) != 2:
            raise ValueError(f'speeds parameter should be a two value tuples: {speeds}')
        if speeds[0] > speeds[1]:
            speeds = speeds[1], speeds[0]
        speeds = np.deg2rad(speeds)
        if np.isscalar(steps) or len(steps) != 2:
            raise ValueError(f'steps parameter should be a two value tuples: {steps}')
        if steps[0] < steps[1]:
            steps = steps[1], steps[0]

        # Getting current XML for total cobras
        des = pfiDesign.PFIDesign(self.xml)
        self.nCobras = len(des.centers)

        slopeF = np.zeros(self.nCobras)
        slopeR = np.zeros(self.nCobras)
        ontF = np.zeros(self.nCobras)
        ontR = np.zeros(self.nCobras)
        _ontF = []
        _ontR = []
        _spdF = []
        _spdR = []

        # get the average speeds for onTimeHigh, small step size since it's fast
        self.logger.info(f'Initial run, onTime = {onTimeHigh}')
        runDir, duds = self._makePhiMotorMap(cmd, newXml, repeat=repeat, steps=onTimeHighSteps, phiOnTime=onTimeHigh, fast=True)
        spdF = np.load(runDir / 'data' / 'phiSpeedFW.npy')
        spdR = np.load(runDir / 'data' / 'phiSpeedRV.npy')

        # assume a typical value for bad cobras, sticky??
        limitSpeed = np.deg2rad(0.02)
        spdF[spdF<limitSpeed] = limitSpeed
        spdR[spdR<limitSpeed] = limitSpeed

        _ontF.append(np.full(self.nCobras, onTimeHigh))
        _ontR.append(np.full(self.nCobras, onTimeHigh))
        _spdF.append(spdF.copy())
        _spdR.append(spdR.copy())

        for (fast, speed, step) in zip([False, True], speeds, steps):
            # calculate on time
            self.logger.info(f'Run for best {"Fast" if fast else "Slow"} motor maps')
            for c_i in self.goodIdx:
                ontF[c_i] = self.phiModel.getOntimeFromData(speed, _spdF[0][c_i], onTimeHigh)
                ontR[c_i] = self.phiModel.getOntimeFromData(speed, _spdR[0][c_i], onTimeHigh)

            for n in range(iteration):
                ontF[ontF>self.pfi.maxPhiOntime] = self.pfi.maxPhiOntime
                ontR[ontR>self.pfi.maxPhiOntime] = self.pfi.maxPhiOntime
                ontF[ontF<onTimeLow] = onTimeLow
                ontR[ontR<onTimeLow] = onTimeLow
                self.logger.info(f'Run for {fast} ontime {n+1}/{iteration}, onTime = {np.round([ontF, ontR],4)}')
                runDir, duds = self._makePhiMotorMap(cmd, newXml, repeat=repeat, steps=step, phiOnTime=[ontF, ontR], fast=fast)
                spdF = np.load(runDir / 'data' / 'phiSpeedFW.npy')
                spdR = np.load(runDir / 'data' / 'phiSpeedRV.npy')
                _ontF.append(ontF.copy())
                _ontR.append(ontR.copy())
                _spdF.append(spdF.copy())
                _spdR.append(spdR.copy())

                # try the same on-time again for bad measuement
                spdF[spdF<=0.0] = speed
                spdR[spdR<=0.0] = speed

                # calculate on time
                for c_i in self.goodIdx:
                    ontF[c_i] = self.thetaModel.getOntimeFromData(speed, spdF[c_i], ontF[c_i])
                    ontR[c_i] = self.thetaModel.getOntimeFromData(speed, spdR[c_i], ontR[c_i])

        # try to find best on time, maybe.....
        ontF = self._searchOnTime(speeds[0], np.array(_spdF), np.array(_ontF))
        ontR = self._searchOnTime(speeds[0], np.array(_spdR), np.array(_ontR))
        ontF[ontF>self.pfi.maxPhiOntime] = self.pfi.maxPhiOntime
        ontR[ontR>self.pfi.maxPhiOntime] = self.pfi.maxPhiOntime

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        runDir, duds = self._makePhiMotorMap(cmd, newXml, repeat=3, steps=250, phiOnTime=[ontF, ontR], fast=False)
        self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
        self.pfi.loadModel([self.xml])

        # for fast motor maps
        ontF = self._searchOnTime(speeds[1], np.array(_spdF), np.array(_ontF))
        ontR = self._searchOnTime(speeds[1], np.array(_spdR), np.array(_ontR))

        # build motor maps
        self.logger.info(f'Build motor maps, best onTime = {np.round([ontF, ontR],4)}')
        runDir, duds = self._makePhiMotorMap(cmd, newXml, repeat=3, steps=125, phiOnTime=[ontF, ontR], fast=True)
        
        self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
        self.pfi.loadModel([self.xml])
        

        return self.xml

    def _searchOnTime(self, speed, sData, tData):
        """ There should be some better ways to do!!! """
        onTime = np.zeros(self.nCobras)

        for c in self.goodIdx:
            s = sData[:,c]
            t = tData[:,c]
            model = SpeedModel()
            err = model.buildModel(s, t)

            if err:
                self.logger.warn(f'Building model failed #{c+1}, set to max value')
                onTime[c] = np.max(t)
            else:
                onTime[c] = model.toOntime(speed)
                if not np.isfinite(onTime[c]):
                    self.logger.warn(f'Curve fitting failed #{c+1}, set to median value')
                    onTime[c] = np.median(t)
#            if np.median(s[t==np.max(t)]) <= speed:
#                self.logger.warn(f'Cobra #{c+1} is sticky, set to max value')
#                onTime[c] = np.max(t)
#                continue

#            try:
#                params, params_cov = optimize.curve_fit(speedFunc, t, s, p0=[10, 0.06])
#                if params[0] < 0 or params[1] < 0:
#                    # remove some slow data and try again
#                    self.logger.warn(f'Curve fitting error #{c+1}, try again')
#                    s[t==np.max(t)] = np.max(s[t==np.max(t)])
#                    params, params_cov = optimize.curve_fit(speedFunc, t, s, p0=[10, 0.06])
#                if params[0] < 0 or params[1] < 0:
#                    raise
#                onTime[c] = invSpeedFunc(speed, params[0], params[1])
#            except:
#                onTime[c] = np.median(t)
#                self.logger.warn(f'Curve fitting failed #{c+1}, set to median value')

        return onTime

    def moveToDesign(self,cmd):
        """ Move cobras to the pfsDesign. """

        raise NotImplementedError('moveToDesign')
        cmd.finish()

    def _mcsExpose(self, cmd, expTime=None, doCentroid=True):
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

        cmdString = "expose object expTime=%0.1f %s" % (expTime,
                                                                   'doCentroid' if doCentroid else '')
        cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                      forUserCmd=cmd, timeLim=expTime+30)
        if cmdVar.didFail:
            cmd.warn('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))
            return None

        filekey= self.actor.models['mcs'].keyVarDict['filename'][0]
        filename = pathlib.Path(filekey)
        datapath = filename.parents[0]
        frameId = int(filename.stem[4:], base=10)
        cmd.inform(f'frameId={frameId}')
        cmd.inform(f'filename={filename}')

        return datapath, filename, frameId

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

