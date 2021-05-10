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
from ics.fpsActor.utils import display as vis
reload(vis)

from procedures.moduleTest import calculation
reload(calculation)
from procedures.moduleTest.speedModel import SpeedModel

from ics.cobraCharmer import pfi as pfiControl
from ics.cobraCharmer import pfiDesign
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
            ('ledlight', '@(on|off)', self.ledlight),
            ('loadDesign', '<id>', self.loadDesign),
            ('loadModel', '<xml>', self.loadModel),
            ('movePhiToAngle', '<angle> <iteration>', self.movePhiToAngle),
            ('moveToHome','@(phi|theta|all)', self.moveToHome),
            ('setGeometry', '@(phi|theta) <runDir>', self.setGeometry),            
            ('moveToObsTarget', '', self.moveToObsTarget),
            ('gotoSafeFromPhi60','',self.gotoSafeFromPhi60),
            ('gotoVerticalFromPhi60','',self.gotoVerticalFromPhi60),
            ('makeMotorMap','@(phi|theta) <stepsize> <repeat> [@slowOnly]',self.makeMotorMap),
            ('angleConverge','@(phi|theta) <angleTargets> [@doGeometry]',self.angleConverge),
            ('targetConverge','@(ontime|speed) <totalTargets> <maxsteps>',self.targetConverge),
            ('motorOntimeSearch','@(phi|theta)',self.motorOntimeSearch),
            ('visCobraSpots','@(phi|theta) <runDir>',self.visCobraSpots),
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

    def getPositionsForFrame(self, frameId):
        mcsData = self.nv.readCentroid(frameId)
        self.logger.info(f'mcs data {mcsData.shape[0]}')
        centroids = {'x':mcsData['centroidx'].values.astype('float'),
                     'y':mcsData['centroidy'].values.astype('float')}
        return centroids

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

    def ledlight(self, cmd):
        """Turn on/off the fiducial fiber light"""
        cmdKeys = cmd.cmd.keywords

        light_on = 'on' in cmdKeys
        light_off = 'off' in cmdKeys

        if light_on:
            cmdString = f'led on'
            infoString = 'Turn on fiducial fibers'
            
        else:
            cmdString = f'led off'
            infoString = 'Turn off fiducial fibers'
        
        cmdVar = self.actor.cmdr.call(actor='peb', cmdStr=cmdString,
                                       forUserCmd=cmd, timout=10)
           
        self.logger.info(f'{infoString}')
   

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
    

    def setGeometry(self, cmd):

        cmdKeys = cmd.cmd.keywords
        runDir = pathlib.Path(cmd.cmd.keywords['runDir'].values[0])

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        if phi is True:
            self.cc.setPhiGeometryFromRun(runDir)
            self.logger.info(f'Using PHI geometry from {runDir}')
        else:
            self.cc.setThetaGeometryFromRun(runDir)
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
            self.cc.pfi.loadModel([self.xml])
            
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
            self.cc.pfi.loadModel([self.xml])

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
        allfiber = 'all' in cmdKeys
        
        if phi is True:
            self.cc.moveToHome(self.cc.goodCobras, phiEnable=True)

        if theta is True:
            self.cc.moveToHome(self.cc.goodCobras, thetaEnable=True)

        if allfiber is True:
            # move to home position
            self.cc.moveToHome(self.cc.goodCobras, thetaEnable=True, phiEnable=True, thetaCCW=False)

        cmd.finish(f'Move all arms back to home')



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

    def moveToThetaAngleFromPhi60(self, cmd):
        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.
        """
        phiAngle=60.0
        tolerance=np.rad2deg(0.05)
        angle = (180.0 - phiAngle) / 2.0
        thetaAngles = np.full(len(self.allCobras), -angle, dtype='f4')
        thetaAngles[np.arange(0,self.nCobras,2)] += 0
        thetaAngles[np.arange(1,self.nCobras,2)] += 180
        
        dataPath, diffAngles, moves = eng.moveThetaAngles(cc.goodIdx, thetaAngles, 
            relative=False, local=True, tolerance=0.002, tries=12, fast=False, newDir=True)       
        
        cmd.finish(f'gotoSafeFromPhi60 is finished')

    def movePhiToAngle(self, cmd):
        """ Making comvergence test for a specific arm. """
        cmdKeys = cmd.cmd.keywords
        angle = cmd.cmd.keywords['angle'].values[0]
        
        
        itr = cmd.cmd.keywords['iteration'].values[0]

        if itr == 0:
            itr = 8

        self.logger.info(f'Move phi to angle = {angle}')
        
        eng.setPhiMode()    
        # move phi to 60 degree for theta test
        dataPath, diffAngles, moves = eng.movePhiAngles(self.cc.goodIdx, np.deg2rad(angle), 
            relative=False, local=True, tolerance=0.002, tries=12, fast=False, newDir=True)
        
        self.logger.info(f'Data path : {datapath}')    
        cmd.finish(f'PHI is now at {angle} degress!!')

    
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

 
    def moveToObsTarget(self,cmd):
        """ Move cobras to the pfsDesign. """

        raise NotImplementedError('moveToObsTarget')
        cmd.finish()


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

    def visCobraSpots(self, cmd):
        cmdKeys = cmd.cmd.keywords
        runDir = pathlib.Path(cmd.cmd.keywords['runDir'].values[0])
        self.logger.info(f'Loading model = {self.xml}')
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi:        
            vis.visCobraSpots(runDir,self.xml, arm='phi')
        