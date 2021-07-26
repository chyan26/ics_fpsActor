from opdb import opdb
import signal
import os
import subprocess as sub
from procedures.moduleTest import engineer as eng
from procedures.moduleTest import cobraCoach
from ics.cobraCharmer import cobraState
from ics.cobraCharmer.fpgaState import fpgaState
from ics.cobraCharmer import pfiDesign
from ics.cobraCharmer import pfi as pfiControl
from procedures.moduleTest.speedModel import SpeedModel
from procedures.moduleTest import calculation
import pathlib
import sys
from importlib import reload
import datetime

from astropy.io import fits

import numpy as np
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
from ics.fpsActor.utils import display as vis
reload(vis)

reload(calculation)


reload(pfiControl)
reload(cobraCoach)


class FpsCmd(object):
    def __init__(self, actor):
        # This lets us access the rest of the actor.
        self.actor = actor

        self.nv = najaVenator.NajaVenator()

        self.tranMatrix = None
        self._db = None
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
            ('buildTransMatrix', '[<frameId>]', self.buildTransMatrix),
            ('fpgaSim', '@(on|off) [<datapath>]', self.fpgaSim),
            ('ledlight', '@(on|off)', self.ledlight),
            ('loadDesign', '<id>', self.loadDesign),
            ('loadModel', '<xml>', self.loadModel),
            ('movePhiToAngle', '<angle> <iteration>', self.movePhiToAngle),
            ('moveToHome', '@(phi|theta|all)', self.moveToHome),
            ('setCobraMode', '@(phi|theta|normal)', self.setCobraMode),
            ('setGeometry', '@(phi|theta) <runDir>', self.setGeometry),
            ('moveToObsTarget', '', self.moveToObsTarget),
            ('moveToSafePosition', '', self.moveToSafePosition),
            ('gotoVerticalFromPhi60', '', self.gotoVerticalFromPhi60),
            ('makeMotorMap', '@(phi|theta) <stepsize> <repeat> [@slowOnly]', self.makeMotorMap),
            ('makeOntimeMap', '@(phi|theta)', self.makeOntimeMap),
            ('angleConverge', '@(phi|theta) <angleTargets>', self.angleConverge),
            ('targetConverge', '@(ontime|speed) <totalTargets> <maxsteps>', self.targetConverge),
            ('motorOntimeSearch', '@(phi|theta)', self.motorOntimeSearch),
            ('visCobraSpots', '@(phi|theta) <runDir>', self.visCobraSpots),
            ('calculateBoresight', '[<startFrame>] [<endFrame>]', self.calculateBoresight)
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("angle", types.Int(), help="arm angle"),
                                        keys.Key("stepsize", types.Int(), help="step size of motor"),
                                        keys.Key("repeat", types.Int(),
                                                 help="number of iteration for motor map generation"),
                                        keys.Key("angleTargets", types.Int(),
                                                 help="Target number for angle convergence"),
                                        keys.Key("totalTargets", types.Int(),
                                                 help="Target number for 2D convergence"),
                                        keys.Key("maxsteps", types.Int(),
                                                 help="Maximum step number for 2D convergence test"),
                                        keys.Key("xml", types.String(), help="XML filename"),
                                        keys.Key("datapath", types.String(),
                                                 help="Mock data for simulation mode"),
                                        keys.Key("runDir", types.String(), help="Directory of run data"),
                                        keys.Key("startFrame", types.Int(),
                                                 help="starting frame for boresight calculating"),
                                        keys.Key("endFrame", types.Int(),
                                                 help="ending frame for boresight calculating"),
                                        keys.Key("visit", types.Int(), help="PFS visit to use"),
                                        keys.Key("frameId", types.Int(), help="PFS Frame ID"),
                                        keys.Key("iteration", types.Int(), help="Interation number"),
                                        keys.Key(
                                            "id", types.Long(), help="fpsDesignId for the field,which defines the fiber positions"),
                                        keys.Key("mask", types.Int(), help="mask for power and/or reset"),
                                        keys.Key("expTime", types.Float(), help="Seconds for exposure"),
                                        )

        self.logger = logging.getLogger('fps')
        self.logger.setLevel(logging.INFO)

        self.cc = None
        self.fpgaHost = None
        self.p = None
        self.simDataPath = None

    def connectToDB(self, cmd):
        """connect to the database if not already connected"""

        if self._db is not None:
            return self._db

        try:
            config = self.actor.config
            hostname = config.get('db', 'hostname')
            dbname = config.get('db', 'dbname', fallback='opdb')
            port = config.get('db', 'port', fallback=5432)
            username = config.get('db', 'username', fallback='pfs')
        except Exception as e:
            raise RuntimeError(f'failed to load opdb configuration: {e}')

        try:
            db = opdb.OpDB(hostname, port, dbname, username, 'pfspass')
            db.connect()

        except:
            raise RuntimeError("unable to connect to the database")

        if cmd is not None:
            cmd.inform('text="Connected to Database"')

        self._db = db
        return self._db

    def fpgaSim(self, cmd):
        """Turn on/off simulalation mode of FPGA"""
        cmdKeys = cmd.cmd.keywords
        datapath = cmd.cmd.keywords['datapath'].values[0] if 'datapath' in cmdKeys else None

        simOn = 'on' in cmdKeys
        simOff = 'off' in cmdKeys

        my_env = os.environ.copy()

        if simOn is True:
            self.fpgaHost = 'localhost'
            self.logger.info(f'Starting a FPGA simulator.')
            self.p = sub.Popen(['fpgaSim'], env=my_env)

            self.logger.info(f'FPGA simulator started with PID = {self.p.pid}.')
            if datapath is None:
                self.logger.warn(f'FPGA simulator is ON but datapath is not given.')
            else:
                self.simDataPath = datapath

        if simOff is True:
            self.fpgaHost = 'fpga'

            self.logger.info(f'Stopping FPGA simulator.')
            self.simDataPath = None

            os.kill(self.p.pid, signal.SIGKILL)
            os.kill(self.p.pid+1, signal.SIGKILL)

        cmd.finish(f"text='fpgaSim command finished.'")

    def loadModel(self, cmd):
        """ Loading cobr"""
        xml = cmd.cmd.keywords['xml'].values[0]
        self.logger.info(f'Input XML file = {xml}')
        self.xml = pathlib.Path(xml)

        mod = 'ALL'

        if self.simDataPath is None:
            self.cc = cobraCoach.CobraCoach('fpga', loadModel=False, actor=self.actor, cmd=cmd)
        else:
            cmd.inform(f"text='Connecting to FPGA simulator'")
            self.cc = cobraCoach.CobraCoach(self.fpgaHost, loadModel=False, simDataPath=self.simDataPath,
                                            actor=self.actor, cmd=cmd)

        self.cc.loadModel(file=pathlib.Path(self.xml))
        eng.setCobraCoach(self.cc)

        cmd.finish(f"text='Loading model = {self.xml}'")

    def getPositionsForFrame(self, frameId):
        mcsData = self.nv.readCentroid(frameId)
        self.logger.info(f'mcs data {mcsData.shape[0]}')
        centroids = {'x': mcsData['centroidx'].values.astype('float'),
                     'y': mcsData['centroidy'].values.astype('float')}
        return centroids

    @staticmethod
    def dPhiAngle(target, source, doWrap=False, doAbs=False):
        d = np.atleast_1d(target - source)

        if doAbs:
            d[d < 0] += 2*np.pi
            d[d >= 2*np.pi] -= 2*np.pi

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
            a[a < 0] += 2*np.pi
            a[a >= 2*np.pi] -= 2*np.pi

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

    def setCobraMode(self, cmd):
        cmdKeys = cmd.cmd.keywords

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        normal = 'normal' in cmdKeys

        if phi is True:
            eng.setPhiMode()
            self.logger.info(f'Cobra is now in PHI mode')

        if theta is True:
            eng.setThetaMode()
            self.logger.info(f'Cobra is now in THETA mode')

        if normal is True:
            eng.setNormalMode()
            self.logger.info(f'Cobra is now in NORMAL mode')

        cmd.finish(f"text='Setting corba mode is finished'")

    def setGeometry(self, cmd):

        cmdKeys = cmd.cmd.keywords
        runDir = pathlib.Path(cmd.cmd.keywords['runDir'].values[0])

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        if phi is True:
            # eng.setPhiMode()
            self.cc.setPhiGeometryFromRun(runDir)
            self.logger.info(f'Using PHI geometry from {runDir}')
        else:
            # eng.setThetaMode()
            self.cc.setThetaGeometryFromRun(runDir)
            self.logger.info(f'Using THETA geometry from {runDir}')
        cmd.finish(f"text='Setting geometry is finished'")

    def makeMotorMap(self, cmd):
        """ Making motor map. """
        cmdKeys = cmd.cmd.keywords

        # self._connect()
        repeat = cmd.cmd.keywords['repeat'].values[0]
        stepsize = cmd.cmd.keywords['stepsize'].values[0]
        visit = self.actor.visitor.setOrGetVisit(cmd)

        slowOnlyArg = 'slowOnly' in cmdKeys
        if slowOnlyArg is True:
            slowOnly = True
        else:
            slowOnly = False

        # limitOnTime=0.08

        delta = 0.1

        # Switch from default no centroids to default do centroids
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        # print(self.goodIdx)
        if phi is True:
            eng.setPhiMode()
            steps = stepsize
            #repeat = 3
            day = time.strftime('%Y-%m-%d')

            self.logger.info(f'Running PHI SLOW motor map.')
            newXml = f'{day}-phi-slow.xml'
            #runDir, bad = self._makePhiMotorMap(cmd, newXml, repeat=repeat,steps=steps,delta=delta, fast=False)
            runDir, bad = eng.makePhiMotorMaps(
                newXml, steps=steps, totalSteps=6000, repeat=repeat, fast=False)

            self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
            self.cc.pfi.loadModel([self.xml])

            if slowOnly is False:
                self.logger.info(f'Running PHI Fast motor map.')
                newXml = f'{day}-phi-final.xml'
                runDir, bad = eng.makePhiMotorMaps(
                    newXml, steps=steps, totalSteps=6000, repeat=repeat, fast=True)

        else:
            eng.setThetaMode()
            steps = stepsize
            #repeat = 3
            day = time.strftime('%Y-%m-%d')

            self.logger.info(f'Running THETA SLOW motor map.')
            newXml = f'{day}-theta-slow.xml'
            #runDir, bad = self._makeThetaMotorMap(cmd, newXml, repeat=repeat,steps=steps,delta=delta, fast=False)
            runDir, bad = eng.makeThetaMotorMaps(
                newXml, totalSteps=10000, repeat=repeat, steps=steps, delta=delta, fast=False)

            self.xml = pathlib.Path(f'{runDir}/output/{newXml}')
            self.cc.pfi.loadModel([self.xml])

            if slowOnly is False:
                self.logger.info(f'Running THETA FAST motor map.')
                newXml = f'{day}-theta-final.xml'
                #runDir, bad = self._makeThetaMotorMap(cmd, newXml, repeat=repeat,steps=steps,delta=delta, fast=True)
                runDir, bad = eng.makeThetaMotorMaps(
                    newXml, repeat=repeat, steps=steps, delta=delta, fast=True)

        cmd.finish(f'Motor map sequence finished')

    def moveToHome(self, cmd):
        cmdKeys = cmd.cmd.keywords

        self.actor.visitor.setOrGetVisit(cmd)

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys
        allfiber = 'all' in cmdKeys

        if phi is True:
            eng.setPhiMode()
            self.cc.moveToHome(self.cc.goodCobras, phiEnable=True)

        if theta is True:
            eng.setThetaMode()
            self.cc.moveToHome(self.cc.goodCobras, thetaEnable=True)

        if allfiber is True:
            eng.setNormalMode()
            diff = self.cc.moveToHome(self.cc.goodCobras, thetaEnable=True, phiEnable=True, thetaCCW=False)

            self.logger.info(f'Averaged position offset comapred with cobra center = {np.mean(diff)}')

        cmd.finish(f'Move all arms back to home')

    def targetConverge(self, cmd):
        """ Making target convergence test. """
        cmdKeys = cmd.cmd.keywords
        runs = cmd.cmd.keywords['totalTargets'].values[0]
        maxsteps = cmd.cmd.keywords['maxsteps'].values[0]
        ontime = 'ontime' in cmdKeys
        speed = 'speed' in cmdKeys

        visit = self.actor.visitor.setOrGetVisit(cmd)

        eng.setNormalMode()
        self.logger.info(f'Moving cobra to home position')
        self.cc.moveToHome(self.cc.goodCobras, thetaEnable=True, phiEnable=True, thetaCCW=False)

        self.logger.info(f'Making transformation using home position')
        daytag = time.strftime('%Y%m%d')

        # We don't need to home phi again since there is a home sequence above.
        newXml = eng.convertXML2(f'{daytag}.xml', homePhi=False)

        self.logger.info(f'Using new XML = {newXml} as default setting')
        self.xml = newXml

        self.cc.loadModel(file=pathlib.Path(self.xml))
        # eng.setCobraCoach(self.cc)

        if ontime is True:

            self.logger.info(f'Run convergence test of {runs} targets with constant on-time')
            self.logger.info(f'Setting max step = {maxsteps}')
            eng.setConstantOntimeMode(maxSteps=maxsteps)

            targets, moves = eng.convergenceTest2(self.cc.goodIdx, runs=runs, thetaMargin=np.deg2rad(15.0),
                                                  phiMargin=np.deg2rad(15.0), thetaOffset=0,
                                                  phiAngle=(np.pi*5/6, np.pi/3, np.pi/4),
                                                  tries=16, tolerance=0.2, threshold=20.0,
                                                  newDir=True, twoSteps=False)

        if speed is True:
            self.logger.info(f'Run convergence test of {runs} targets with constant speed')
            self.logger.info(f'Setting max step = {maxsteps}')

            mmTheta = np.load('/data/MCS/20210505_016/data/thetaOntimeMap.npy')
            mmThetaSlow = np.load('/data/MCS/20210505_017/data/thetaOntimeMap.npy')
            mmPhi = np.load('/data/MCS/20210506_013/data/phiOntimeMap.npy')
            mmPhiSlow = np.load('/data/MCS/20210506_014/data/phiOntimeMap.npy')

            self.logger.info(f'On-time maps loaded.')

            mmTheta = self._cleanAnomaly(mmTheta)
            mmThetaSlow = self._cleanAnomaly(mmThetaSlow)
            mmPhi = self._cleanAnomaly(mmPhi)
            mmPhiSlow = self._cleanAnomaly(mmPhiSlow)

            eng.setConstantSpeedMaps(mmTheta, mmPhi, mmThetaSlow, mmPhiSlow)

            eng.setConstantSpeedMode(maxSegments=int({maxsteps}/100), maxSteps=100)

            self.logger.info(f'Setting maxstep = 100, nSeg = {int({maxsteps}/100)}')
            targets, moves = eng.convergenceTest2(cc.goodIdx, runs=runs,
                                                  thetaMargin=np.deg2rad(15.0), phiMargin=np.deg2rad(15.0),
                                                  thetaOffset=0, phiAngle=(np.pi*5/6, np.pi/3, np.pi/4),
                                                  tries=16, tolerance=0.2, threshold=20.0, newDir=True, twoSteps=False)

        cmd.finish(f'target convergece is finished')

    def angleConverge(self, cmd):
        """ Making comvergence test for a specific arm. """
        cmdKeys = cmd.cmd.keywords
        runs = cmd.cmd.keywords['angleTargets'].values[0]
        visit = self.actor.visitor.setOrGetVisit(cmd)

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi is True:

            self.logger.info(f'Run phi convergence test of {runs} targets')
            eng.setPhiMode()

            #eng.phiConvergenceTest(self.cc.goodIdx, runs={run}, tries=12, fast=False, tolerance=0.1)
            cmd.finish(f'angleConverge of phi arm is finished')
        else:
            self.logger.info(f'Run theta convergence test of {runs} targets')
            eng.setThetaMode()
            #eng.thetaConvergenceTest(self.cc.goodIdx, runs={run}, tries=12, fast=False, tolerance=0.1)
            cmd.finish(f'angleConverge of theta arm is finished')

    def moveToThetaAngleFromPhi60(self, cmd):
        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.
        """
        cmdKeys = cmd.cmd.keywords
        visit = self.actor.visitor.setOrGetVisit(cmd)

        phiAngle = 60.0
        tolerance = np.rad2deg(0.05)
        angle = (180.0 - phiAngle) / 2.0
        thetaAngles = np.full(len(self.allCobras), -angle, dtype='f4')
        thetaAngles[np.arange(0, self.nCobras, 2)] += 0
        thetaAngles[np.arange(1, self.nCobras, 2)] += 180

        dataPath, diffAngles, moves = eng.moveThetaAngles(cc.goodIdx, thetaAngles,
                                                          relative=False, local=True, tolerance=0.002, tries=12, fast=False, newDir=True)

        cmd.finish(f'gotoSafeFromPhi60 is finished')

    def movePhiToAngle(self, cmd):
        """ Making comvergence test for a specific arm. """
        cmdKeys = cmd.cmd.keywords
        angle = cmd.cmd.keywords['angle'].values[0]
        itr = cmd.cmd.keywords['iteration'].values[0]
        visit = self.actor.visitor.setOrGetVisit(cmd)

        if itr == 0:
            itr = 8

        self.logger.info(f'Move phi to angle = {angle}')

        # move phi to 60 degree for theta test
        dataPath, diffAngles, moves = eng.movePhiAngles(self.cc.goodIdx, np.deg2rad(angle),
                                                        relative=False, local=True, tolerance=0.002, tries=12, fast=False, newDir=True)

        self.logger.info(f'Data path : {dataPath}')
        cmd.finish(f'PHI is now at {angle} degress!!')

    def moveToSafePosition(self, cmd):
        """ Move cobras to nominal safe position: thetas OUT, phis in.
        Assumes phi is at 60deg and that we know thetaPositions.

        """
        visit = self.actor.visitor.setOrGetVisit(cmd)
        eng.moveToSafePosition(self.cc.goodIdx, tolerance=0.2,
                               tries=24, homed=False, newDir=False, threshold=20.0, thetaMargin=np.deg2rad(15.0))

        cmd.finish(f'gotoSafeFromPhi60 is finished')

    def motorOntimeSearch(self, cmd):
        """ FPS interface of searching the on time parameters for a specified motor speed """
        cmdKeys = cmd.cmd.keywords
        visit = self.actor.visitor.setOrGetVisit(cmd)

        # self._connect()

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi is True:
            day = time.strftime('%Y-%m-%d')
            newXml = f'{day}-phi_opt.xml'

            xml = eng.phiOnTimeSearch(newXml, speeds=(0.06, 0.12), steps=(500, 250), iteration=3, repeat=1)

            cmd.finish(f'motorOntimeSearch of phi arm is finished')
        else:
            day = time.strftime('%Y-%m-%d')
            newXml = f'{day}-theta_opt.xml'
            xml = eng.thetaOnTimeSearch(newXml, speeds=(0.06, 0.12), steps=[1000, 500], iteration=3, repeat=1)
            self.logger.info(f'Theta on-time optimal XML = {xml}')
            cmd.finish(f'motorOntimeSearch of theta arm is finished')

    def makeOntimeMap(self, cmd):
        """ Making on-time map. """
        cmdKeys = cmd.cmd.keywords
        visit = self.actor.visitor.setOrGetVisit(cmd)

        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi is True:

            self.logger.info(f'Running phi fast on-time scan.')
            dataPath, ontimes, angles, speeds = eng.phiOntimeScan(speed=np.deg2rad(0.12),
                                                                  steps=10, totalSteps=6000, repeat=1, scaling=4.0)

            self.logger.info(f'Running phi slow on-time scan.')
            dataPath, ontimes, angles, speeds = eng.phiOntimeScan(speed=np.deg2rad(0.06),
                                                                  steps=20, totalSteps=9000, repeat=1, scaling=4.0)
        else:
            self.logger.info(f'Running theta fast on-time scan.')
            dataPath, ontimes, angles, speeds = eng.thetaOntimeScan(speed=np.deg2rad(0.12), steps=10,
                                                                    totalSteps=10000, repeat=1, scaling=3.0)

            self.logger.info(f'Running theta slow on-time scan.')
            dataPath, ontimes, angles, speeds = eng.thetaOntimeScan(speed=np.deg2rad(0.06), steps=20,
                                                                    totalSteps=15000, repeat=1, scaling=3.0, tolerance=np.deg2rad(3.0))

        cmd.finish(f'Motor on-time scan is finished.')

    def moveToObsTarget(self, cmd):
        """ Move cobras to the pfsDesign. """
        visit = self.actor.visitor.setOrGetVisit(cmd)

        datapath = '/home/pfs/mhs/devel/ics_cobraCharmer/procedures/moduleTest/hyoshida/'
        target_file = f'{datapath}/pfsdesign_test_20201228_command_positions.csv'
        data = pd.read_csv(target_file)

        thetaAngle = data['theta(rad)'].values
        phiAngle = data['phi(rad)'].values

        targets = np.zeros([2394, 2])

        targets[:, 0] = thetaAngle
        targets[:, 1] = phiAngle

        positions = self.cc.pfi.anglesToPositions(self.cc.allCobras, thetaAngle, phiAngle)

        dataPath, atThetas, atPhis, moves = eng.moveThetaPhi(self.cc.goodIdx, thetaAngle[self.cc.goodIdx],
                                                             phiAngle[self.cc.goodIdx], relative=False, local=True, tolerance=0.2, tries=12, homed=False,
                                                             newDir=True, thetaFast=False, phiFast=False, threshold=20.0, thetaMargin=np.deg2rad(15.0))

        np.save(dataPath / 'positions', positions)
        np.save(dataPath / 'targets', targets)
        np.save(dataPath / 'moves', moves)

        cmd.finish(f'MoveToObsTarget sequence finished')

    def getAEfromFF(self, cmd, frameId):
        """ Checking distortion with fidicial fibers.  """

        #frameId= frameID
        moveId = 1

        offset = [0, -85]
        rotCent = [[4471], [2873]]

        telInform = self.nv.readTelescopeInform(frameId)
        za = 90-telInform['azi']
        inr = telInform['instrot']

        inr = inr-180
        if(inr < 0):
            inr = inr+360

        mcsData = self.nv.readCentroid(frameId, moveId)
        ffData = self.nv.readFFConfig()
        #sfData = self.nv.readCobraConfig()

        ffData['x'] -= offset[0]
        ffData['y'] -= offset[1]

        # correect input format
        xyin = np.array([mcsData['centroidx'], mcsData['centroidy']])

        # call the routine
        xyout = CoordTransp.CoordinateTransform(xyin, za, 'mcs_pfi', inr=inr, cent=rotCent)

        mcsData['pfix'] = xyout[0]
        mcsData['pfiy'] = xyout[1]

        #d = {'ffID': np.arange(len(xyout[0])), 'pfix': xyout[0], 'pfiy': xyout[1]}
        # transPos=pd.DataFrame(data=d)

        match = self._findHomes(ffData, mcsData)

        pts1 = np.zeros((1, len(match['orix']), 2))
        pts2 = np.zeros((1, len(match['orix']), 2))

        pts1[0, :, 0] = match['orix']
        pts1[0, :, 1] = match['oriy']

        pts2[0, :, 0] = match['pfix']
        pts2[0, :, 1] = match['pfiy']

        afCoeff, inlier = cv2.estimateAffinePartial2D(pts2, pts1)

        mat = {}
        mat['affineCoeff'] = afCoeff
        mat['xTrans'] = afCoeff[0, 2]
        mat['yTrans'] = afCoeff[1, 2]
        mat['xScale'] = np.sqrt(afCoeff[0, 0]**2+afCoeff[0, 1]**2)
        mat['yScale'] = np.sqrt(afCoeff[1, 0]**2+afCoeff[1, 1]**2)
        mat['angle'] = np.arctan2(afCoeff[1, 0]/np.sqrt(afCoeff[0, 0]**2+afCoeff[0, 1]**2),
                                  afCoeff[1, 1]/np.sqrt(afCoeff[1, 0]**2+afCoeff[1, 1]**2))

        self.tranMatrix = mat

    def applyAEonCobra(self, cmd, frameId):

        #frameId= 16493
        moveId = 1

        offset = [0, -85]
        rotCent = [[4471], [2873]]

        telInform = self.nv.readTelescopeInform(frameId)
        za = 90-telInform['azi']
        inr = telInform['instrot']
        inr = inr-180
        if(inr < 0):
            inr = inr+360

        mcsData = self.nv.readCentroid(frameId, moveId)
        sfData = self.nv.readCobraConfig()

        sfData['x'] -= offset[0]
        sfData['y'] -= offset[1]

        # correect input format
        xyin = np.array([mcsData['centroidx'], mcsData['centroidy']])

        # call the routine
        xyout = CoordTransp.CoordinateTransform(xyin, za, 'mcs_pfi', inr=inr, cent=rotCent)

        mcsData['pfix'] = xyout[0]
        mcsData['pfiy'] = xyout[1]

        pts2 = np.zeros((1, len(xyout[1]), 2))

        pts2[0, :, 0] = xyout[0]
        pts2[0, :, 1] = xyout[1]

        afCor = cv2.transform(pts2, self.tranMatrix['affineCoeff'])

        mcsData['pfix'] = afCor[0, :, 0]
        mcsData['pfiy'] = afCor[0, :, 1]

        match = self._findHomes(sfData, mcsData)

        match['dx'] = match['orix'] - match['pfix']
        match['dy'] = match['oriy'] - match['pfiy']

        return match

    def calculateBoresight(self, cmd):
        """ Function for calculating the rotation center """
        cmdKeys = cmd.cmd.keywords

        startFrame = cmdKeys["startFrame"].values[0]
        endFrame = cmdKeys["endFrame"].values[0]

        nframes = endFrame - startFrame + 1
        frameid = (np.arange(nframes)+startFrame)

        xCorner = []
        yCorner = []

        for i in frameid:
            mcsData = self.nv.readCentroid(i, 1)
            x = mcsData['centroidx']
            y = mcsData['centroidy']

            x0, x1, y0, y1 = fpstool.getCorners(x, y)
            xCorner.append(x0)
            yCorner.append(y0)

        xCorner = np.array(xCorner)
        yCorner = np.array(yCorner)

        coords = [xCorner, yCorner]
        xc, yc, r, _ = fpstool.least_squares_circle(xCorner, yCorner)

        data = {'visitid': startFrame, 'xc': xc, 'yc': yc}

        self.nv.writeBoresightTable(data)

        cmd.finish(f'mcsBoresight={xc:0.4f},{yc:0.4f}')

    def buildTransMatrix(self, cmd):
        """ Buiding transformation matrix using FF"""
        cmdKeys = cmd.cmd.keywords
        if 'frameId' in cmdKeys:
            frameId = cmdKeys['frameId'].values[0]

        self.logger.info(f'Build transformation matrix with FF on frame {frameId}')

        # Use this latest matrix as initial guess for automatic calculating.
        db = self.connectToDB(cmd)
        sql = f'''SELECT * from mcs_pfi_transformation 
            WHERE mcs_frame_id < {frameId} ORDER BY mcs_frame_id DESC
            FETCH FIRST ROW ONLY
            '''
        transMatrix = db.fetch_query(sql)
        scale = transMatrix['x_scale'].values[0]
        xoffset = transMatrix['x_trans'].values[0]
        yoffset = transMatrix['y_trans'].values[0]
        angle = transMatrix['angle'].values[0]
        self.logger.info(f'Latest matrix = {xoffset} {yoffset} scale = {scale}, angle={angle}')

        # Loading FF from DB
        ff_f3c = self.nv.readFFConfig()['x'].values+self.nv.readFFConfig()['y'].values*1j
        rx, ry = fpstool.projectFCtoPixel([ff_f3c.real, ff_f3c.imag], scale, angle, [xoffset, yoffset])

        # Load MCS data from DB
        self.logger.info(f'Load frame from DB')
        mcsData = self.nv.readCentroid(frameId)

        target = np.array([rx, ry]).T.reshape((len(rx), 2))
        source = np.array([mcsData['centroidx'].values, mcsData['centroidy'].values]
                          ).T.reshape((len(mcsData['centroidx'].values), 2))

        match = fpstool.pointMatch(target, source)
        ff_mcs = match[:, 0]+match[:, 1]*1j

        mat = fpstool.getAffineFromFF(ff_mcs, ff_f3c)

        dataInfo = {'mcs_frame_id': frameId,
                    'x_trans': mat['xTrans'],
                    'y_trans': mat['yTrans'],
                    'x_scale': mat['xScale'],
                    'y_scale': mat['yScale'],
                    'angle': np.rad2deg(mat['angle']),
                    'calculated_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

        self.logger.info(
            f"New matrix = {mat['xTrans']} {mat['yTrans']} scale = {mat['xScale']}, angle={np.rad2deg(mat['angle'])}")
        try:
            db.insert('mcs_pfi_transformation', pd.DataFrame(data=dataInfo, index=[0]))
        except:
            self.logger.info(f'Updating transformation matrix')
            db.update('mcs_pfi_transformation', pd.DataFrame(data=dataInfo, index=[0]))

        # Building the camera model
        self.logger.info(f"Building camera model")
        f3c_mcs_camModel, mcs_f3c_camModel = fpstool.buildCameraModelFF(ff_mcs, ff_f3c)

        cmd.finish('text="Building tranformation matrix finished"')

    def visCobraSpots(self, cmd):
        cmdKeys = cmd.cmd.keywords
        runDir = pathlib.Path(cmd.cmd.keywords['runDir'].values[0])
        self.logger.info(f'Loading model = {self.xml}')
        phi = 'phi' in cmdKeys
        theta = 'theta' in cmdKeys

        if phi:
            vis.visCobraSpots(runDir, self.xml, arm='phi')
