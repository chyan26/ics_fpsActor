import json
import numpy as np
import psycopg2
import psycopg2.extras
import sys

import opscore.protocols.keys as keys
import opscore.protocols.types as types

from opscore.utility.qstr import qstr

from .. import fpsState

class FpsCmd(object):
    def __init__(self, actor):
        # This lets us access the rest of the actor.
        self.actor = actor
        self
        # Declare the commands we implement. When the actor is started
        # these are registered with the parser, which will call the
        # associated methods when matched. The callbacks will be
        # passed a single argument, the parsed and typed command.
        #
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('loadDesign', 'id', self.loadDesign),
            ('moveToDesign', '', self.moveToDesign),
            ('cameraTest', '<cnt> [<expTime>] [@noCentroids]', self.cameraTest),
            ('testloop', '<cnt> [<expTime>]', self.testloop),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("id", types.Long(),
                                                 help="fpsDesignId for the field, which defines the fiber positions"),
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

    def loadDesign(self, cmd):
        """ Load our design from the given pfsDesignId. """

        designId = cmd.cmd.keywords['id'].values[0]

        ret = self.loadField(designId)

        cmd.fail("text='Not yet implemented'")

    def moveToDesign(self,cmd):
        """ Move cobras to the pfsDesign. """

        raise NotImplementedError('moveToDesign')
        cmd.finish()

    def _mcsExpose(self, cmd, expTime=None, doCentroid=True):
        """ Request a single exposure. """

        cmdString = "expose object expTime=%0.1f %s" % (expTime,
                                                        'doCentroid' if doCentroids else '')
        cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                      forUserCmd=cmd, timeLim=expTime+10)
        if cmdVar.didFail:
            cmd.warn('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))
            return False
        return True

    def cameraTest(self, cmd):
        """ Camera Loop Test. """

        cmdKeys = cmd.cmd.keywords
        cnt = cmdKeys["cnt"].values[0]
        expTime = cmdKeys["expTime"].values[0] \
            if "expTime" in cmdKeys \
            else 1.0
        doCentroid = 'noCentroids' not in cmdKeys

        for i in range(cnt):
            cmd.inform(f'text="taking exposure loop {i+1}/cnt"')
            ret = self._mcsExpose(cmd, expTime=expTime, doCentroid=doCentroid)
            if not ret:
                cmd.fail('text="exposure failed"')
                return

        cmd.finish()

    def loopTest(self, cmd):
        """ Run the expose-move loop a few times. For development. """

        cmdKeys = cmd.cmd.keywords
        cnt = cmdKeys["cnt"].values[0]
        expTime = cmdKeys["expTime"].values[0] \
            if "expTime" in cmdKeys \
            else 1.0

        for i in range(cnt):
            cmd.inform('text="loop = "%i'%(i))
            cmdString = "centroidOnDummy expTime=%0.1f" % (expTime)
            cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                          forUserCmd=cmd, timeLim=expTime)

            if cmdVar.didFail:
                 cmd.fail('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))

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

