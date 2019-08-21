import pathlib
import sys

import numpy as np
import psycopg2
import io
import pandas as pd

import opscore.protocols.keys as keys
import opscore.protocols.types as types

from opscore.utility.qstr import qstr

from ics.fpsActor import fpsState

class FpsCmd(object):
    def __init__(self, actor):
        # This lets us access the rest of the actor.
        self.actor = actor
    
        self.db='db-ics'
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
            ('testCamera', '[<cnt>] [<expTime>] [@noCentroids]', self.testCamera),
            ('testLoop', '<cnt> [<expTime>]', self.testLoop),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("id", types.Long(),
                                                 help="fpsDesignId for the field, "
                                                      "which defines the fiber positions"),
                                        keys.Key("expTime", types.Float(), 
                                                 help="Seconds for exposure"))

    @property
    def conn(self):
        if self._conn is not None:
            return self._conn

        pwpath=os.path.join(os.environ['ICS_MCSACTOR_DIR'],
                            "etc", "dbpasswd.cfg")

        try:
            file = open(pwpath, "r")
            passstring = file.read()
        except:
            raise RuntimeError(f"could not get db password from {pwpath}")

        try:
            connString = "dbname='opdb' user='pfs' host="+self.db+" password="+passstring
            self.actor.logger.info(f'connecting to {connString}')
            conn = psycopg2.connect(connString)
            self._conn = conn
        except Exception as e:
            raise RuntimeError("unable to connect to the database {connString}: {e}")

        return self._conn

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

    def _mcsExpose(self, cmd, expTime=None, doCentroid=True):
        """ Request a single MCS exposure, with centroids by default.

        Args
        ----
        cmd : `actorcore.Command`
          What we report back to.
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
                                      forUserCmd=cmd, timeLim=expTime+10)
        if cmdVar.didFail:
            cmd.warn('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))
            return None

        filekey= self.actor.models['mcs'].keyVarDict['filename'][0]
        filename = pathlib.Path(filekey)
        visit = int(filename.stem[4:], base=10)
        cmd.inform(f'visit={visit}')

        return visit

    def _readFFPosition(self, conn):
        """ Read positions of all fidicial fibers"""
    
        buf = io.StringIO()

        cmd = f"""copy (select * from "FiducialFiberPosition") to stdout delimiter ',' """

        with conn.cursor() as curs:
            curs.copy_expert(cmd, buf)
        conn.commit()
        buf.seek(0,0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                    delimiter=',',usecols=(0,1,6,7))


        d = {'ffID': arr[:,0], 'fiberID': arr[:,0], 'x': arr[:,1], 'y':arr[:,2]}

        df=pd.DataFrame(data=d)


        return df

    def _readSFPosition(self, conn):
        """ Read positions of all science fibers"""
    
        buf = io.StringIO()

        cmd = f"""copy (select * from "FiberPosition") to stdout delimiter ',' """

        with conn.cursor() as curs:
            curs.copy_expert(cmd, buf)
        conn.commit()
        buf.seek(0,0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                    delimiter=',',usecols=(0,1,2))

        d = {'fiberID': arr[:,0], 'x': arr[:,1], 'y':arr[:,2]}

        df=pd.DataFrame(data=d)


        return df

    def _readCentroid(self, conn, frameId, moveId):
        """ Read centroid information from databse"""
    
        buf = io.StringIO()

        cmd = f"""copy (select "fiberId", "centroidx", "centroidy" from "mcsData"
                where frameId={frameId} and moveId={moveId}) to stdout delimiter ',' """

        with conn.cursor() as curs:
            curs.copy_expert(cmd, buf)
        conn.commit()
        buf.seek(0,0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                    delimiter=',',usecols=(0,1,2))

        d = {'fiberID': arr[:,0], 'centroidx': arr[:,1], 'centroidy':arr[:,2]}

        df=pd.DataFrame(data=d)


        return df


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
        cnt = cmdKeys["cnt"].values[0] \
              if 'cnt' in cmdKeys \
                 else 7
        expTime = cmdKeys["expTime"].values[0] \
            if "expTime" in cmdKeys \
            else None

        for i in range(cnt):
            cmd.inform(f'text="taking exposure loop {i+1}/{cnt}"')
            visit = self._mcsExpose(cmd, expTime=expTime, doCentroid=True)
            if not visit:
                cmd.fail('text="exposure failed"')
                return

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

