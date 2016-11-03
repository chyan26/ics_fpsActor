#!/usr/bin/env python

import json
import base64
import numpy
import time
import sys
sys.path.append("/home/chyan/mhs/devel/ics_fpsActor/python/fpsActor/mpsClient")
                
import opscore.protocols.keys as keys
import opscore.protocols.types as types
import pfi_interface as pfi

from opscore.utility.qstr import qstr

class FpsCmd(object):
    
    odometer=0
    
    def __init__(self, actor):
        # This lets us access the rest of the actor.
        self.actor = actor

        # Declare the commands we implement. When the actor is started
        # these are registered with the parser, which will call the
        # associated methods when matched. The callbacks will be
        # passed a single argument, the parsed and typed command.
        #
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('setupField', 'fieldID', self.setupField),
            ('setupOdometer', '<odo>', self.setupOdometer),
            ('testloop', '<cnt> [<expTime>]', self.testloop),
            ('home', '<cnt> [<expTime>]', self.home),
            ('dbinit', '', self.dbinit),
            ('runmpstest', '', self.runmpstest),
        ]

        # Define typed command arguments for the above commands.
        self.keys = keys.KeysDictionary("fps_fps", (1, 1),
                                        keys.Key("cnt", types.Int(), help="times to run loop"),
                                        keys.Key("odo", types.Int(), help="exposure odometer"),
                                        keys.Key("runmpstest", types.Int(), help="mps test"),
                                        keys.Key("fieldID", types.String(), 
                                                 help="ID for the field, which defines the fiber positions"),
                                        keys.Key("expTime", types.Float(), 
                                                 help="Seconds for exposure"))

    def ping(self, cmd):
        """Query the actor for liveness/happiness."""

        cmd.finish("text='Present and (probably) well'")

    def status(self, cmd):
        """Report status and version; obtain and send current data"""

        self.actor.sendVersionKey(cmd)

        keyStrings = ['text="nothing to say, really"']
        keyMsg = '; '.join(keyStrings)

        cmd.inform(keyMsg)
        cmd.diag(sys.path)
        cmd.diag('text="still nothing to say"')
        cmd.finish()
        
    def runmpstest(self, cmd):
        """Report status and version; obtain and send current data"""
        
        
        datastring=pfi.pack_mps_software(shutdown=False, restart=True, save_database=False)
        fileName='pack_mps_software.bin'

        with open(fileName, "wb") as f:
            f.write(datastring)


        p={'Module_Id':[0,1],'Positioner_Id':[2,2],'Current_Position_X':[0,0],'Current_Position_Y':[0,1],\
            'Target_Position_X':[10,20],'Target_Position_Y':[10,20], 'X_axes_Uncertainty':[0.2,0.2],\
            'Y_axes_Uncertainty':[0.2,0.2],'Joint1_Delay_Count':[0,1],'Joint2_Delay_Count':[0,1],'fixed_arm':[0,0],\
            'target_latched':[1,1]}

        datastring=pfi.pack_move_to_target(sequence_number=1, iteration_number=0, positions=p, obstacle_avoidance=0, enable_blind_move=0)
        fileName='pack_move_to_target.bin'
        
        with open(fileName, "wb") as f:
            f.write(datastring)


        keyStrings = ['text="MPS TEST nothing to say, really"']
        keyMsg = '; '.join(keyStrings)

        cmd.inform(keyMsg)
        cmd.diag('text="MPS TEST still nothing to say"')
        cmd.finish()


    def setupField(self, cmd):
        """ Fully configure all the fibers for the given field. """

        cmd.fail("text='Not yet implemented'")

    def home(self, cmd):
        """ Home the actuators. """

        cmd.fail("text='Not yet implemented'")

    def targetPositions(self, fieldName):
        """ return the (x,y) cobra positions for the given field.

        Obviously, you'd fetch from some database...
        """

        return numpy.random.random(9600).reshape(4800,2).astype('f4')
        
        cmd.finish()
        
    def testloop(self, cmd):
        """ Run the expose-move loop a few times. For development. """

        cnt = cmd.cmd.keywords["cnt"].values[0]
        expTime = cmd.cmd.keywords["expTime"].values[0] \
          if "expTime" in cmd.cmd.keywords \
          else 0.0


        times = numpy.zeros((cnt, 4), dtype='f8')
        
        targetPos = self.targetPositions("some field ID")
        for i in range(cnt):
            times[i,0] = time.time()

            # Fetch measured centroid from the camera actor
            cmdString = "centroid expTime=%0.1f" % (expTime)
            cmdVar = self.actor.cmdr.call(actor='mcs', cmdStr=cmdString,
                                          forUserCmd=cmd, timeLim=expTime+5.0)
            if cmdVar.didFail:
                cmd.fail('text=%s' % (qstr('Failed to expose with %s' % (cmdString))))
            #    return
            # Encoding will be encapsulated.
            rawCentroids = self.actor.models['mcs'].keyVarDict['centroidsChunk'][0]
            centroids = numpy.fromstring(base64.b64decode(rawCentroids), dtype='f4').reshape(2400,2)
            times[i,1] = time.time()

            # Command the actuators to move.
            cmdString = 'moveTo chunk=%s' % (base64.b64encode(targetPos.tostring()))
            cmdVar = self.actor.cmdr.call(actor='mps', cmdStr=cmdString,
                                          forUserCmd=cmd, timeLim=5.0)
            if cmdVar.didFail:
                cmd.fail('text=%s' % (qstr('Failed to move with %s' % (cmdString))))
                return
            times[i,2] = time.time()

            cmdVar = self.actor.cmdr.call(actor='mps', cmdStr="ping",
                                          forUserCmd=cmd, timeLim=5.0)
            if cmdVar.didFail:
                cmd.fail('text=%s' % (qstr('Failed to ping')))
                return
            times[i,3] = time.time()

        for i, itimes in enumerate(times):
            cmd.inform('text="dt[%d]=%0.4f, %0.4f, %0.4f"' % (i+1, 
                                                              itimes[1]-itimes[0],
                                                              itimes[2]-itimes[1],
                                                              itimes[3]-itimes[2],
                                                              ))
        cmd.inform('text="dt[mean]=%0.4f, %0.4f, %0.4f"' % ((times[:,1]-times[:,0]).sum()/cnt,
                                                            (times[:,2]-times[:,1]).sum()/cnt,
                                                            (times[:,3]-times[:,2]).sum()/cnt))
        cmd.inform('text="dt[max]=%0.4f, %0.4f, %0.4f"' % ((times[:,1]-times[:,0]).max(),
                                                           (times[:,2]-times[:,1]).max(),
                                                           (times[:,3]-times[:,2]).max()))
                                                            
        cmd.finish()
        
    def setupOdometer(self, cmd):
        """Setting the odometer number and start a data base table."""
        odometer = cmd.cmd.keywords["odo"].values[0]
        cmd.inform('"odometer"= %d'%(odometer))

        
        cmd.finish()
        
    def dbinit(self, cmd):
        """ Initializing the database.  """
  
        try:
            conn = psycopg2.connect("dbname='fps' user='pfs' host='localhost' password='pfs@hydra'")
        except:
            print "I am unable to connect to the database"
        
        cur = conn.cursor()
        cur.execute("CREATE TABLE FPS_INFO(runid VARCHAR(20) PRIMARY KEY," 
                                           "odometer INT, hst_time time, ut_time time," 
                                           "ra float8, dec float8, temp float4, fps_version VARCHAR(20),"
                                           "db_version VARCHAR(20))")  
        
        cur.execute("CREATE TABLE MORTORMAP_INFO(fibre_id INT PRIMARY KEY, odometer INT, mortormap_version VARCHAR(20)"  
                                           "mortormap_path VARCHAR(256), mortormap_date VARCHAR(20))") 
        
        conn.commit()
  
            
        cmd.finish("text='FPS database initializing finished.'")
          
