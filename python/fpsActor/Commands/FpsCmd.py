import json
import base64
import numpy
import time
import psycopg2
import sys
sys.path.append("/home/chyan/mhs/devel/ics_fpsActor/python/fpsActor/mpsClient")

                
import opscore.protocols.keys as keys
import opscore.protocols.types as types
import pfi_interface as pfi

from opscore.utility.qstr import qstr

class FpsCmd(object):
    
    odometer=0
    fieldid=0
    f3ctarget=[]
    
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
            ('setField', 'fieldID', self.setField),
            ('setOdometer', '<odo>', self.setOdometer),
            ('getOdometer', '', self.getOdometer),
            ('testloop', '<cnt> [<expTime>]', self.testloop),
            ('home', '<cnt> [<expTime>]', self.home),
            ('dbinit', '', self.dbinit),
            ('targetPositions', '', self.targetPositions),
            ('sendCommand', '', self.sendCommand),
            ('getResponse', '', self.getResponse),
            ('runmpsdianostic', '', self.getResponse),
            ('runmpstest', '', self.runmpstest),
            ('gohomeall', '', self.gohomeall),
            ('movetotarget', '', self.movetotarget),
            ('movepositioner', '', self.movetpositioner),
            ('gettelemetrydata', '', self.gettelemetrydata),
            ('setcurrentposition', '', self.setcurrentposition),

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
        
    def sendCommand(self,cmd):
         """ sending command to MPS host """
         cmd.diag('text="Sending command to MPS server."')
     
         mpshost=""
         mpsport=8888
         
         m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)
         data=""
         
         m.send_command(data)
         
         cmd.diag('text="Sending command to MPS server finished."')
         cmd.finish()

    def getResponse(self,cmd):
         """ sending command to MPS host """
         cmd.diag('text="Sending command to MPS server."')
     
         mpshost=""
         mpsport=8888
         
         m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)
         size=20
         
         data=m.get_response(size)
         
         cmd.diag('text="Sending command to MPS server finished."')
         cmd.finish()

    def gohomeall(self,cmd):
        """ Home all the science fibres """
            
        mpshost=""
        mpsport=8888
        
        m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)
        telemetry=m.go_home_all(obstacle_avoidance=True, enable_blind_move=False, j1_use_fast_map=False, j2_use_fast_map=False)
        
        cmd.diag('text="Go_Home_All command finished."')
        cmd.finish()
        
    def movetotarget(self,cmd):   
        """ Move science fibres to certain place"""

        mpshost=""
        mpsport=8888
        
        #m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)

        p={'Module_Id':[1,2],'Positioner_Id':[2,2],'Current_Position_X':[0,0],'Current_Position_Y':[0,1],\
            'Target_Position_X':[10,20],'Target_Position_Y':[10,20], 'X_axes_Uncertainty':[0.2,0.2],\
            'Y_axes_Uncertainty':[0.2,0.2],'Theta_Joint1_Delay_Count':[0,1],'Phi_Joint2_Delay_Count':[0,1],'Flags':[0,0]}
        
        telemetry=m.move_to_target(sequence_number=0, iteration_number=0, positions=p, obstacle_avoidance=0, enable_blind_move=0)
        
        telemetry=m.move_to_target(sequence_number=0, iteration_number=1, positions=p, obstacle_avoidance=0, enable_blind_move=0)
        
        
        cmd.diag('text="MPS move_to_target command finished."')
        cmd.finish()    
        
    def movepositioner(self,cmd):
        """ Move positioner without collision checking. """
        
        mpshost=""
        mpsport=8888
        
        #m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)

        p={'Module_Id':[1,2],'Positioner_Id':[2,2],'Theta-Joint1':[10,20],'Phi-Joint2':[10,20],'Flags':[0,0]}
        m=mps.move_positioner(p)
        
        cmd.diag('text="move_positioner command finished."')
        cmd.finish()    
        
    def gettelemetrydata(self):
        """ Getting telemetry data """
        
        mpshost=""
        mpsport=8888
        
        #m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)
        p={'Module_Id':[1,2],'Positioner_Id':[2,2]}
        
        telemetry=m.get_telemetry_data(p)
        
        # Then, we can do something here
        
        cmd.diag('text="get_telemetry_data command finished."')
        cmd.finish()  
     
    def setcurrentpositions(self):
        """ Set current position """
        mpshost=""
        mpsport=8888
        
        #m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)
        p={'Module_Id':[1,2],'Positioner_Id':[2,2],'Current_Position_X':[0,0],'Current_Position_Y':[0,1],'Flags':[0,0]}
        m.set_current_position(p)
            
        cmd.diag('text="set_current_position command finished."')
        cmd.finish()  
        
    def runmpsdianostic(self,cmd):
        """ Asking MPS to run system disnostic"""
        
        mpshost=""
        mpsport=8888
    
        m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)
        m.run_diagnostic()
        
        cmd.diag('text="MPS dianostic command finished."')
        cmd.finish()            
    
    def setpowerorreset(self,cmd):
        
        mpshost=""
        mpsport=8888
        m=mps.MPSClient(host=mpshost,port=mpsport,command_header_counter=0)
        
        m.set_power_or_reset(cmd, set_motor_freq, sectors)
        
        cmd.diag('text="MPS set_power_or_rest command finished."')
        cmd.finish()
        
    def runmpstest(self, cmd):
        """Sequence of testing commands."""
        
        
        datastring=pfi.pack_mps_software(shutdown=False, restart=True, save_database=False)
        fileName='pack_mps_software.bin'

        with open(fileName, "wb") as f:
            f.write(datastring)


        p={'Module_Id':[1,2],'Positioner_Id':[2,2],'Current_Position_X':[0,0],'Current_Position_Y':[0,1],\
            'Target_Position_X':[10,20],'Target_Position_Y':[10,20], 'X_axes_Uncertainty':[0.2,0.2],\
            'Y_axes_Uncertainty':[0.2,0.2],'Theta_Joint1_Delay_Count':[0,1],'Phi_Joint2_Delay_Count':[0,1],'Flags':[0,0]}

        datastring=pfi.pack_move_to_target(sequence_number=1, iteration_number=0, positions=p, obstacle_avoidance=0, enable_blind_move=0)
        fileName='pack_move_to_target.bin'
        
        with open(fileName, "wb") as f:
            f.write(datastring)


        keyStrings = ['text="MPS TEST nothing to say, really"']
        keyMsg = '; '.join(keyStrings)

        cmd.inform(keyMsg)
        cmd.diag('text="MPS TEST still nothing to say"')
        cmd.finish()


    def setField(self, cmd):
        """ Fully configure all the fibers for the given field. """

        cmd.fail("text='Not yet implemented'")

    def home(self, cmd):
        """ Home the actuators. """

        cmd.fail("text='Not yet implemented'")

    def targetPositions(self, fieldName):
        """ return the (x,y) cobra positions for the given field.
        Obviously, you'd fetch from some database...
        """

        self.f3ctarget=numpy.random.random(5000).reshape(2500,2).astype('f4')
        
        fieldName.finish()
        
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
        
    def setOdometer(self, cmd):
        """Setting the odometer number and start a data base table."""
        self.odometer = cmd.cmd.keywords["odo"].values[0]
        cmd.inform('text="odometer = "%i'%(self.odometer))

        
        cmd.finish()
    
    def getOdometer(self, cmd):
        """Getting the odometer number and start a data base table."""
        #odometer = cmd.cmd.keywords["odo"].values[0]
        cmd.inform('text="odometer = "%i'%(self.odometer))

        
        cmd.finish()    
        
    def dbinit(self, cmd):
        """ Initializing the database tables.  """
  
        try:
            conn = psycopg2.connect("dbname='fps' user='pfs' host='localhost' password='pfs@hydra'")
        except:
            print "I am unable to connect to the database"
        
        cur = conn.cursor()
        cur.execute("select * from information_schema.tables where table_name=%s", ('TARGET',))
        if bool(cur.rowcount) is False:     
            cur.execute("CREATE TABLE TARGET(odometer INT, runid VARCHAR(20), fid int, f3c_x float8, f3c_y float8,"
                        "flag INT)")
            conn.commit()

        cur.execute("select * from information_schema.tables where table_name=%s", ('FPS_INFO',))
        if bool(cur.rowcount) is False:
            cur.execute("CREATE TABLE FPS_INFO(runid VARCHAR(20) PRIMARY KEY," 
                        "odometer INT, hst_time time, ut_time time," 
                        "ra float8, dec float8, temp float4, fps_version VARCHAR(20),"
                        "db_version VARCHAR(20))")  
            conn.commit()

        cur.execute("select * from information_schema.tables where table_name=%s", ('MORTORMAP_INFO',))
        if bool(cur.rowcount) is False:
            cur.execute("CREATE TABLE MORTORMAP_INFO(fibre_id INT PRIMARY KEY, odometer INT, mortormap_version VARCHAR(20),"  
                        "mortormap_path VARCHAR(256), mortormap_date VARCHAR(20))") 
        
            conn.commit()
        conn.close()
            
        cmd.finish("text='FPS database initializing finished.'")