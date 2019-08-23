
import io
import pandas as pd
import psycopg2


class najaVenator:
    """ 
        A class of interface providing connection capability with opDB. 
        Naja is the genus name of cobra and venator is latin word for hunter.  
    
    """
    def __init__(self):
        
        self.db='db-ics'
        self.conn = None
        
        pass
    
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

    def readFFConfg(self):
        """ Read positions of all fidicial fibers"""

        conn = self.conn 
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

    def readCobraConfig(self):
        """ Read positions of all science fibers"""
        conn = self.conn 

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

    def readCentroid(self, frameId, moveId):
        """ Read centroid information from databse"""
        conn = self.conn 

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

    def __del__(self):
        if self.conn is not None:
            self.conn.close()
        pass

najaVenator = NajaVenator()