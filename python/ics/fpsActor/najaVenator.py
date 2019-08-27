
import io
import os
import numpy as np
import pandas as pd
import psycopg2


class NajaVenator(object):
    """ 
        A class of interface providing connection capability with opDB. 
        Naja is the genus name of cobra and venator is latin word for hunter.  
    
    """
    def __init__(self):
        
        self.db='db-ics'
        self._conn = None
        
    
    @property
    def conn(self):
        if self._conn is not None:
            return self._conn

        try:
            connString = "dbname='opdb' user='pfs' host="+self.db
            # Skipself.actor.logger.info(f'connecting to {connString}')
            conn = psycopg2.connect(connString)
            self._conn = conn
        except Exception as e:
            raise RuntimeError(f"unable to connect to the database {connString}: {e}")

        return self._conn

    def readFFConfig(self):
        """ Read positions of all fidicial fibers"""

        conn = self.conn 
        buf = io.StringIO()

        cmd = f"""copy (select * from "FiberPosition" where "ftype" = 'ff') to stdout delimiter ',' """

        with conn.cursor() as curs:
            curs.copy_expert(cmd, buf)
        conn.commit()
        buf.seek(0,0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                    delimiter=',',usecols=(0,1,2,3))


        d = {'fiberID': arr[:,0], 'x': arr[:,2], 'y':arr[:,3]}

        df=pd.DataFrame(data=d)


        return df

    def readCobraConfig(self):
        """ Read positions of all science fibers"""
        conn = self.conn 

        buf = io.StringIO()

        cmd = f"""copy (select * from "FiberPosition" where "ftype" == "cobra") to stdout delimiter ',' """

        with conn.cursor() as curs:
            curs.copy_expert(cmd, buf)
        conn.commit()
        buf.seek(0,0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                    delimiter=',',usecols=(0,1,2,3))

        d = {'fiberID': arr[:,0], 'x': arr[:,2], 'y':arr[:,3]}

        df=pd.DataFrame(data=d)


        return df

    def readCentroid(self, frameId, moveId):
        """ Read centroid information from databse"""
        conn = self.conn 

        buf = io.StringIO()

        cmd = f"""copy (select "fiberId", "centroidx", "centroidy" from "mcsData"
                where "frameId"={frameId} and "moveId"={moveId}) to stdout delimiter ',' """

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

    def readTelescopeInform(self, frameId):
        conn = self.conn 

        buf = io.StringIO()
        
        cmd = f"""copy (select * from "mcsexposure"
                where "frameid"={frameId}) to stdout delimiter ',' """

        with conn.cursor() as curs:
            curs.copy_expert(cmd, buf)
        conn.commit()
        buf.seek(0,0)

        arr = np.genfromtxt(buf, dtype='f4',
                            delimiter=',',usecols=range(7))
        #print(arr.shape)
        d = {'frameId': arr[1], 'alt': arr[4], 'azi': arr[5], 'instrot':arr[6]}

        return d
        
    def writeTelescopeInform(self, data):

        pass

    def __del__(self):
        if self.conn is not None:
            self.conn.close()
        pass

#NajaVenator = najaVenator()
