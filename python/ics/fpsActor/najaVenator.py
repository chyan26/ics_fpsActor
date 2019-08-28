
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

        with conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0,0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                    delimiter=',',usecols=(0,1,2,3))


        d = {'fiberID': arr[:,0].astype('int32'), 'x': arr[:,2].astype('float'), 'y':arr[:,3].astype('float')}

        df=pd.DataFrame(data=d)


        return df

    def readCobraConfig(self):
        """ Read positions of all science fibers"""
        conn = self.conn 

        buf = io.StringIO()

        cmd = f"""copy (select * from "FiberPosition" where "ftype" = 'cobra') to stdout delimiter ',' """

        with conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0,0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                    delimiter=',',usecols=(0,1,2,3))

        d = {'fiberID': arr[:,0].astype('int32'), 'x': arr[:,2].astype('float'), 'y':arr[:,3].astype('int32')}

        df=pd.DataFrame(data=d)
       

        return df

    def readCentroid(self, frameId, moveId):
        """ Read centroid information from databse"""
        conn = self.conn 

        buf = io.StringIO()

        cmd = f"""copy (select "mcsId", "fiberId", "centroidx", "centroidy" from "mcsData"
                where "frameId"={frameId} and "moveId"={moveId}) to stdout delimiter ',' """

        with conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0,0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype=[("mcsId",'i4'),("fiberID",'i4'),("centroidx",'f4'),("centroidy",'f4')],
                    delimiter=',',usecols=(0,1,2,3))

        df=pd.DataFrame(data=arr)
        
        return df

    def readTelescopeInform(self, frameId):
        conn = self.conn 

        buf = io.StringIO()
        
        cmd = f"""copy (select * from "mcsexposure"
                where "frameid"={frameId}) to stdout delimiter ',' """

        with conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        #conn.commit()
        buf.seek(0,0)

        arr = np.genfromtxt(buf, dtype='f4',
                            delimiter=',',usecols=range(7))
        #print(arr.shape)
        d = {'frameId': arr[1], 'alt': arr[4], 'azi': arr[5], 'instrot':arr[6]}

        return d

    def writeCobraConfig(self,matchCatalog,frameid):
        conn = self.conn 

        measBuf = io.StringIO()
        new=matchCatalog.dropna(thresh=6)
        new.to_csv(measBuf,sep=',',index=False, header=False) 

        #np.savetxt(measBuf, centArr[:,1:7], delimiter=',', fmt='%0.6g')
        measBuf.seek(0,0)


        #connString = "dbname='opdb' user='pfs' host=db-ics"
                    # Skipself.actor.logger.info(f'connecting to {connString}')
        #conn = psycopg2.connect(connString)


        colname = ['"fiberId"','"mcsId"', '"pfiNominal_x"', '"pfiNominal_y"','"pfiCenter_x"', '"pfiCenter_y"','"mcsCenter_x"',
        '"mcsCenter_y"','"pfiDiff_x"','"pfiDiff_y"',]
            
        buf = io.StringIO()
        for l_i in range(len(new)):
            line = '%s' % (measBuf.readline())
            buf.write(line)
            #print(line)
        buf.seek(0,0)

        with conn:
            with conn.cursor() as curs:
                curs.copy_from(buf,'"CobraConfig"',',',
                        columns=colname)

        buf.seek(0,0)

        
        #cmd.inform('text="Cobra config for frame %s populated."' % (frameid))

        return buf
    
    def writeTelescopeInform(self, data):
        pass

    def __del__(self):
        if self.conn is not None:
            self.conn.close()
        pass

#NajaVenator = najaVenator()
