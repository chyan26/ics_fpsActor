from importlib import reload

import io
import os
import logging
import numpy as np
import pandas as pd
import psycopg2
import time
import datetime

from opdb import opdb
reload(opdb)

class NajaVenator(object):
    """ 
        A class of interface providing connection capability with opDB. 
        Naja is the genus name of cobra and venator is latin word for hunter.  

    """

    def __init__(self):

        self.db = 'db-ics'
        self._conn = None

        self._dbConn = opdb.OpDB(hostname='db-ics', dbname='opdb', username='pfs')

    @property
    def conn(self):
        if self._conn is not None:
            return self._conn
        pwpath = os.path.join(os.environ['ICS_FPSACTOR_DIR'],
                              "etc", "dbpasswd.cfg")

        try:
            file = open(pwpath, "r")
            passstring = file.read()
        except:
            raise RuntimeError(f"could not get db password from {pwpath}")

        try:
            connString = "dbname='opdb' user='pfs' host="+self.db+" password="+passstring
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

        cmd = f"""copy (select * from "fiducial_fiber_geometry" where fiducial_fiber_calib_id = 1) to stdout delimiter ',' """

        with conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0, 0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                            delimiter=',', usecols=(0, 1, 2))

        d = {'fiberID': arr[:, 0].astype('int32'), 'x': arr[:, 1].astype(
            'float'), 'y': arr[:, 2].astype('float')}

        df = pd.DataFrame(data=d)

        return df

    def readCobraConfig(self):
        """ Read positions of all science fibers"""
        conn = self.conn

        buf = io.StringIO()

        cmd = f"""copy (select * from "FiberPosition" where "ftype" = 'cobra') to stdout delimiter ',' """

        with conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0, 0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                            delimiter=',', usecols=(0, 1, 2, 3))

        d = {'fiberID': arr[:, 0].astype('int32'), 'x': arr[:, 2].astype(
            'float'), 'y': arr[:, 3].astype('int32')}

        df = pd.DataFrame(data=d)

        return df

    def readCentroidOld(self, frameId):
        """ Read centroid information from databse"""
        conn = self.conn

        buf = io.StringIO()

        cmd = f"""copy (select "mcs_frame_id", "spot_id", "mcs_center_x_pix", "mcs_center_y_pix" from "mcs_data"
                where "mcs_frame_id"={frameId}) to stdout delimiter ',' """

        with conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0, 0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype=[("mcsId", 'i4'), ("fiberID", 'i4'), ("centroidx", 'f4'), ("centroidy", 'f4')],
                            delimiter=',', usecols=(0, 1, 2, 3))

        df = pd.DataFrame(data=arr)

        return df

    def readCentroid(self, frameId):
        """ Read centroid information from database. This requires INSTRM-1110."""
        conn = self._dbConn

        sql = f"""select * from mcs_data where mcs_frame_id={frameId}"""
        df = conn.bulkSelect('mcs_data', sql)

        # We got a full table, with original names. Trim and rename to
        # what is expected here.
        renames = dict(mcs_frame_id='mcsId',
                       spot_id='fiberId',
                       mcs_center_x_pix='centroidx',
                       mcs_center_y_pix='centroidy')
        df = df[renames.keys()]
        df.rename(columns=renames, inplace=True)
        return df

    def readTelescopeInform(self, frameId):
        conn = self.conn

        buf = io.StringIO()

        cmd = f"""copy (select * from "mcs_exposure"
                where "mcs_frame_id"={frameId}) to stdout delimiter ',' """

        with conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        # conn.commit()
        buf.seek(0, 0)

        arr = np.genfromtxt(buf, dtype='f4',
                            delimiter=',', usecols=range(7))
        d = {'frameId': arr[0], 'alt': arr[3], 'azi': arr[4], 'instrot': arr[5]}

        return d

    def writeBoresightTable(self, data):
        conn = self.conn

        now = datetime.datetime.now()
        now.strftime("%Y-%m-%d %H:%M:%S")

        cmd = f""" INSERT INTO mcsboresight (visitid, datatime, x, y) \
        VALUES ({data['visitid']}, '{now.strftime("%Y-%m-%d %H:%M:%S")}',\
        {data['xc']}, {data['yc']}) """

        with conn:
            with conn.cursor() as curs:
                curs.execute(cmd)

    def writeCobraConfig(self, matchCatalog, frameid):
        conn = self.conn

        measBuf = io.StringIO()
        new = matchCatalog.dropna(thresh=6)
        new.to_csv(measBuf, sep=',', index=False, header=False)

        #np.savetxt(measBuf, centArr[:,1:7], delimiter=',', fmt='%0.6g')
        measBuf.seek(0, 0)

        colname = ['"fiberId"', '"mcsId"', '"pfiNominal_x"', '"pfiNominal_y"', '"pfiCenter_x"', '"pfiCenter_y"', '"mcsCenter_x"',
                   '"mcsCenter_y"', '"pfiDiff_x"', '"pfiDiff_y"', ]

        buf = io.StringIO()
        for l_i in range(len(new)):
            line = '%s' % (measBuf.readline())
            buf.write(line)
            # print(line)
        buf.seek(0, 0)

        with conn:
            with conn.cursor() as curs:
                curs.copy_from(buf, '"CobraConfig"', ',',
                               columns=colname)
        buf.seek(0, 0)

        return buf

    def writeTelescopeInform(self, data):
        pass

    def __del__(self):
        if self.conn is not None:
            self.conn.close()
        pass

class cobraTargetTable(object):
    def __init__(self, visitid, tries, calibModel):

        self.db = 'db-ics'
        self._conn = None

        self._dbConn = opdb.OpDB(hostname='db-ics', dbname='opdb', username='pfs')
        self.visitid = visitid
        self.tries = tries

        self.interation = 1
        self.calibModel = calibModel

    def makeTargetTable(self, moves, cobraCoach):
        """
        Make the target table for the convergence move.

        
        """
        
        cc = cobraCoach

        pfs_config_id = 0

        firstStepMove =  moves['position'][:,0]
        firstThetaAngle = moves['thetaAngle'][:,0]
        firstPhiAngle = moves['phiAngle'][:,0]



        targetStepMove =  moves['position'][:,2]
        targetThetaAngle = moves['thetaAngle'][:,2]
        targetPhiAngle = moves['phiAngle'][:,2]
        
        targetTable = {'pfs_visit_id':[],
                    'iteration':[],
                    'cobra_id':[],
                    'pfs_config_id':[],
                    'pfi_nominal_x_mm':[],
                    'pfi_nominal_y_mm':[],
                    'pfi_target_x_mm' :[],
                    'pfi_target_y_mm':[],
                    'motor_target_theta':[],
                    'motor_target_phi':[],
            }

        for iteration in range(self.tries):
            for idx in range(cc.nCobras):
                targetTable['pfs_visit_id'].append(self.visitid)
                targetTable['pfs_config_id'].append(pfs_config_id)

                targetTable['cobra_id'].append(idx+1)
                targetTable['iteration'].append(iteration+1)

                targetTable['pfi_nominal_x_mm'].append(self.calibModel.centers[idx].real)
                targetTable['pfi_nominal_y_mm'].append(self.calibModel.centers[idx].imag)

                if idx in cc.badIdx:
                    # Using cobra center for bad cobra targets
                    targetTable['pfi_target_x_mm'].append(self.calibModel.centers[idx].real)
                    targetTable['pfi_target_y_mm'].append(self.calibModel.centers[idx].imag)

                    targetTable['motor_target_theta'].append(0)
                    targetTable['motor_target_phi'].append(0)

                else: 
                    if iteration < 2:
                        targetTable['pfi_target_x_mm'].append(firstStepMove[cc.goodIdx == idx].real[0])
                        targetTable['pfi_target_y_mm'].append(firstStepMove[cc.goodIdx == idx].imag[0])
                        targetTable['motor_target_theta'].append(firstThetaAngle[cc.goodIdx == idx][0])
                        targetTable['motor_target_phi'].append(firstPhiAngle[cc.goodIdx == idx][0])
                    else:
                        targetTable['pfi_target_x_mm'].append(targetStepMove[cc.goodIdx == idx].real[0])
                        targetTable['pfi_target_y_mm'].append(targetStepMove[cc.goodIdx == idx].imag[0])
                        targetTable['motor_target_theta'].append(targetThetaAngle[cc.goodIdx == idx][0])
                        targetTable['motor_target_phi'].append(targetPhiAngle[cc.goodIdx == idx][0])

        self.dataTable = pd.DataFrame(targetTable)
        
        return self.dataTable

    def writeTargetTable(self):
        
        self._dbConn.insert("cobra_target", self.dataTable)

    #def __del__(self):
    #    pass
