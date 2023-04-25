import io
from importlib import reload

import numpy as np
import pandas as pd
import psycopg2
from opdb import opdb

reload(opdb)


class NajaVenator(object):
    """ 
        A class of interface providing connection capability with opDB. 
        Naja is the genus name of cobra and venator is latin word for hunter.  

    """

    def __init__(self):
        self._dbConn = opdb.OpDB(hostname='db-ics', dbname='opdb', username='pfs')

    @staticmethod
    def connect():
        """Return connection object, password needs to be defined in /home/user/.pgpass."""
        return psycopg2.connect(host='db-ics', dbname='opdb', user='pfs')

    def readFFConfig(self):
        """Read positions of all fidicial fibers."""
        buf = io.StringIO()
        cmd = f"""COPY (SELECT * from fiducial_fiber_geometry WHERE fiducial_fiber_calib_id = 1) to stdout delimiter ','"""

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0, 0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                            delimiter=',', usecols=(0, 1, 2))

        d = {'fiberID': arr[:, 0].astype('int32'), 'x': arr[:, 1].astype('float'), 'y': arr[:, 2].astype('float')}

        return pd.DataFrame(d)

    def readCobraConfig(self):
        """Read positions of all science fibers."""
        buf = io.StringIO()
        cmd = f"""COPY (SELECT * from "FiberPosition" WHERE "ftype" = 'cobra') to stdout delimiter ','"""

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0, 0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype='f4',
                            delimiter=',', usecols=(0, 1, 2, 3))

        d = {'fiberID': arr[:, 0].astype('int32'), 'x': arr[:, 2].astype('float'), 'y': arr[:, 3].astype('int32')}

        return pd.DataFrame(d)

    def readCentroidOld(self, frameId):
        """ Read centroid information from databse"""

        buf = io.StringIO()

        cmd = f"""COPY (SELECT mcs_frame_id, spot_id, mcs_center_x_pix, mcs_center_y_pix from mcs_data WHERE mcs_frame_id={frameId}) to stdout delimiter ','"""

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0, 0)

        # Skip the frameId, etc. columns.
        arr = np.genfromtxt(buf, dtype=[("mcsId", 'i4'), ("fiberID", 'i4'), ("centroidx", 'f4'), ("centroidy", 'f4')],
                            delimiter=',', usecols=(0, 1, 2, 3))

        return pd.DataFrame(arr)

    def readCentroid(self, frameId):
        """ Read centroid information from database. This requires INSTRM-1110."""
        conn = self._dbConn

        sql = f"""SELECT * from mcs_data WHERE mcs_frame_id={frameId}"""
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
        buf = io.StringIO()

        cmd = f"""COPY (SELECT * from mcs_exposure WHERE mcs_frame_id={frameId}) to stdout delimiter ','"""

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.copy_expert(cmd, buf)
        buf.seek(0, 0)

        arr = np.genfromtxt(buf, dtype='f4', delimiter=',', usecols=range(7))
        arr = arr[[0, 3, 4, 5]].reshape(1, 4)

        return pd.DataFrame(arr, columns=['frameId', 'alt', 'azi', 'instrot'])

    def writeBoresightTable(self, data):
        cmd = f""" INSERT INTO mcs_boresight (pfs_visit_id, mcs_boresight_x_pix, mcs_boresight_y_pix, calculated_at)  
        VALUES ({data['visitid']}, {data['xc']}, {data['yc']}, 'now')"""

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.execute(cmd)

    def writeCobraConfig(self, matchCatalog, frameid):
        measBuf = io.StringIO()
        new = matchCatalog.dropna(thresh=6)
        new.to_csv(measBuf, sep=',', index=False, header=False)

        # np.savetxt(measBuf, centArr[:,1:7], delimiter=',', fmt='%0.6g')
        measBuf.seek(0, 0)

        colname = ['"fiberId"', '"mcsId"', '"pfiNominal_x"', '"pfiNominal_y"', '"pfiCenter_x"', '"pfiCenter_y"',
                   '"mcsCenter_x"', '"mcsCenter_y"', '"pfiDiff_x"', '"pfiDiff_y"', ]

        buf = io.StringIO()
        for l_i in range(len(new)):
            buf.write(str(measBuf.readline()))

        buf.seek(0, 0)

        with self.connect() as conn:
            with conn.cursor() as curs:
                curs.COPY_from(buf, '"CobraConfig"', ',', columns=colname)

        buf.seek(0, 0)

        return buf

    def writeTelescopeInform(self, data):
        pass


class CobraTargetTable(object):
    def __init__(self, visitid, tries, calibModel):
        self._dbConn = opdb.OpDB(hostname='db-ics', dbname='opdb', username='pfs')
        self.visitid = visitid
        self.tries = tries

        self.interation = 1
        self.calibModel = calibModel

    def makeTargetTable(self, moves, cobraCoach, goodIdx):
        """Make the target table for the convergence move."""
        cc = cobraCoach

        pfs_config_id = 0

        firstStepMove = moves['position'][:, 0]
        firstThetaAngle = moves['thetaAngle'][:, 0]
        firstPhiAngle = moves['phiAngle'][:, 0]

        targetStepMove = moves['position'][:, 2]
        targetThetaAngle = moves['thetaAngle'][:, 2]
        targetPhiAngle = moves['phiAngle'][:, 2]

        targetTable = {'pfs_visit_id': [],
                       'iteration': [],
                       'cobra_id': [],
                       'pfs_config_id': [],
                       'pfi_nominal_x_mm': [],
                       'pfi_nominal_y_mm': [],
                       'pfi_target_x_mm': [],
                       'pfi_target_y_mm': [],
                       'motor_target_theta': [],
                       'motor_target_phi': [],
                       }

        for iteration in range(self.tries):
            for idx in range(cc.nCobras):
                targetTable['pfs_visit_id'].append(self.visitid)
                targetTable['pfs_config_id'].append(pfs_config_id)

                targetTable['cobra_id'].append(idx + 1)
                targetTable['iteration'].append(iteration + 1)

                targetTable['pfi_nominal_x_mm'].append(self.calibModel.centers[idx].real)
                targetTable['pfi_nominal_y_mm'].append(self.calibModel.centers[idx].imag)

                if idx in cc.badIdx or idx not in goodIdx:
                    # Using cobra center for bad cobra targets
                    targetTable['pfi_target_x_mm'].append(self.calibModel.centers[idx].real)
                    targetTable['pfi_target_y_mm'].append(self.calibModel.centers[idx].imag)

                    targetTable['motor_target_theta'].append(0)
                    targetTable['motor_target_phi'].append(0)

                else:
                    if iteration < 2:
                        targetTable['pfi_target_x_mm'].append(firstStepMove[goodIdx == idx].real[0])
                        targetTable['pfi_target_y_mm'].append(firstStepMove[goodIdx == idx].imag[0])
                        targetTable['motor_target_theta'].append(firstThetaAngle[goodIdx == idx][0])
                        targetTable['motor_target_phi'].append(firstPhiAngle[goodIdx == idx][0])
                    else:
                        targetTable['pfi_target_x_mm'].append(targetStepMove[goodIdx == idx].real[0])
                        targetTable['pfi_target_y_mm'].append(targetStepMove[goodIdx == idx].imag[0])
                        targetTable['motor_target_theta'].append(targetThetaAngle[goodIdx == idx][0])
                        targetTable['motor_target_phi'].append(targetPhiAngle[goodIdx == idx][0])

        self.dataTable = pd.DataFrame(targetTable)

        return self.dataTable

    def writeTargetTable(self):
        """Write self.dataTable to cobra_target table."""
        self._dbConn.insert("cobra_target", self.dataTable)
