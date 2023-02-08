import numpy as np
import pandas as pd
import pfs.utils.ingestPfsDesign as ingestPfsDesign
import pfs.utils.pfsConfigUtils as pfsConfigUtils
from opdb import opdb
from pfs.datamodel import PfsDesign, PfsConfig
from pfs.utils.fiberids import FiberIds

__all__ = ["makeVanillaPfsConfig", "makeTargetsArray", "updatePfiNominal", "updatePfiCenter", "writePfsConfig",
           "ingestPfsConfig"]

pfsDesignDir = '/data/pfsDesign'


def makeVanillaPfsConfig(pfsDesignId, visit0):
    """Just make a PfsConfig file identical to PfsDesign."""
    pfsDesign = PfsDesign.read(pfsDesignId=pfsDesignId, dirName=pfsDesignDir)
    return PfsConfig.fromPfsDesign(pfsDesign=pfsDesign, visit=visit0, pfiCenter=pfsDesign.pfiNominal)


def makeTargetsArray(pfsConfig):
    """Construct target array from pfsConfig file."""
    allCobraIds = np.arange(2394, dtype='int32') + 1
    fiberId = pfsConfig.fiberId
    cobraId = FiberIds().fiberIdToCobraId(fiberId)
    # targets vector has an entry for each cobra and sorted by cobraId.
    targets = np.empty((2394, 2), dtype=pfsConfig.pfiNominal.dtype)
    targets[:] = np.NaN
    # cobraMask is boolean array(shape=cobraId.shape)
    cobraMask = np.isin(cobraId, allCobraIds)
    # only existing cobraId.
    cobraId = cobraId[cobraMask]
    # assigning target vector directly.
    targets[cobraId - 1] = pfsConfig.pfiNominal[cobraMask]
    isNan = np.logical_or(np.isnan(targets[:, 0]), np.isnan(targets[:, 1]))

    return targets[:, 0] + targets[:, 1] * 1j, isNan


def updatePfiNominal(pfsConfig, cmd=None):
    """Just a placeholder for now."""
    pass


def updatePfiCenter(pfsConfig, cmd=None):
    """Update final cobra positions after converging to pfsDesign."""

    def fetchFinalConvergence(visitId):
        """Retrieve final cobra position in mm.

        Parameters
        ----------
        visitId : `int`
            Convergence identifier.
        """
        sql = f'select pfs_visit_id, iteration, cobra_id, pfi_center_x_mm, pfi_center_y_mm from cobra_match cm where ' \
              f'cm.iteration=(select max(cm2.iteration) from cobra_match cm2 where cm2.pfs_visit_id = {visitId}) ' \
              f'and cm.pfs_visit_id={visitId} order by cobra_id asc'

        db = opdb.OpDB(hostname="db-ics", username="pfs", dbname="opdb")
        lastIteration = db.fetch_query(sql)
        return lastIteration

    # Retrieve dataset
    lastIteration = fetchFinalConvergence(pfsConfig.visit)
    # Fill final position with NaNs.
    pfiCenter = np.empty(pfsConfig.pfiNominal.shape, dtype=pfsConfig.pfiNominal.dtype)
    pfiCenter[:] = np.NaN
    # Construct the index.
    fiberId = FiberIds().cobraIdToFiberId(lastIteration.cobra_id.to_numpy())
    fiberIndex = pd.DataFrame(dict(fiberId=pfsConfig.fiberId, tindex=np.arange(len(pfsConfig.fiberId))))
    fiberIndex = fiberIndex.set_index('fiberId').loc[fiberId].tindex.to_numpy()
    # Set final cobra position.
    pfiCenter[fiberIndex, 0] = lastIteration.pfi_center_x_mm.to_numpy()
    pfiCenter[fiberIndex, 1] = lastIteration.pfi_center_y_mm.to_numpy()
    pfsConfig.pfiCenter = pfiCenter

    if cmd:
        cmd.inform('text="pfsConfig.pfiCenter updated successfully."')

    return lastIteration.iteration.max()


def writePfsConfig(pfsConfig, cmd=None):
    """Write final pfsConfig."""
    ret = pfsConfigUtils.writePfsConfig(pfsConfig)

    if cmd:
        cmd.inform(f'text="{pfsConfig.filename} written to disk')

    return ret


def ingestPfsConfig(pfsConfig, cmd=None, **kwargs):
    """Ingest PfsConfig file in opdb tables."""
    ret = ingestPfsDesign.ingestPfsConfig(pfsConfig, **kwargs)

    if cmd:
        cmd.inform('text="pfsConfig successfully inserted in opdb !"')

    return ret
