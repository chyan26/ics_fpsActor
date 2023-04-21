import os

import numpy as np
import pandas as pd
import pfs.utils.pfsDesignUtils as pfsDesignUtils
from pfs.datamodel import TargetType, PfsDesign
from pfs.utils.fiberids import FiberIds

pfsDesignDir = '/data/pfsDesign'


def readDesign(pfsDesignId):
    """Read PfsDesign from pfsDesignDir."""
    return PfsDesign.read(pfsDesignId=pfsDesignId, dirName=pfsDesignDir)


def writeDesign(pfsDesign):
    """Write PfsDesign to pfsDesignDir, do not override."""
    fullPath = os.path.join(pfsDesignDir, pfsDesign.filename)
    doWrite = not os.path.isfile(fullPath)

    if doWrite:
        pfsDesign.write(pfsDesignDir)

    return doWrite, fullPath


def createHomeDesign(calibModel, goodIdx):
    """Create home design from current calibModel, ra and dec are faked."""
    gfm = pd.DataFrame(FiberIds().data)
    sgfm = gfm.set_index('scienceFiberId').loc[np.arange(2394) + 1].reset_index().sort_values('cobraId')
    sgfm['x'] = np.real(calibModel.centers)
    sgfm['y'] = np.imag(calibModel.centers)

    # setting targetType.
    MOVE_MASK = np.isin(sgfm.cobraId - 1, goodIdx)
    sgfm['targetType'] = TargetType.UNASSIGNED
    sgfm.loc[MOVE_MASK, 'targetType'] = TargetType.HOME
    targetType = sgfm.sort_values('fiberId').targetType.to_numpy()

    # setting position to NaN where no target.
    sgfm.loc[~MOVE_MASK, 'x'] = np.NaN
    sgfm.loc[~MOVE_MASK, 'y'] = np.NaN

    pfiNominal = sgfm.sort_values('fiberId')[['x', 'y']].to_numpy()
    ra = 100 + 1e-3 * pfiNominal[:, 0]
    dec = 100 + 1e-3 * pfiNominal[:, 1]

    pfsDesign = pfsDesignUtils.makePfsDesign(pfiNominal=pfiNominal, ra=ra, dec=dec, targetType=targetType,
                                             arms='brn', designName='cobraHome')

    return pfsDesign


def homeMaskFromDesign(pfsDesign):
    """Return cobra mask where targetType==HOME."""
    return cobraIndexFromDesign(pfsDesign, targetType=TargetType.HOME)


def cobraIndexFromDesign(pfsDesign, targetType):
    """Return cobra mask from a given pfsDesign and targetType."""
    gfm = pd.DataFrame(FiberIds().data)
    sgfm = gfm.set_index('scienceFiberId').loc[np.arange(2394) + 1].reset_index().sort_values('cobraId')

    fiberId = pfsDesign[pfsDesign.targetType == targetType].fiberId
    cobraIds = sgfm[sgfm.fiberId.isin(fiberId)].cobraId.to_numpy()
    return cobraIds - 1
