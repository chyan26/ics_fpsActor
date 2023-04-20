import numpy as np
import pandas as pd
import pfs.utils.pfsDesignUtils as pfsDesignUtils
from pfs.datamodel import TargetType
from pfs.utils.fiberids import FiberIds


def createHomeDesign(calibModel, goodIdx):
    """Create home design from current calibModel, ra and dec are faked."""
    gfm = pd.DataFrame(FiberIds().data)
    sgfm = gfm.set_index('scienceFiberId').loc[np.arange(2394) + 1].reset_index().sort_values('cobraId')
    sgfm['x'] = np.real(calibModel.centers)
    sgfm['y'] = np.imag(calibModel.centers)

    pfiNominal = sgfm.sort_values('fiberId')[['x', 'y']].to_numpy()
    # faking to generate a unique hash.
    ra = 100 + 1e-3 * pfiNominal[:, 0]
    dec = 100 + 1e-3 * pfiNominal[:, 1]

    MOVE_MASK = np.isin(sgfm.cobraId - 1, goodIdx)
    sgfm['targetType'] = TargetType.UNASSIGNED
    sgfm.loc[MOVE_MASK, 'targetType'] = TargetType.HOME
    targetType = sgfm.sort_values('fiberId').targetType.to_numpy()
    pfsDesign = pfsDesignUtils.makePfsDesign(pfiNominal=pfiNominal, ra=ra, dec=dec, targetType=targetType,
                                             arms='brn', designName='cobraHome')
    return pfsDesign
