# from pfs.utils import butler
import fitsio

# temporary shim to access target
#
def _findPfsDesignFile(visit):
    """Load pfsDesign file from """
    import pathlib

    designRoot = pathlib.Path('/data/pfsDesign')
    filename = f'pfsDesign-0x{visit:016x}.fits'

    matched = tuple(designRoot.glob(filename))
    if len(matched) == 0:
        raise KeyError(f'did not find any pfDesign files for visit={visit}, filename={filename}')
    elif len(matched) > 1:
        raise KeyError(f'found too many pfDesign files for visit={visit}, filename={filename}: {matched}')

    return matched[0]

def _loadFullPfsDesign(visit):
    designFile = _findPfsDesignFile(visit)
    designInfo = fitsio.read(designFile, 1)

    return designInfo

def loadPfsDesign(visit):
    """Load and return the pfsDesign content.

    We will use the butler and the datamodel directly, but cannot for tonight.
    """

    SCIENCE = 1

    designInfo = _loadFullPfsDesign(visit)

    cobraDesigns = designInfo[designInfo['targetType'] == SCIENCE]
    targets = cobraDesigns['pfiNominal']

    return targets
