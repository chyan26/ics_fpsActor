import pandas as pd
from ics.fpsActor.utils import pfsDesign


class DesignFileHandle(object):
    def __init__(self, designId, maskFile=None):

        self.maskFile = maskFile
        self.designId = designId
        self.targets = None
        self.goodIdx = None
        self.badIdx = None

        if designId is not None:
            self._loadTargets()

    def _loadTargets(self):
        targetPos = pfsDesign.loadPfsDesign(self.designId)
        targets = targetPos[:, 0] + targetPos[:, 1] * 1j

        self.targets = targets

    def loadMask(self):

        if self.maskFile is not None:
            cobraInfo = pd.read_csv(self.maskFile)
            notMoveCobra = cobraInfo.loc[cobraInfo['bitMask'] == 0]
            doMoveCobra = cobraInfo.loc[cobraInfo['bitMask'] == 1]

            self.badIdx = notMoveCobra['cobraId'].values - 1
            self.goodIdx = doMoveCobra['cobraId'].values - 1
