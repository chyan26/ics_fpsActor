from ics.fpsActor.utils import pfsDesign
import pathlib
import pandas as pd

class DesignFileHandle():
    def __init__(self, designId, maskFile = None):
     
        self.maskFile = None
        if maskFile is not None:
            self.maskFile = maskFile
        self.designId = designId
        self.targets = None
        self.goodIdx = None
        self.badIdx = None
        
        self._loadTargets()
        
    def _loadTargets(self):
        targetPos = pfsDesign.loadPfsDesign(self.designId)
        targets = targetPos[:,0]+targetPos[:,1]*1j
        
        self.targets = targets
 
    
    def loadMask(self):
        
        if self.maskFile is not None:  
            cobraInfo = pd.read_csv(self.maskFile)
            notMoveCobra = cobraInfo.loc[cobraInfo['bitMask'] == 0]
            doMoveCobra = cobraInfo.loc[cobraInfo['bitMask'] == 1]

            self.badIdx = notMoveCobra['cobraId'].values-1
            self.goodIdx = doMoveCobra['cobraId'].values-1

