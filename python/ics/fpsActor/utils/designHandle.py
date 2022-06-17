from pfs.datamodel import PfsDesign
from pfs.utils.fiberids import FiberIds
import pathlib
import pandas as pd
from ics.fpsActor.utils import pfsDesign
import numpy as np

class DesignFileHandle(object):
    def __init__(self, designId, maskFile=None, calibModel = None):

        self.maskFile = maskFile
        self.designId = designId
        self.targets = None
        self.goodIdx = None
        self.badIdx = None
        self.calibModel = calibModel
        
        if self.designId is not None:
            self._loadTargets()

        if self.calibModel is not None:
            self.centers = self.calibModel.centers


    def _loadTargets(self):
        target = PfsDesign.read(pfsDesignId=self.designId, dirName='/data/pfsDesign')
        gfm = FiberIds()
        
        fiberId = target.fiberId
        cobraId = gfm.fiberIdToCobraId(fiberId)
        
        xx = target.pfiNominal[:,0]
        yy = target.pfiNominal[:,1]
        
        cobra_x = []
        cobra_y = []

        goodIdx = []
        badIdx = []

        bitMask = np.zeros(2394)+1
        for i in range(2394):
            ind = np.where(cobraId == i+1) 
            cobra_x.append(xx[ind[0][0]])
            cobra_y.append(yy[ind[0][0]])
            if np.isnan(xx[ind[0][0]]):
                bitMask[i] = 0
                badIdx.append(i)
            else:
                goodIdx.append(i)
                
        cobra_x = np.array(cobra_x)
        cobra_y = np.array(cobra_y)
        
        goodIdx = np.array(goodIdx)
        badIdx = np.array(badIdx)


        self.goodIdx = goodIdx
        self.badIdx = badIdx
        targets = cobra_x+cobra_y*1j
        
        self.targets = targets

    def fillCalibModelCenter(self):
        
        targets = self.targets
        for idx, t in enumerate(targets):
            if np.isnan(t.real):
                targets[idx] = self.centers[idx]
        self.targets = targets

    def loadMask(self):

        if self.maskFile is not None:
            cobraInfo = pd.read_csv(self.maskFile)
            notMoveCobra = cobraInfo.loc[cobraInfo['bitMask'] == 0]
            doMoveCobra = cobraInfo.loc[cobraInfo['bitMask'] == 1]

            self.badIdx = notMoveCobra['cobraId'].values - 1
            self.goodIdx = doMoveCobra['cobraId'].values - 1
