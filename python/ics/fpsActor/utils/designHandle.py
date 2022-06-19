from pfs.datamodel import PfsDesign
from pfs.utils.fiberids import FiberIds
import pathlib
import pandas as pd
from ics.fpsActor.utils import pfsDesign
import numpy as np
import logging

class DesignFileHandle(object):
    def __init__(self, designId=None, maskFile=None, calibModel = None):
        
        # Initializing the logger
        logging.basicConfig(format="%(asctime)s.%(msecs)03d %(levelno)s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
        self.logger = logging.getLogger('DesignFileHandle')
        self.logger.setLevel(logging.INFO)

        self.maskFile = maskFile
        self.targets = None

        self.goodIdx = None
        self.badIdx = None
        self.calibModel = calibModel
           

        if designId is not None:
            self.designId = designId
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
        
        self.targetMoveIdx  = np.array(goodIdx)
        self.targetNotMoveIdx = np.array(badIdx)

        targets = cobra_x+cobra_y*1j
        
        self.targets = targets

    def fillCalibModelCenter(self):
        
        targets = self.targets
        for idx, t in enumerate(targets):
            if np.isnan(t.real):
                targets[idx] = self.centers[idx]+(0.5+0.5j)
        self.targets = targets

    def loadMask(self):

        if self.maskFile is not None:
            cobraInfo = pd.read_csv(self.maskFile)
            notMoveCobra = cobraInfo.loc[cobraInfo['bitMask'] == 0]
            doMoveCobra = cobraInfo.loc[cobraInfo['bitMask'] == 1]

            self.targetMoveIdx = doMoveCobra['cobraId'].values - 1
            self.targetNotMoveIdx = notMoveCobra['cobraId'].values - 1

            #self.goodIdx = self.targetMoveIdx
            #self.badIdx = self.targetMoveIdx

    def loadCalibCobra(self, calibModel):
        
        
        des = calibModel
        cobras = []
        for i in des.findAllCobras():
            c = func.Cobra(des.moduleIds[i],
                        des.positionerIds[i])
            cobras.append(c)
        allCobras = np.array(cobras)
        nCobras = len(allCobras)

        goodNums = [i+1 for i,c in enumerate(allCobras) if
                des.cobraIsGood(c.cobraNum, c.module)]
        badNums = [e for e in range(1, nCobras+1) if e not in goodNums]


        self.goodIdx = np.array(goodNums, dtype='i4') - 1
        self.badIdx = np.array(badNums, dtype='i4') - 1
