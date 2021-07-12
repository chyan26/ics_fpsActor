import numpy as np

import fitsio
import sep
import matplotlib.pyplot as plt
from ics.cobraCharmer import pfiDesign
import pathlib


def getObjects(im, sigma=5.0):
    data = im.astype('f4')
    bkg = sep.Background(data)
    data_sub = data - bkg

    mn = np.mean(data_sub)
    std = np.std(data_sub)
    thresh = sigma * std
    objects = sep.extract(data_sub, thresh=thresh)

    return objects, data_sub, bkg


def spots(name, sigma=5.0, doTrim=True, disp=None):
    im = fitsio.read(name)
    objects, imSub, _ = getObjects(im, sigma=sigma)

    if disp is not None:
        disp.set('frame clear')
        disp.set_np2arr(imSub)
        disp.set('regions color green')
        for o in objects:
            disp.set(f"regions command {{point {o['x']} {o['y']}}}")

    if doTrim:
        # CIT Only -- wrap this, CPL.
        w = (objects['y'] < (objects['x'] + 500)) & (objects['y'] > (objects['x'] - 500))
        objects = objects[w]

        if disp is not None:
            disp.set('regions color red')
            for o in objects:
                disp.set(f"regions command {{circle {o['x']} {o['y']} 10}}")

    return objects, imSub


def visCobraSpots(runDir, xml, arm='phi'):
    path = f'{runDir}/data/'

    if arm is 'phi':
        centers = np.load(path + 'phiCenter.npy')
        radius = np.load(path + 'phiRadius.npy')
        fw = np.load(path + 'phiFW.npy')
        rv = np.load(path + 'phiRV.npy')
        af = np.load(path + 'phiAngFW.npy')
        ar = np.load(path + 'phiAngRV.npy')
        sf = np.load(path + 'phiSpeedFW.npy')
        sr = np.load(path + 'phiSpeedRV.npy')
        mf = np.load(path + 'phiMMFW.npy')
        mr = np.load(path + 'phiMMRV.npy')
        badRange = np.load(path + 'badRange.npy')
        badmm = np.load(path + 'badMotorMap.npy')

    model = pfiDesign.PFIDesign(pathlib.Path(xml))
    model.fixModuleIds()

    from ics.cobraCharmer import func
    cobras = []
    for i in model.findAllCobras():
        c = func.Cobra(model.moduleIds[i],
                       model.positionerIds[i])
        cobras.append(c)

    allCobras = np.array(cobras)
    nCobras = len(allCobras)

    goodNums = [i+1 for i, c in enumerate(allCobras) if
                model.cobraIsGood(c.cobraNum, c.module)]
    badNums = [e for e in range(1, nCobras+1) if e not in goodNums]

    goodIdx = np.array(goodNums, dtype='i4') - 1
    badIdx = np.array(badNums, dtype='i4') - 1

    cobra = goodIdx

    plt.figure(2)
    plt.clf()

    plt.subplot()
    ax = plt.gca()
    ax.axis('equal')
    for idx in cobra:
        c = plt.Circle((centers[idx].real, centers[idx].imag),
                       radius[idx], color='g', fill=False)
        ax.add_artist(c)

    for n in range(1):
        for k in cobra:
            if k % 3 == 0:
                c = 'r'
                d = 'c'
            elif k % 3 == 1:
                c = 'g'
                d = 'm'
            else:
                c = 'b'
                d = 'y'
            ax.plot(fw[k][n, 0].real, fw[k][n, 0].imag, c + 'o')
            ax.plot(rv[k][n, 0].real, rv[k][n, 0].imag, d + 's')
            ax.plot(fw[k][n, 1:].real, fw[k][n, 1:].imag, c + '.')
            ax.plot(rv[k][n, 1:].real, rv[k][n, 1:].imag, d + '.')
            ax.plot(centers[k].real, centers[k].imag, 'ro')
            #ax.text(centers[k].real, centers[k].imag,f'{k}',)
            #ax.plot(centers[k].real, centers[k].imag, 'ro')

    #ax.plot(phiData[382,15,:,1], phiData[382,15,:,2], 'o', color='red')
    plt.show()
