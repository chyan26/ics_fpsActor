import numpy as np
import astropy.io.fits as pyfits

#from scipy.stats import sigmaclip
import numpy.ma as ma
from scipy import optimize
import math
import cv2
import pandas as pd
from ics.cobraCharmer.utils import butler
import cmath

def getCorners(x, y):

    ds, inds = getOrientation(x, y)
    ind = np.argsort(ds)

    x0 = x[inds[ind[3]]]
    x1 = x[inds[ind[2]]]
    y0 = y[inds[ind[3]]]
    y1 = y[inds[ind[2]]]

    return x0, x1, y0, y1


def getOrientation(xlast, ylast):

    xm = xlast.mean()
    ym = ylast.mean()

    # find the four 'corners' by distance from the mean point

    # divide into quadrands
    ind1 = np.where(np.logical_or(xlast-xm > 0, ylast-ym > 0))[0]
    ind2 = np.where(np.logical_or(xlast-xm > 0, ylast-ym < 0))[0]
    ind3 = np.where(np.logical_or(xlast-xm < 0, ylast-ym > 0))[0]
    ind4 = np.where(np.logical_or(xlast-xm < 0, ylast-ym < 0))[0]
    d1 = np.sqrt((xlast[ind1]-xm)**2+(ylast[ind1]-ym)**2)
    d2 = np.sqrt((xlast[ind2]-xm)**2+(ylast[ind2]-ym)**2)
    d3 = np.sqrt((xlast[ind3]-xm)**2+(ylast[ind3]-ym)**2)
    d4 = np.sqrt((xlast[ind4]-xm)**2+(ylast[ind4]-ym)**2)

    # distances for each
    d = np.sqrt((xlast-xm)**2+(ylast-ym)**2)
    d1 = d.copy()
    d2 = d.copy()
    d3 = d.copy()
    d4 = d.copy()

    # mask irrelevant points
    d1[ind1] = 0
    d2[ind2] = 0
    d3[ind3] = 0
    d4[ind4] = 0

    # max distance
    dm1 = d1.max()
    dm2 = d2.max()
    dm3 = d3.max()
    dm4 = d4.max()

    # index thereof
    ind1 = d1.idxmax()
    ind2 = d2.idxmax()
    ind3 = d3.idxmax()
    ind4 = d4.idxmax()

    # now find the two largest. These will be the good corners

    ds = np.array([dm1, dm2, dm3, dm4])
    inds = np.array([ind1, ind2, ind3, ind4])

    ind = np.argsort(ds)

    # and the positions
    x1 = xlast[inds[ind[3]]]
    y1 = ylast[inds[ind[3]]]
    x2 = xlast[inds[ind[2]]]
    y2 = ylast[inds[ind[2]]]

    return ds, inds


def calc_R(x, y, xc, yc):
    """
    calculate the distance of each 2D points from the center (xc, yc)
    """
    return np.sqrt((x-xc)**2 + (y-yc)**2)


def f(c, x, y):
    """
    calculate the algebraic distance between the data points
    and the mean circle centered at c=(xc, yc)
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def least_squares_circle(x, y):
    """
    Circle fit using least-squares solver.
    Inputs:
        - coords, list or numpy array with len>2 of the form:
        [
    [x_coord, y_coord],
    ...,
    [x_coord, y_coord]
    ]
    Outputs:
        - xc : x-coordinate of solution center (float)
        - yc : y-coordinate of solution center (float)
        - R : Radius of solution (float)
        - residu : MSE of solution against training data (float)
    """
    # coordinates of the barycenter

    #x = np.array([x[0] for x in coords])
    #y = np.array([x[1] for x in coords])
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R)**2)
    return xc, yc, R, residu


def projectPixeltoFC(coord, scale, rotation, fieldCenter):

    xx = (coord[0]-fieldCenter[0])/scale
    yy = (coord[1]-fieldCenter[1])/scale
    rx, ry = rotatePoint2([xx, yy], [fieldCenter[0]/scale, fieldCenter[1]/scale], rotation)

    return rx, ry


def projectFCtoPixel(coord, scale, rotation, fieldCenter):

    xx = (coord[0]*scale)+fieldCenter[0]
    yy = (coord[1]*scale)+fieldCenter[1]
    rx, ry = rotatePoint2([xx, yy], [fieldCenter[0], fieldCenter[1]], rotation)

    return rx, ry


def rotatePoint2(coord, ori, angle):
    """Only rotate a point around the origin (0, 0)."""
    radians = np.deg2rad(angle)
    x = coord[0] - ori[0]
    y = coord[1] - ori[1]
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx+ori[0], yy+ori[1]

def alignDotOnImage(runDir, arm=None):
    
    if arm is None:
        arm = 'phi'

    fits= f'{runDir}/data/{arm}ForwardStack0.fits'
    f =pyfits.open(fits)
    data=f[1].data


    ffDotDF=pd.read_csv(butler.configPathForFFDot())
    dotDF = pd.read_csv(butler.configPathForDot(version='mcs'))
    import sep
    std = np.std(data)

    objects = sep.extract(data.astype('float'),
                                thresh=1000,
                                filter_type='conv', clean=False,
                                deblend_cont=1.0)
    obj=pd.DataFrame(objects)

    target=np.array([ffDotDF['x_pixel'].values,ffDotDF['y_pixel'].values]).T
    source=np.array([obj['x'].values,obj['y'].values]).T.reshape((len(obj['x'].values), 2))

    ff_mcs=pointMatch(target, source,scale=0.5)

    afCoeff,inlier=cv2.estimateAffinePartial2D(target, ff_mcs)
    afCor=cv2.transform(np.array([np.array((dotDF['x_dot'].values,dotDF['y_dot'].values)).T]),afCoeff)
    newDotPos=afCor[0]
    newDot = newDotPos[:,0]+newDotPos[:,1]*1j
    newFFpos = ff_mcs[:,0]+ff_mcs[:,1]*1j

    return newDot, dotDF['r_dot'].values, newFFpos

def checkPhiOpenAngle(centers, radius, fw, dotpos, dotradii, angleList, verbose=False):
    
    L1radii=[]
    blockIdx = []
    for cobraIdx in range(len(centers)):
        angle = angleList[cobraIdx]
        
        # first Calculate the point 60 degree away from HS
        angleAtOpen = (np.deg2rad(angle) + np.angle(fw[cobraIdx,0,0]-centers[cobraIdx]))
        pointAtOpen = cmath.rect(radius[cobraIdx],angleAtOpen)+centers[cobraIdx] 
        L1temp =  np.abs(pointAtOpen - fw[cobraIdx,0,0])
        distDot = np.abs(dotpos[cobraIdx] - fw[cobraIdx,0,0])
        L1radii.append(L1temp)
        diff = np.abs(L1temp - distDot)

        if diff < dotradii[cobraIdx]:
            if verbose:
                print(f'{cobraIdx} affected by dot. L1temp={L1temp} DotDist={distDot}')
            blockIdx.append(cobraIdx)
            
    L1radii=np.array(L1radii)
    
    return L1radii, blockIdx


def pointMatch(target, source, scale=None):
    """
        target: the origin position 
        source: detected source to be searched for matching, 
                in the form of (x0, y0), (x1, y1) ....... 

    """
    # Looking for proper distance
    dist_all = []
    for i in range(len(target)):
        d = np.sqrt(np.sum((target[i]-source)**2, axis=1))
        if ~np.isnan(np.min(d)):
            dist_all.append(np.min(d))

        # print(np.min(d))
    dist_all = np.array(dist_all)
    if scale is None:
        scale = 0.5
    dist = np.median(dist_all)+scale*np.std(dist_all)



    matched = []
    for i in range(len(target)):
        d = np.sqrt(np.sum((target[i]-source)**2, axis=1))

        idx = np.where(d < dist)
        nmatched = len(idx[0])
        if nmatched == 1:
            matched.append(source[idx[0]][0])

        elif nmatched > 1:

            newIdx = np.where(d == np.min(d))

            matched.append(source[newIdx[0][0]])
        else:
            matched.append(np.array([np.nan, np.nan]))

    matched = np.array(matched)

    return matched


def getAffineFromFF(ff_mcs, ff_f3c):

    imgarr = []
    objarr = []

    # Building Affine Transformation
    for i in range(len(ff_mcs)):

        if ~np.isnan(ff_mcs[i].real):

            imgarr.append([ff_f3c[i].real, ff_f3c[i].imag])
            objarr.append([ff_mcs[i].real, ff_mcs[i].imag])
    imgarr = np.array([imgarr])
    objarr = np.array([objarr])

    afCoeff, inlier = cv2.estimateAffinePartial2D(imgarr, objarr)
    mat = {}
    mat['affineCoeff'] = afCoeff
    mat['xTrans'] = afCoeff[0, 2]
    mat['yTrans'] = afCoeff[1, 2]
    mat['xScale'] = np.sqrt(afCoeff[0, 0]**2+afCoeff[0, 1]**2)
    mat['yScale'] = np.sqrt(afCoeff[1, 0]**2+afCoeff[1, 1]**2)
    mat['angle'] = np.arctan2(afCoeff[1, 0]/np.sqrt(afCoeff[0, 0]**2+afCoeff[0, 1]**2),
                              afCoeff[1, 1]/np.sqrt(afCoeff[1, 0]**2+afCoeff[1, 1]**2))

    return mat


def buildCameraModelFF(ff_mcs, ff_f3c):

    # Give the image size in (width, height)
    imageSize = (10000, 7096)

    # preparing two arrays for opencv operation.
    objarr = []
    imgarr = []

    # Re-arrange the array for CV2 convention
    for i in range(len(ff_mcs)):

        if ~np.isnan(ff_mcs[i].real):

            imgarr.append([ff_mcs[i].real, ff_mcs[i].imag])
            objarr.append([ff_f3c[i].real, ff_f3c[i].imag, 0])

    objarr = np.array([objarr])
    imgarr = np.array([imgarr])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objarr.astype(np.float32),
                                                       imgarr.astype(np.float32), imageSize, None, None)

    f3c_mcs_camModel = {'camMatrix': mtx, 'camDistor': dist, 'camRotVec': rvecs, 'camTranVec': tvecs}

    objarr = []
    imgarr = []
    # Re-arrange the array for CV2 convention
    for i in range(len(ff_mcs)):

        if ~np.isnan(ff_mcs[i].real):

            imgarr.append([ff_f3c[i].real, ff_f3c[i].imag])
            objarr.append([ff_mcs[i].real, ff_mcs[i].imag, 0])

    objarr = np.array([objarr])
    imgarr = np.array([imgarr])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objarr.astype(np.float32),
                                                       imgarr.astype(np.float32), imageSize, None, None)

    mcs_f3c_camModel = {'camMatrix': mtx, 'camDistor': dist, 'camRotVec': rvecs, 'camTranVec': tvecs}

    return f3c_mcs_camModel, mcs_f3c_camModel


def mapF3CtoMCS(ff_mcs, ff_f3c, cobra_f3c):
    # Give the image size in (width, height)
    imageSize = (10000, 7096)

    # preparing two arrays for opencv operation.
    objarr = []
    imgarr = []

    # Re-arrange the array for CV2 convention
    for i in range(len(ff_mcs)):

        if ~np.isnan(ff_mcs[i].real):

            imgarr.append([ff_mcs[i].real, ff_mcs[i].imag])
            objarr.append([ff_f3c[i].real, ff_f3c[i].imag, 0])

    objarr = np.array([objarr])
    imgarr = np.array([imgarr])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objarr.astype(np.float32),
                                                       imgarr.astype(np.float32), imageSize, None, None)

    camProperty = {'camMatrix': mtx, 'camDistor': dist, 'camRotVec': rvecs, 'camTranVec': tvecs}

    tot_error = 0

    for i in range(len(objarr)):
        imgpoints2, _ = cv2.projectPoints(objarr[i].astype(np.float32), rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgarr[i].astype(np.float32), imgpoints2[:, 0, :],
                         cv2.NORM_L2)/len(imgpoints2[:, 0, :])
        tot_error = tot_error+error

    print("total error: ", tot_error/len(objarr))

    cobra_obj = np.array([cobra_f3c.real, cobra_f3c.imag, np.zeros(len(cobra_f3c))]).T

    imgpoints2, _ = cv2.projectPoints(cobra_obj.astype(np.float32),
                                      rvecs[0], tvecs[0], mtx, dist)

    imgarr2 = imgpoints2[:, 0, :]

    output = imgarr2[:, 0]+imgarr2[:, 1]*1j
    err = tot_error/len(objarr)

    return output, camProperty, err


def mapMCStoF3C(ff_mcs, ff_f3c, cobra_mcs):
    # Give the image size in (width, height)
    imageSize = (10000, 7096)

    # preparing two arrays for opencv operation.
    objarr = []
    imgarr = []

    # Re-arrange the array for CV2 convention
    for i in range(len(ff_mcs)):

        if ~np.isnan(ff_mcs[i].real):

            imgarr.append([ff_f3c[i].real, ff_f3c[i].imag])
            objarr.append([ff_mcs[i].real, ff_mcs[i].imag, 0])

    objarr = np.array([objarr])
    imgarr = np.array([imgarr])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objarr.astype(np.float32),
                                                       imgarr.astype(np.float32), imageSize, None, None)

    camProperty = {'camMatrix': mtx, 'camDistor': dist, 'camRotVec': rvecs, 'camTranVec': tvecs}

    tot_error = 0

    for i in range(len(objarr)):
        imgpoints2, _ = cv2.projectPoints(objarr[i].astype(np.float32), rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgarr[i].astype(np.float32), imgpoints2[:, 0, :],
                         cv2.NORM_L2)/len(imgpoints2[:, 0, :])
        tot_error = tot_error+error

    print("total error: ", tot_error/len(objarr))

    cobra_obj = np.array([cobra_mcs.real, cobra_mcs.imag, np.zeros(len(cobra_mcs))]).T

    imgpoints2, _ = cv2.projectPoints(cobra_obj.astype(np.float32),
                                      rvecs[0], tvecs[0], mtx, dist)

    imgarr2 = imgpoints2[:, 0, :]
    output = imgarr2[:, 0]+imgarr2[:, 1]*1j
    err = tot_error/len(objarr)

    return output, camProperty, err
