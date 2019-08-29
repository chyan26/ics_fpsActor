import numpy as np
from astropy.io.fits import getdata
import matplotlib.pylab as plt
from scipy.stats import sigmaclip
import numpy.ma as ma
from scipy import optimize
import sys
import cv2
import itertools
from scipy import optimize
from astropy.io import fits

def getCorners(x,y):
    
    ds,inds=getOrientation(x,y)
    ind=np.argsort(ds)

    x0=x[inds[ind[3]]]
    x1=x[inds[ind[2]]]
    y0=y[inds[ind[3]]]
    y1=y[inds[ind[2]]]

    return x0,x1,y0,y1
    
def getOrientation(xlast,ylast):

    
    xm=xlast.mean()
    ym=ylast.mean()

    #find the four 'corners' by distance from the mean point

    #divide into quadrands
    ind1 = np.where( np.logical_or( xlast-xm > 0, ylast-ym>0) )[0]
    ind2 = np.where( np.logical_or( xlast-xm > 0, ylast-ym<0) )[0]
    ind3 = np.where( np.logical_or( xlast-xm < 0, ylast-ym>0) )[0]
    ind4 = np.where( np.logical_or( xlast-xm < 0, ylast-ym<0) )[0]
    d1=np.sqrt((xlast[ind1]-xm)**2+(ylast[ind1]-ym)**2)
    d2=np.sqrt((xlast[ind2]-xm)**2+(ylast[ind2]-ym)**2)
    d3=np.sqrt((xlast[ind3]-xm)**2+(ylast[ind3]-ym)**2)
    d4=np.sqrt((xlast[ind4]-xm)**2+(ylast[ind4]-ym)**2)

    #distances for each
    d=np.sqrt((xlast-xm)**2+(ylast-ym)**2)
    d1=d.copy()
    d2=d.copy()
    d3=d.copy()
    d4=d.copy()

    #mask irrelevant points
    d1[ind1]=0
    d2[ind2]=0
    d3[ind3]=0
    d4[ind4]=0

    #max distance
    dm1=d1.max()
    dm2=d2.max()
    dm3=d3.max()
    dm4=d4.max()

    #index thereof
    ind1=d1.idxmax()
    ind2=d2.idxmax()
    ind3=d3.idxmax()
    ind4=d4.idxmax()

    #now find the two largest. These will be the good corners
    
    ds=np.array([dm1,dm2,dm3,dm4])
    inds=np.array([ind1,ind2,ind3,ind4])

    ind=np.argsort(ds)

    #and the positions
    x1=xlast[inds[ind[3]]]
    y1=ylast[inds[ind[3]]]
    x2=xlast[inds[ind[2]]]
    y2=ylast[inds[ind[2]]]

    return ds,inds

def calc_R(x,y, xc, yc):
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

def least_squares_circle(x,y):
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
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu
