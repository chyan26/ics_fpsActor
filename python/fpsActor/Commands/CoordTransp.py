#!/usr/bin/env python

import os,sys,re
import math as mt
import numpy as np
import scipy
from scipy import interpolate as ipol


## input x,y point list and zenith angle
def MCStoPFI(xysky, za):

   arg=[mt.atan2(j,i)+mt.pi for i,j in zip(*xysky)]


   print >> sys.stderr , "Scaling"

   scale=ScalingFactor(xysky)

   #deviation
   # base
   #print >> sys.stderr , "Offset 1"
   #offx1,offy1=OffsetBase(xysky)

   # z-dependent
   print >> sys.stderr , "Offset 2"
   offx2,offy2=DeviationZenithAngle(xysky,za)

   xyf3c=[]
   #print zip(scale,arg,offx1,offy1,offx2,offy2)
   #for s,t,ox1,oy1,ox2,oy2 in zip(scale,arg,offx1,offy1,offx2,offy2):
   for s,t,ox2,oy2 in zip(scale,arg,offx2,offy2):
       x=s*mt.cos(t)+ox2
       y=s*mt.sin(t)+oy2
       #x=s*mt.cos(t)+ox1+ox2
       #y=s*mt.sin(t)+oy1+oy2
       #print x,y,x+y
       #xyf3c.append([x,y])
       #xyf3c.append([x,y,s,t,ox1,oy1,ox2,oy2])
       xyf3c.append([x,y,s,t,ox2,oy2])

   #print xyf3c
   return xyf3c

# differential : z
def DeviationZenithAngle(xysky,za):

    coeffz=DiffCoeff(za)/DiffCoeff(30.)

    # x : dx = c0*x*y
    cx0=0.000503107811479
    # y : dy = c0*x^2 + c1*y^2 + c2*y^4 + c3*x^2*y^2 + c4
    cy0=0.00030369690700435151
    cy1=0.00063595572244913589 
    cy2=2.8397096204379468e-06 
    cy3=3.3134621802542193e-06
    cy4=-0.022829824902876293
    # 5.0550506324752235e-05, 0.00025206581695524168, -0.000343496

    # y : slope cy5(z) * y
    za_a=[0.,30.,60.]
    sl_a=[0.,6.4670295752240791e-07,0.00028349223770956881]

    sl_itrp=ipol.splrep(za_a,sl_a,k=2,s=0)
    cy5=ipol.splev(za,sl_itrp)

    offx=[]
    offy=[]
    for x,y in zip(*xysky):
        #print x,y
        dx=coeffz*cx0*x*y
        dy=coeffz*(cy0*x*x+cy1*y*y+cy2*np.power(y,4.)+cy3*x*x*y*y+cy4)+cy5*y
        offx.extend([dx])
        offy.extend([dy])

    return offx,offy

def DiffCoeff(za):

    za*=np.pi/180.
    return 0.995339*(1.741417*np.sin(za)+(1.-np.cos(za)))


## Offset at base
def OffsetBase(xysky):

    # sky-x sky-y off-x off-y
    # data in 2016
    # dfile="data2016/offset_base_2016.dat"
    # data in 2017
    dfile="data_Diff/offset_base.dat"
    fi=open(dfile)
    line=fi.readlines()
    fi.close

    lines=[i.split() for i in line]
    #IpolD=map(list, zip(*lines))
    IpolD=np.swapaxes(np.array(lines,dtype=float),0,1)


    #print IpolD

    #x_itrp=ipol.bisplrep(IpolD[0,:],IpolD[1,:],IpolD[2,:],s=0)
    #y_itrp=ipol.bisplrep(IpolD[0,:],IpolD[1,:],IpolD[3,:],s=0)
    x_itrp=ipol.SmoothBivariateSpline(IpolD[0,:],IpolD[1,:],IpolD[2,:],kx=5,ky=5,s=1)
    y_itrp=ipol.SmoothBivariateSpline(IpolD[0,:],IpolD[1,:],IpolD[3,:],kx=5,ky=5,s=1)

    print >> sys.stderr , "Interpol Done."

    offsetx=[]
    offsety=[]
    for i,j in zip(*xysky):
        offsetx.extend(x_itrp.ev(i,j))
        offsety.extend(y_itrp.ev(i,j))

    return offsetx,offsety

## Scaling Factor: function of r + interpol
def ScalingFactor(xysky):

    dist=[mt.sqrt(i*i+j*j) for i,j in zip(*xysky)]

    # scale1 : rfunction
    scale1=[ScalingFactor_Rfunc(r) for r in dist]

    # scale2 : interpolation
    # Derive Interpolation function
    sc_intf=ScalingFactor_Inter()
    scale2=ipol.splev(dist,sc_intf)

    scale=[ x+y for x,y in zip(scale1, scale2)]

    return scale
    #return scale1

## Scaling Factor: function of r
def ScalingFactor_Rfunc(r):

    # data in 2017
    c0=26.209255623259196 
    c1=0.0077741519133454062 
    c2=2.4652054436469228e-05
    # data in 2016
    # c0=26.206909839797163 
    # c1=0.0077672173203211514 
    # c2=2.4830489935734334e-05

    return c0*r+c1*np.power(r,3.)+c2*np.power(r,5.)

## Scaling Factor: interpolation func.
def ScalingFactor_Inter():

    # data in 2016
    dfile="data2016/scale_interp_2016.dat"
    # data in 2017
    dfile="data_Diff/scale_interp.dat"

    fi=open(dfile)
    line=fi.readlines()
    fi.close

    lines=[i.split() for i in line]
    IpolD=map(list, zip(*lines))

    r_itrp=ipol.splrep(IpolD[0],IpolD[1],s=0)

    return r_itrp
