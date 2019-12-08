"""

Example showing how to run various analyis code

"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sigmaclip
import sys
from importlib import reload  #for debugging purposes
from astropy.io import fits
import numpy.ma as ma

import os


import analysisRoutines as ar

#-------------------------------------------------------


# bookkeeping - firs tand last sets of frames, plust the directory with the image files
sourceDir="/Volumes/Vaal/Aug19/Day1/"
tpe=0

fframe=[8450,8500,8550,8573,8623,8677,8727,8777,8843,8893,8943,8993,9043,9093,9148,9198,9248,9298,9348,9398,9448,9498,9548,9598,9648,9698,9748,9798,9848,9898,9948,9998,10048,10098,10148,10198,10248,10298,10348,10398,10448,10498,10548,10598,10648,10698,10748,10798,10848,10898,10948,10998,11048,11098,11148,11198,11248]


lframe=[8499,8549,8570,8622,8672,8726,8776,8836,8892,8942,8992,9042,9092,9142,9168,9247,9297,9347,9397,9447,9497,9547,9597,9647,9697,9747,9797,9847,9897,9947,9997,10047,10097,10147,10197,10247,10297,10347,10397,10447,10497,10547,10597,10647,10697,10747,10797,10847,10897,10947,10997,11047,11097,11147,11197,11247,11297]
fframe=[8450]
lframe=[8499]
#-------------------------------------------------------

tpe=0

#cycle through the list
for parm1,parm2,parm3 in zip(fframe,lframe,fframe):

    #get frameIDs
    frameIDs,centroidFile,files=ar.getFrameIDs(parm1,parm2,parm3,tpe,sourceDir)
    loadPref="NewSet/ndump_"

    #read data from "database"

    fibreID,xArray,yArray,centroidID,xPixArray,yPixArray,fxArray,fyArray,peakArray,xyArray,qualArray,xNomArray,yNomArray,dxArray,dyArray=ar.readFibresSet(frameIDs,loadPref)

    loadPref=loadPref="NewSet/ntrans_"

    xTransArray,yTransArray,xScaleArray,yScaleArray,rotArray,afCoeffArray=ar.readTransSet(frameIDs,loadPref)

    loadPref="NewSet/nphys_"
    phys=ar.readPhysTidy(frameIDs[0],loadPref)

    #calculate seeing parameters
    loadPref="NewSet/ndump_"
    xAv,yAv,fxAv,fyAv,rmsVal,rmsX,rmsY,xd,yd,trans,rot,scaleX,scaleY,fxArray,fyArray=ar.calcSeeingMM(frameIDs,loadPref)

    #and plot them
    rmsRange=[0,0.01]
    fxRange=[1,3]
    fyRange=[1,5]
    stitle=""
    units="mm"

    ar.plotSeeing(frameIDs,xAv,yAv,fxAv,fyAv,fxArray,fyArray,rmsVal,rmsX,rmsY,fxRange,fyRange,rmsRange,units,stitle)

    delFiles=1
    vmi=0
    vma=0.02
    outPref="test"

    loadPref="NewSet/nphys_"
    phys=ar.readPhysTidy(frameIDs[0],loadPref)
    
    loadPref="NewSet/nphys_"
    st=ar.getTimes(frameIDs[0],loadPref)
    
    #ar.makeMovies(frameIDs,st,xArray,yArray,fxArray,fyArray,delFiles,outPref,vmi,vma)

    loadPref="NewSet/"
    
    ar.snapShots(files,frameIDs,loadPref,outPref,phys['inr'],st)


    


