"""

Tools for post run analysis and plots. 

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sigmaclip
import sys
from importlib import reload  #for debugging purposes
from astropy.io import fits
import numpy.ma as ma

import os

#the try/except is for using in the MHS environment, or standalone. 

try:
    import mcsActor.Visualization.mcsRoutines as mcs
except:
    import mcsRoutines as mcs

try:
    import mcsActor.Visualization.fpsRoutines as fps
except:
    import fpsRoutines as fps

try:
    import mcsActor.Visualization.visRoutines as vis
except:
    import visRoutines as vis

try:
    import mcsActor.Visualization.plotRoutines as visplot
except:
    import plotRoutines as visplot

    
def readFibres(frameID,loadPref):

    """

    wrapper function to read matched fibreIDs from a numpy save file. To
    be substituted for a routine that returns the exact same
    parameters from the database.

    This routine is for a single image. 
     
    Input: frameID of data
           loadPref: prefix for the file location (including path)

    Output:  numpy masked arrays containing the following
    
    fibreID: fibreID
    x,y: position in mm, after transformations
    centroidID: original centroid ID
    xPix,yPix: original positions in pixels
    fxArry,fy: FWHM (x and y in image coordinates)
    peak: peak values
    xy: third value in second moment
    qual: quality flag
    xNom, yNom: expected mm coordinates 
    dx, dy: difference between expected/measured coordinates in mm


    """

    #read the numpy array 
    vals=np.load(loadPref+str(frameID)+".npy")

    #assign the values
    fibreID=vals[:,0].ravel()
    x=vals[:,1].ravel()
    y=vals[:,2].ravel()
    centroidID=vals[:,3].ravel()
    xPix=vals[:,4].ravel()
    yPix=vals[:,5].ravel()
    fx=vals[:,6].ravel()
    fy=vals[:,7].ravel()
    peak=vals[:,8].ravel()
    xy=vals[:,9].ravel()
    qual=vals[:,10].ravel()
    xNom=vals[:,11].ravel()
    yNom=vals[:,12].ravel()
    dx=vals[:,13].ravel()
    dy=vals[:,14].ravel()


    fibreID=convertArray(fibreID)
    x=convertArray(xArray)
    y=convertArray(yArray)
    centroidID=convertArray(centroidID)
    xPix=convertArray(xPixArray)
    yPix=convertArray(yPixArray)
    fx=convertArray(fxArray)
    fy=convertArray(fyArray)
    peak=convertArray(peakArray)
    xy=convertArray(xyArray)
    qual=convertArray(qualArray)
    xNom=convertArray(xNomArray)
    yNom=convertArray(yNomArray)
    dx=convertArray(dxArray)
    dy=convertArray(dyArray)
   
    return fibreID,x,y,centroidID,xPix,yPix,fx,fy,peak,xy,qual,xNom,yNom,dx,dy


def readFibresSet(frameIDs,loadPref):

    """
    wrapper function to read a set of matched fibreIDs from a file. to
    be substituted for a routine that returns the exact same
    parameters from the database.

    Input: a list of frameIDs (don't need to be consecutive)
           loadPref: prefix for the file location (including path)

    Output: numpy masked arrays containing the following
    
    fibreID: fibreID
    xArray,yArray: position in mm, after transformations
    centroidID: original centroid ID
    xPixArray,yPixArray: original positions in pixels
    fxArry,fyArray: FWHM (x and y in image coordinates)
    peakArray: peak values
    xyArray: third value in second moment
    qualArray: quality flag
    xNomArray, yNomArray: expected mm coordinates 
    dxArray, dyArray: difference between expected/measured coordinates in mm

    """

    # create variables

    fibreID=[]
    xArray=[]
    yArray=[]
    centroidID=[]
    xPixArray=[]
    yPixArray=[]
    fxArray=[]
    fyArray=[]
    peakArray=[]
    xyArray=[]
    qualArray=[]
    xNomArray=[]
    yNomArray=[]
    dxArray=[]
    dyArray=[]
    

    for frameID in frameIDs:
        #read the values
        vals=np.load(loadPref+str(frameID)+".npy")

        #add to the list
        fibreID.append(vals[:,0])
        xArray.append(vals[:,1])
        yArray.append(vals[:,2])
        centroidID.append(vals[:,3])
        xPixArray.append(vals[:,4])
        yPixArray.append(vals[:,5])
        fxArray.append(vals[:,6])
        fyArray.append(vals[:,7])
        peakArray.append(vals[:,8])
        xyArray.append(vals[:,9])
        qualArray.append(vals[:,10])
        xNomArray.append(vals[:,11])
        yNomArray.append(vals[:,12])
        dxArray.append(vals[:,13])
        dyArray.append(vals[:,14])

     
    fibreID=convertArray(fibreID).T
    xArray=convertArray(xArray).T
    yArray=convertArray(yArray).T
    centroidID=convertArray(centroidID).T
    xPixArray=convertArray(xPixArray).T
    yPixArray=convertArray(yPixArray).T
    fxArray=convertArray(fxArray).T
    fyArray=convertArray(fyArray).T
    peakArray=convertArray(peakArray).T
    xyArray=convertArray(xyArray).T
    qualArray=convertArray(qualArray).T
    xNomArray=convertArray(xNomArray).T
    yNomArray=convertArray(yNomArray).T
    dxArray=convertArray(dxArray).T
    dyArray=convertArray(dyArray).T
   
    return fibreID,xArray,yArray,centroidID,xPixArray,yPixArray,fxArray,fyArray,peakArray,xyArray,qualArray,xNomArray,yNomArray,dxArray,dyArray

def readTrans(frameID,loadPref):

    """

    read transformation coefficients from file.  To be replaced with equivalent
    database routine. 

    This routine is for a single image

    Input: a list of frameIDs (don't need to be consecutive)
           loadPref: prefix for the file location (including path)

    Output: numpy masked arrays containing the following (in mm coordinate system)
    
    xTrans: x translation
    yTrans: y translation
    xScale: x scale
    yScale: y scale
    rot: rotation
    afCoeff: affine transform array



    """


    afCoeff=np.load(loadPref+str(frameID)+".npy")
    afCoeff=np.load(loadPref+str(frameID)+".npy")

    #convert to individual components
    xScale=np.sqrt(afCoeff[0,0]**2+afCoeff[0,1]**2)
    yScale=np.sqrt(afCoeff[1,0]**2+afCoeff[1,1]**2)
    xTrans=afCoeff[0,2]
    yTrans=afCoeff[1,2]

    rot=np.arctan2(afCoeff[1,0]/np.sqrt(afCoeff[0,0]**2+afCoeff[0,1]**2),
                            afCoeff[1,1]/np.sqrt(afCoeff[1,0]**2+afCoeff[1,1]**2))

    return xTrans,yTrans,xScale,yScale,rot,afCoeff
    
def readTransSet(frameIDs,loadPref):

    """

    read a set of transformation data from numpy save files. To be replaced with equivalent
    database routine. 

    Input: a list of frameIDs (don't need to be consecutive)
           loadPref: prefix for the file location (including path)

    Output: numpy masked arrays containing the following (in mm coordinate system)
    
    xTransArray: x translation
    yTransArray: y translation
    xScaleArray: x scale
    yScaleArray: y scale
    rotArray: rotation
    afCoeffArray: affine transform arrays

    """


    #setup variables
    xTransArray=[]
    yTransArray=[]
    rotArray=[]
    xScaleArray=[]
    yScaleArray=[]
    afCoeffArray=[]

    
    for frameID in frameIDs:

        #read transformation array from the file
        afCoeff=np.load(loadPref+str(frameID)+".npy")

        #convert to individual components
        sx=np.sqrt(afCoeff[0,0]**2+afCoeff[0,1]**2)
        sy=np.sqrt(afCoeff[1,0]**2+afCoeff[1,1]**2)
        xd=afCoeff[0,2]
        yd=afCoeff[1,2]

        rotation=np.arctan2(afCoeff[1,0]/np.sqrt(afCoeff[0,0]**2+afCoeff[0,1]**2),
                            afCoeff[1,1]/np.sqrt(afCoeff[1,0]**2+afCoeff[1,1]**2))

        #append to the llist
        xTransArray.append(xd)
        yTransArray.append(yd)
        xScaleArray.append(sx)
        yScaleArray.append(sy)
        rotArray.append(rotation)
        afCoeffArray.append(afCoeff)
        
    #convert to numpy arrays and return
    xTransArray=np.array(xTransArray).T
    yTransArray=np.array(yTransArray).T
    xScaleArray=np.array(xScaleArray).T
    yScaleArray=np.array(yScaleArray).T
    rotArray=np.array(rotArray).T
    afCoeffArray=np.array(afCoeffArray).T
    
    return xTransArray,yTransArray,xScaleArray,yScaleArray,rotArray,afCoeffArray
    
def convertArray(arr):

    """

    numpy save files don't handle masked arrays, so this goes from NaNs to mask. 

    """
    
    arr=np.array(arr)
    arr=ma.masked_invalid(arr)
    return arr
        
def subSmooth(xAv,yAv,parm,order,normal):

    """
    subtract the smooth component of a 2D xy function, choosing between a  linear function,
    or a second order function.  Optionally add back on the mean of the original data set. 

    Input: 
    
    xAv,yAv: coordinates
    parm: paramter to fit
    order: 1 for linear, 2 for quadratic
    normal: 1 to add back mean of ata

    output: 

    parmSub: parm with the smooth component subtracted
    smooth: fit for smooth comput

    """

    #transform the values, selection only non masked values
    X=xAv[~xAv.mask].flatten()
    Y=yAv[~xAv.mask].flatten()
    F=parm[~xAv.mask].flatten()

    #and full xy coordinates
    X1=xAv.flatten()
    Y1=yAv.flatten()

    pAv=F.mean()
    #create the axes for the fit. One version for the unmasked values, one for all. 

    if(order==1):
        A=np.array([X*0+1, X, Y]).T
        A1=np.array([X1*0+1, X1, Y1]).T
    if(order==2):
        A=np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
        A1=np.array([X1*0+1, X1, Y1, X1**2, X1**2*Y1, X1**2*Y1**2, Y1**2, X1*Y1**2, X1*Y1]).T

    #do fit
    coeff,r,rank,s=np.linalg.lstsq(A,F,rcond=None)
    smooth=np.zeros((len(xAv)))

    #create smooth map for all the points
    for i in range(len(coeff)):
        smooth=smooth+coeff[i]*A1.T[i]

    #subtract
    parmSub=parm-smooth

    if(normal==1):
        parmSub=parmSub+pAv
        
    return parmSub,smooth

def readPhysTidy(frameID,loadPref):

    """

    Same as readPhys, but rounds off rotation/elevation values for tidier plots. 

    """


    ff=np.load(loadPref+str(frameID)+".npy")
    print(ff.shape)

    phys={}
    
    phys["t"]=np.float(ff[0])    
    phys["za"]=90-int(5*round(float(ff[1])/5))
    phys["inr"]=int(5 * round(float(ff[2])/5)) 
    phys["adc"]=np.float(ff[3])    
    phys["date"]=ff[4]    
    phys["ut"]=ff[5]    
    phys["hst"]=ff[6] 
    phys["hum"]=np.float(ff[7])    
    phys["tmp"]=np.float(ff[8])    
    phys["prs"]=np.float(ff[9])    
    phys["ohum"]=np.float(ff[10])    
    phys["otmp"]=np.float(ff[11])    
    phys["oprs"]=np.float(ff[12])    
    phys["ownd"]=np.float(ff[13])    
    phys["mcm1t"]=np.float(ff[14])    
    phys["mctopt"]=np.float(ff[15])    
    phys["mccftt"]=np.float(ff[16])    
    phys["mccovt"]=np.float(ff[17])    
    phys["mccint"]=np.float(ff[18])
    phys["mccott"]=np.float(ff[19])    
    phys["mcelet"]=np.float(ff[20])    
    phys["mcflow"]=np.float(ff[21])   

    return phys
       
def readPhys(frameID,loadPref):

    """
    Retrieve telemetry, exposure time, etc. from numpy file (replaced by database equivalent)

    INput: frameID
           loadPref: prefix for file
    """

    ff=np.load(loadPref+str(frameID)+".npy")

    phys={}
    
    phys["t"]=np.float(ff[0])    
    phys["el"]=np.float(90-ff[1])    
    phys["inr"]=np.float(ff[2])    
    phys["adc"]=np.float(ff[3])    
    phys["date"]=ff[4]
    phys["ut"]=ff[5]    
    phys["hst"]=ff[6] 
    phys["hum"]=np.float(ff[7])    
    phys["tmp"]=np.float(ff[8])    
    phys["prs"]=np.float(ff[9])    
    phys["ohum"]=np.float(ff[10])    
    phys["otmp"]=np.float(ff[11])    
    phys["oprs"]=np.float(ff[12])    
    phys["ownd"]=np.float(ff[13])    
    phys["mcm1t"]=np.float(ff[14])    
    phys["mctopt"]=np.float(ff[15])    
    phys["mccftt"]=np.float(ff[16])    
    phys["mccovt"]=np.float(ff[17])    
    phys["mccint"]=np.float(ff[18])
    phys["mccott"]=np.float(ff[19])    
    phys["mcelet"]=np.float(ff[20])    
    phys["mcflow"]=np.float(ff[21])   

    return phys

def getTimes(frameID,loadPref):

    """

    Utility routine to generate a string with the time/rotation/elevation information. To be replaced with 
    database equivalent. 


    """

    phys=readPhysTidy(frameID,loadPref)
    
    t=np.float(phys["t"])
    za=phys["za"]
    inr=phys["inr"]
    
    st="t={:.1f} za={:d} inr={:d}".format(t,za,inr)
    
    return st
    
def calcSeeingMM(frameIDs,loadPref):

    """

    Calculate a set of seeing plots in MM coordinates, from spots that have already been matched and 
    transformed.

    Input:
       frameIDs: list of frame IDs
       loadPref: prefix for loading from numpy files

    Output:
        xAv,yAv: average positions of spots 
        fxAv,fyA: average FWHMs of spots 
        rmsVal,rmsX,rmsY: rms of positions, wtih affine transoformation subtracted. 
        xd,yd,trans,rot,scaleX,scaleY: list transformaiton parameters for each frame


    """

    
    #retrieve values
    fibreID,xArray,yArray,centroidID,xPixArray,yPixArray,fxArray,fyArray,peakArray,xyArray,qualArray,xNomArray,yNomArray,dxArray,dyArray=readFibresSet(frameIDs,loadPref)
    xfirst=xArray[:,0].ravel()
    yfirst=yArray[:,0].ravel()

    #retrieve the transformations used
    trans,xTrans,yTrans,rot,scaleX,scaleY=readTransSet(frameIDs,loadPref)
        
    #set up the variables
    
    dd=np.zeros(xArray.shape)
    xd=np.zeros(xArray.shape)
    yd=np.zeros(xArray.shape)

    #calculate distances compared to first frame
    for i in range(xArray.shape[0]):
        dd[i,:]=np.sqrt((xArray[i,:]-xfirst[i])**2+(yArray[i,:]-yfirst[i])**2)
        xd[i,:]=np.sqrt((xArray[i,:]-xfirst[i])**2)
        yd[i,:]=np.sqrt((yArray[i,:]-yfirst[i])**2)

    #adjust the masks if needed
    dd=ma.masked_where(((dd <=0) | (xArray.mask == True)), dd)
    xd=ma.masked_where(((dd <=0) | (xArray.mask == True)), xd)
    yd=ma.masked_where(((dd <=0) | (xArray.mask == True)), yd)

    #get rms of the values
    rmsVal=dd.std(axis=1)
    rmsX=xd.std(axis=1)
    rmsY=yd.std(axis=1)

    #get the mean values

    xAv,yAv,fxAv,fyAv,peakAv = getAverages(frameIDs,xArray, yArray, fxArray, fyArray, peakArray)

    return xAv,yAv,fxAv,fyAv,rmsVal,rmsX,rmsY,xd,yd,trans,rot,scaleX,scaleY,fxArray,fyArray

def transformPoints(x,y,xd,yd,theta,sx,sy):

    """
    Apply a rigid transformation to the mask (trans, scale, rot). Mostly bookkeeping
    stuff. 

    input:
    x,y: mask positions
    xd,yd: translation
    theta: rotation (radians)
    s: scale

    output: transformed x,y

    """
    
    #create transformation matrix
    matrix=np.zeros((2,3))
    matrix[0,0]=np.cos(theta)*sx
    matrix[0,1]=-np.sin(theta)*sy
    matrix[1,0]=np.sin(theta)*sx
    matrix[1,1]=np.cos(theta)*sy
    matrix[0,2]=xd
    matrix[1,2]=yd

    #bookkeeping for coordinate format
    pts=np.zeros((1,len(x),2))

    pts[0,:,0]=x
    pts[0,:,1]=y

    pts=np.float32(pts)

    #do the transform
    pts1=cv2.transform(pts,matrix)

    #more bookkeeping
    xx=pts1[0,:,0]
    yy=pts1[0,:,1]

    return xx,yy


def calcSeeingPix(frameIDs,loadPref):

    """

    Calculate a set of seeing plots in MM coordinates, from spots that have already been matched and 
    transformed.

    Input:
       frameIDs: list of frame IDs
       loadPref: prefix for loading from numpy files

    Output:
        xAv,yAv: average positions of spots 
        fxAv,fyA: average FWHMs of spots 
        rmsVal,rmsX,rmsY: rms of positions, wtih affine transoformation subtracted. 
        xd,yd,trans,rot,scaleX,scaleY: list transformaiton parameters for each frame


    """
    #retrieve centroids
    centroidFile=loadPref+str(frameID[0]).zfill(6)+"_centroids.dat"
    centroids=np.loadtxt(centroidFile)

    #extract points from first frame
    ind=np.where(centroids[:,0]==frameIDs[0])
    xfirst=centroids[ind,2]
    yfirst=centroids[ind,3]


    #match points to first frame
    xArray,yArray,fxArray,fyArray,backArray,peakArray,qualArray=vis.matchAllPoints(centroids,xfirst,yfirst,tol,frameIDs)

    #calculate transformations
    xTrans,yTrans,xScale,yScale,rot,allTrans = getTransByFrame(xArray,yArray,fxArray,fyArray,peakArray,xfirst,yfirst)

    #calculate and subtract the transform between first frame and points
    xArray1=np.zeros((xArray.shape))
    yArray1=np.zeros((xArray.Sahpe))

    for i in range(nFrames):
        xArray1[:,i], yArray1[:,i] = transformPointsNew(xArray[:,i],yArray[:,i],xdAll[i],ydAll[i],rotAll[i],sxAll[i],syAll[i])

    #some masking
    xArray1=ma.masked_where((xArray1 < 100) | (xArray.mask == True),xArray1)
    yArray1=ma.masked_where((yArray1 < 100) | (xArray.mask == True),yArray1)

        
    #set up the variables
    
    dd=np.zeros(xArray.shape)
    xd=np.zeros(xArray.shape)
    yd=np.zeros(xArray.shape)

    #calculate distances compared to first frame
    for i in range(xArray.shape[0]):
        dd[i,:]=np.sqrt((xArray[i,:]-xfirst[i])**2+(yArray[i,:]-yfirst[i])**2)
        xd[i,:]=np.sqrt((xArray[i,:]-xfirst[i])**2)
        yd[i,:]=np.sqrt((yArray[i,:]-yfirst[i])**2)

    #adjust the masks if needed
    dd=ma.masked_where(((dd <=0) | (xArray.mask == True)), dd)
    xd=ma.masked_where(((dd <=0) | (xArray.mask == True)), xd)
    yd=ma.masked_where(((dd <=0) | (xArray.mask == True)), yd)

    #get rms of the values
    rmsVal=dd.std(axis=1)
    rmsX=xd.std(axis=1)
    rmsY=yd.std(axis=1)

    #get the mean values

    xAv,yAv,fxAv,fyAv,peakAv = getAverages(frameIDs,xArray, yArray, fxArray, fyArray, peakArray)

    return xAv,yAv,fxAv,fyAv,rmsVal,rmsX,rmsY,xd,yd,trans,rot,scaleX,scaleY

def getAverages(frameIDs,xArray, yArray, fxArray, fyArray, peakArray):

    xAv=xArray.mean(axis=1)
    yAv=yArray.mean(axis=1)
    fxAv=fxArray.mean(axis=1)
    fyAv=fyArray.mean(axis=1)
    peakAv=fyArray.mean(axis=1)

    return xAv,yAv,fxAv,fyAv,peakAv

    
def plotSeeing(frameIDs,xAv,yAv,fxAv,fyAv,fxArray,fyArray,rmsVal,rmsX,rmsY,fxRange,fyRange,rmsRange,units,stitle):

    """

    Plot a set of seeing data, with maps of RMS, FWHMs. Basically a wrapper to the plotting routine. 

    Input: 
       frameIDs: lsit of frame IDs
       xAv,yAv,fxAv,fyAv,rmsVal,rmsX,rmsY: output from calcSeeing routine
       fxRange, fyRange, rmsRange: ranges for plots. Can be None
       units: pixels or mm (string)
       stitle: subtitle for plots

    Output:
       three plots, with filenames ######_rms.png ######_fx.png ######_fy.png where the prefix is the # of the first frame

    """

    
    nbins=30
    prefix=str(frameIDs[0])
    inter=0
    stitle=""

    visplot.pairPlot(xAv,yAv,rmsVal,rmsVal.ravel(),rmsRange,"RMS",prefix,"_rms","RMS",units,nbins,inter,stitle=stitle)
    visplot.pairPlot(xAv,yAv,fxAv,fxArray.ravel(),fxRange,"FHWM(x)",prefix,"_fx","FWHM(x)",units,nbins,inter,stitle=stitle)
    visplot.pairPlot(xAv,yAv,fyAv,fyArray.ravel(),fyRange,"FHWM(y)",prefix,"_fy","FWHM(y)",units,nbins,inter,stitle=stitle)
    
def makeMovies(frameIDs,st,xArray,yArray,fxArray,fyArray,delFiles,outPref,vmi,vma):

    """
    
    make a set of movies showing the change in position (after transformations) and the FWHMs
    
    Input: 
       frameIDs list fo frame IDs
       st: string for title of plot
       xArray,yArray,fxArray,fyArray: position and FWHM data
       delFiles: flag = 1 delete intermediate files
       prefix: prefix for output files
       vmi, vma: minimum and maximum for position plot

    Output: 
       m4v movie showing changes in position and FWHMs

    """

    x1=xArray[:,0]
    y1=yArray[:,0]
    
    
    ii=0
    for frame in frameIDs[1:]:
        xd=xArray[:,ii]-xArray[:,ii-1]
        yd=yArray[:,ii]-yArray[:,ii-1]

        dd=np.sqrt(xd**2+yd**2)

        #an estimate of the FWHM size
        szX,smX=subSmooth(x1,y1,fxArray[:,ii],2,1)
        szY,smY=subSmooth(x1,y1,fyArray[:,ii],2,1)

        fig,ax=plt.subplots(2,2,figsize=(14,10))

        scFirst=ax[0,0].scatter(x1,y1,c=dd,vmin=vmi,vmax=vma)
        fig.colorbar(scFirst,ax=ax[0,0])
        
        scFirst=ax[0,1].scatter(x1,y1,c=dd)
        fig.colorbar(scFirst,ax=ax[0,1])
        
        scFWHM=ax[1,0].scatter(x1,y1,c=szX)
        fig.colorbar(scFWHM,ax=ax[1,0])
        
        ax[1,0].set_title("FWHM (x)")
        scFWHM=ax[1,1].scatter(x1,y1,c=szY)
        
        fig.colorbar(scFWHM,ax=ax[1,1])
        ax[1,1].set_title("FWHM (y)")

        plt.suptitle(str(frameIDs[0])+" "+st)
        plt.savefig(outPref+str(frameIDs[0])+"_"+str(int(ii)).zfill(3)+".png")
            
        ii=ii+1

    plt.close('all')
    os.system("ffmpeg -framerate 3 -i "+outPref+str(frameIDs[0])+"_%03d.png "+outPref+str(frameIDs[0])+".m4v")

    if(delFiles==1):
        os.system("rm "+outPref+"*.png")

def getFrameIDs(parm1,parm2,parm3,tpe,sourceDir):

    """
    
    Utility routine to generate a list of frame IDs from the Aug19 run. This will delete
    bad frames (as recorded during the run), and handles both kinds of frame IDs (with or
    without moveIDs.). This is engineering run specific. 

    INput:
       parm1, parm2, parm3: information about first and last frame. For no move IDs this is 
                            first frame ID, last frame ID, 

    """

    frameId1=parm1
    fPref="PFSC"
    dataType='pinhole'
    
    if(frameId1==8777):
        frameSkip=[8805,8806]
    elif(frameId1==11602):
        frameSkip=[11620]
    elif(frameId1==13219):
        frameSkip=[13247]
    elif(frameId1==16831):
        frameSkip=[16853,16852,16851,16850,16849]   
    else:
        frameSkip=[]
        
    files,prefix,centroidFile,frameIDs=vis.getFileNamesAll(parm1,parm2,parm3,frameSkip,sourceDir,fPref,dataType,tpe)
    return frameIDs,centroidFile,files
    
    
def singleFrameMM(frameID,loadPref,st):

    """
    routine to plot disgnostics from a single frame

    """
    
    #load data
    fibreID,x,y,centroidID,xPix,yPix,fx,fy,peak,xy,qual,xNom,yNom,dx,dy=readFibres(frameID,loadPref)

    prefix=string(int(fibreID[0]))
    dd=np.sqrt(dx*dx+dy*dy)

    fig,ax=plt.subplots(2,2)
    ax[0,0].scatter(x,y,c=dd)
    ax[1,1].quiver(x,y,dx,dy)
    ax[1,0].scatter(x,y,fx)
    ax[1,1].scatter(x,y,fy)

    plt.suptitle(st)
    
    plt.savefig(prefix+"_diag.png")
        
def doCheckMM(plist1,plist2,plist3,tpe):

    """

    wrapper function to run multiple sets of plots.

    INput:
      plist1,plist2,plist3: information about run. 

      If there is a single frame ID per rame,
      plist1=list of first frames, plist2=list of last frames. If there is a frameID and moveID,
      plist=list of frameIDs, plist2=first move ID, plist2=last moveID
    
       tpe=1 for single frameIDs, 2 for frameID + moveID
    

    Output: 
       seeing plots in mm units
       movie of parameters for the sets
       snapshots ?? 

    """

    
    delFiles=1
    loadPref="NewSet/ndump_"
    outPref="newset_"
    
    for parm1,parm2,parm3 in zip(plist1,plist2,plist3):

        frameIDs,centroidFile,files=getFrameIDs(parm1,parm2,parm3,tpe,sourceDir)
        st=getTimes(files[0]) 

        xAv,yAv,fxAv,fyAv,rmsVal,rmsX,rmsY,xd,yd,trans,rot,scaleX,scaleY=calcSeeingMM(frameIDs,loadPref)
        plotSeeing(frameIDs,xAv,yAv,fxAv,fyAv,rmsVal,rmsX,rmsY,fxRange,fyRange,rmsRange,units,stitle)
        
        makeMovies(frameIDs,st,loadPref,outPref,vma,delFiles)

def matchAllPoints(centroids,xx,yy,tol,frameIDs):

    """

    takes a set of centroids, matches the positions for each frame to
    the reference points, and returns a set of arrays for each
    variable, in which points that were not detected have been masked.

    input: 
    centroidFile: file with centroid output
    xx,yy: reference positions

    output: 
    xArray,yArray: array of registered positions
    fxArray,fyArray,backArray,peakArray: registered parameters for spots

    """

    #size of output array
    nPoints=len(xx)
    nFiles=len(frameIDs)
    
    xArray=np.zeros((nPoints,nFrames))
    yArray=np.zeros((nPoints,nFrames))
    fxArray=np.zeros((nPoints,nFrames))
    fyArray=np.zeros((nPoints,nFrames))
    peakArray=np.zeros((nPoints,nFrames))
    backArray=np.zeros((nPoints,nFrames))
    qualArray=np.zeros((nPoints,nFrames))

    #load the centroid data
    
    print(str(nfiles)+" frames. Matching ",end="")
    #for each image
    for i in range(nfiles):

        print(str(i+1)+", ",end="")

        #get spot positions, etc from a particular image

        ind=np.where(centroids[:,0]==frameIDs[i])

        #some reshaping to make later routines happy
        ll=centroids[ind,0].shape[1]
        x=centroids[ind,2].reshape(ll)
        y=centroids[ind,3].reshape(ll)
        fx=centroids[ind,4].reshape(ll)
        fy=centroids[ind,5].reshape(ll)
        peak=centroids[ind,6].reshape(ll)
        back=centroids[ind,7].reshape(ll)
        qual=centroids[ind,8].reshape(ll)
 
        #nearest neighbour matching
        for j in range(npoints):

            #get closest point
            dd=np.sqrt((xx[j]-x)**2+(yy[j]-y)**2)
            ind=np.where(dd==dd.min())

            #is it close enough? if so, add to list
            if(dd.min() < tol):
                xArray[j,i]=x[ind]
                yArray[j,i]=y[ind]
                fxArray[j,i]=fx[ind]
                fyArray[j,i]=fy[ind]
                peakArray[j,i]=peak[ind]
                backArray[j,i]=back[ind]
                qualArray[j,i]=qual[ind]
    print()
    #mask unfound values
    
    xArray=ma.masked_where(xArray <= 0 ,xArray)
    yArray=ma.masked_where(xArray <= 0 ,yArray)
    fxArray=ma.masked_where(xArray <= 0 ,fxArray)
    fyArray=ma.masked_where(xArray <= 0 ,fyArray)
    backArray=ma.masked_where(xArray <= 0 ,backArray)
    peakArray=ma.masked_where(xArray <= 0 ,peakArray)
    qualArray=ma.masked_where(xArray <= 0 ,qualArray)
    
    return xArray,yArray,fxArray,fyArray,backArray,peakArray,qualArray

    
def getTransByFrame(xArray,yArray,fxArray,fyArray,peakArray,xm,ym):

    """

    Estimate the affine transformation on a frame by frame basis, compared to reference points.
    Note that points must be matched and sorted. 

    input:
    xArray,yArray: array of psotions
    fxArray,fyArray: array of FWHMs
    backArray,peakArray: array of background/peak values
    xm,ym: reference positions

    output: affine transformation results
    xdAll,ydAll: translations
    sxAll,syAll: scaling
    rotAll: rotation

    """

    #set up variables
    nPoints,nFrames=xArray.shape
    
    xTrans=[]
    xTrans=[]
    yScale=[]
    yScale=[]
    rot=[]

    print("Translating ",end="")

    for i in range(nframes):
        print(i+1,', ',end="")
        #use CV2 library to calculate the affine transformation

        transform,xd,yd,sx,sy,rotation=getTransform(xArray[:,i],yArray[:,i],xm,ym,1)
        xTrans.append(xd)
        yTrans.append(yd)
        xScale.append(sx)
        yScale.append(sy)
        rot.append(rotation)

        allTrans.append(transform)

    print()
    #convert data to numpy arrays
    xTrans=np.array(xTrans)
    yTrans=np.array(yTrans)
    xScale=np.array(sxAll)
    yScale=np.array(syAll)
    rot=np.array(rotAll)
    
    return xdAll,ydAll,sxAll,syAll,rotAll,allTrans


def snapShots(files,frameIDs,loadPref,outPref,rot,st):

    """
    
    Make a set of snapshots showing the PSF shape at various points in the image.

    Input: 
      files: list of files (with path)
      frameIDs: corresponding frame ids
      loadPref: prefix for loading centroid data
      outPref: prefix for output file
      rot: instrument rotation
      st: string for title

    Output:
      series of images with labels

    """

    #retrieve centroids
    centroids=getCentroids(frameIDs[0],frameIDs[-1],loadPref)

    #set of positions scattered across the image, for the four rotations
    #(determined empirically)
    
    if(rot==0):
        xval=[2800,4500,6200,2800,4500,6200,2800,4500,6200]
        yval=[3500,3500,3500,2000,2000,2000,500,500,500]
    elif(rot==90):
        xval=[5200,5200,5200,3600,3600,3600,2000,2000,2000]
        yval=[4500,2800,1200,4500,2800,1200,4500,2800,1200]
    elif(rot==-180):
        xval=[6100,4500,2700,6100,4500,2700,6100,4500,2700]
        yval=[2000,2000,2000,3800,3800,3800,5300,5300,5300]
    elif(rot==-90):
        xval=[3200,3200,3200,5300,5300,5300,6800,6800,6800]
        yval=[1200,2700,4500,1200,2700,4500,1200,2700,4500]

    #cycle through
    for frameID,fname in zip(frameIDs,files):

        #extract positions
        ind=np.where(centroids[:,0]==frameID)
        xx=centroids[ind,2].ravel()
        yy=centroids[ind,3].ravel()

        #load image and get statistics for ranges
        image=vis.getImage(fname)
        rms=image.std()
        mn=image.mean()
        lo=mn
        hi=mn+rms*10

        #cycle thorugh
        fig,axes=plt.subplots(3,3,figsize=[6,6])
        hsize=10
        for ii in range(3):
            for jj in range(3):

                #find the nearest spot to each of the points
                ind=ii*3+jj
                dd=np.sqrt((xval[ind]-xx)**2+(yval[ind]-yy)**2)
                ind1=dd.argmin()
                xv=int(np.round(xx[dd.argmin()].ravel()))
                yv=int(np.round(yy[dd.argmin()].ravel()))

                #plot a region cropped arout the point
                axes[ii,jj].imshow(image[yv-hsize:yv+hsize,xv-hsize:xv+hsize],aspect='equal',vmin=lo,vmax=hi,origin="lower")
                
                #blank ticklabels for readability
                axes[ii,jj].set_xticklabels([])
                axes[ii,jj].set_yticklabels([])
        plt.tight_layout()
                
        plt.suptitle(str(int(frameID))+" ["+st+"]",backgroundcolor="white")
        plt.savefig(outPref+str(ind(frameID))+".png")
       
def getCentroids(frameID1,frameID2,loadPref):
    centroidFile=loadPref+"centroids_"+str(int(frameID1))+"_"+str(int(frameID2))+".dat"
    centroids=np.loadtxt(centroidFile)
    return centroids
    
def getFileNames(frameIDs,sourceDir,tpe):

    files=[]
    for frameID in frameIDs:
        if(tpe==1):
            files.append(sourceDir+"PFSC"+str(int(frameID)).zfill(6)+"00.fits")
        else:
            files.append(sourceDir+"PFSC"+str(int(frameID)).zfill(8)+".fits")

