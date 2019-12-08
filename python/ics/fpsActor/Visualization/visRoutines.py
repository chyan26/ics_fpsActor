
import numpy as np
import numpy.ma as ma
from scipy.stats import sigmaclip
from astropy.io.fits import getdata
import cv2

try:
    import mcsActor.windowedCentroid.centroid as centroid
except:
    import centroid as centroid


def getFileName(frameID,moveID,sourceDir,fPref,dataType):

    """

    get a single file name

    """
    
    
    if(dataType=='taichung'):
        fname=sourceDir+"/"+fPref+str(frameID).zfill(4)+".fits"
        prefix=fPref

    elif(dataType=='pinhole'):
        fname=sourceDir+"/"+fPref+str(frameID).zfill(6)+str(moveID).zfill(2)+".fits"
        prefix="check_"+str(frameID).zfill(5)
        
    return fname,prefix

def getFileNamesAll(parm1,parm2,parm3,frameSkip,sourceDir,fPref,dataType,tpe):

    print(parm1,parm2,frameSkip)
    if(tpe==0):
        frameId1=parm1
        frameId2=parm2
        files,prefix,centroidFile,frameIDs=getFileNames(frameId1,frameId2,frameSkip,sourceDir,fPref,dataType)

    elif(tpe==1):
        frameId1=parm1
        moveId1=parm2
        moveId2=parm3
        files,prefix,centroidFile,frameIDs=getFileNamesVis(frameId1,moveId1,moveId2,frameSkip,sourceDir,fPref,dataType)

    return files,prefix,centroidFile,frameIDs
def getFileNamesAllUC(parm1,parm2,parm3,frameSkip,sourceDir,fPref,dataType,tpe):

    if(tpe==0):
        frameId1=parm1
        frameId2=parm2
        files,prefix,centroidFile,frameIDs=getFileNamesUC(frameId1,frameId2,frameSkip,sourceDir,fPref,dataType)

    elif(tpe==1):
        frameId1=parm1
        moveId1=parm2
        moveId2=parm3
        files,prefix,centroidFile,frameIDs=getFileNamesVisUC(frameId1,moveId1,moveId2,frameSkip,sourceDir,fPref,dataType)

    return files,prefix,centroidFile,frameIDs

def getFileNamesVisUC(frameId1,moveId1,moveId2,frameSkip,sourceDir,fPref,dataType):

    frameIDs=frameId1*100+np.arange(moveId1,moveId2+1)
    files=[]

    #different input data, assemle the file names

    for i in frameIDs:
        files.append(sourceDir+"/"+fPref+str(i).zfill(8)+"_uc.fits")

        prefix="see_"+str(frameId1).zfill(6)+"_"+str(moveId1).zfill(2)+"_"+str(moveId1).zfill(2)
        
    centroidFile=prefix+"_centroidsSEX.dat"

    return files,prefix,centroidFile,frameIDs
       

def getFileNamesVis(frameId1,moveId1,moveId2,frameSkip,sourceDir,fPref,dataType):

    frameIDs=frameId1*100+np.arange(moveId1,moveId2+1)
    files=[]

    #different input data, assemle the file names

    for i in frameIDs:
        files.append(sourceDir+"/"+fPref+str(i).zfill(8)+".fits")

        prefix="see_"+str(frameId1).zfill(6)+"_"+str(moveId1).zfill(2)+"_"+str(moveId1).zfill(2)
        
    centroidFile=prefix+"_centroids.dat"

    return files,prefix,centroidFile,frameIDs

def getFileNames(frameId1,frameId2,frameSkip,sourceDir,fPref,dataType):

    """

    Generate a list of file names with complete path.

    the dataType flag toggles between different input sources
    (Taichung lab vs pinhole mask), and can be added to as needed.
    This will need to be updated when there are exposure + move ids.
    

    Input
       frameId1, frameId2 - first and last frames (inclusive)
       frameSkip - list of frames to be excluded
       sourceDir - full path of source directory
       fPref - file prefix (PFSC for MCS)
       dataType - 'pinhole' - commissioning run format (MCS, move id = 0)
                - 'taichung' - various lab data from testing
       
    Returns
       files = list of fileanmes (full path)
       prefix - prefix for output plot file names
       centroidFile - filename for saving centroids in local operation
       frameIDs - list of frameIds
        

    """
    
    #generate a list of frame IDs, delete any skipped ones

    frameIDs=list(np.arange(frameId1,frameId2+1))
    for ids in frameSkip:
        frameIDs.remove(ids)

    files=[]

    #different input data, assemle the file names

    if(dataType=='taichung'):
        for i in frameIDs:
            files.append(sourceDir+"/"+fPref+str(i).zfill(4)+".fits")

        prefix=fPref
        centroidFile=prefix+"_ncentroids.dat"

    elif(dataType=='pinhole'):
        for i in frameIDs:
            files.append(sourceDir+"/"+fPref+str(i).zfill(6)+"00.fits")

        prefix="see_"+str(frameId1).zfill(5)+"_"+str(frameId2).zfill(5)
        
    centroidFile=prefix+"_centroids.dat"

    return files,prefix,centroidFile,frameIDs


def getFileNamesUC(frameId1,frameId2,frameSkip,sourceDir,fPref,dataType):

    """

    Generate a list of file names with complete path.

    the dataType flag toggles between different input sources
    (Taichung lab vs pinhole mask), and can be added to as needed.
    This will need to be updated when there are exposure + move ids.
    

    Input
       frameId1, frameId2 - first and last frames (inclusive)
       frameSkip - list of frames to be excluded
       sourceDir - full path of source directory
       fPref - file prefix (PFSC for MCS)
       dataType - 'pinhole' - commissioning run format (MCS, move id = 0)
                - 'taichung' - various lab data from testing
       
    Returns
       files = list of fileanmes (full path)
       prefix - prefix for output plot file names
       centroidFile - filename for saving centroids in local operation
       frameIDs - list of frameIds
        

    """
    
    #generate a list of frame IDs, delete any skipped ones
    
    frameIDs=list(np.arange(frameId1,frameId2+1))
    for ids in frameSkip:
        frameIDs.remove(ids)

    files=[]

    #different input data, assemle the file names

    if(dataType=='taichung'):
        for i in frameIDs:
            files.append(sourceDir+"/"+fPref+str(i).zfill(4)+"_uc.fits")

        prefix=fPref
        centroidFile=prefix+"_ncentroids.dat"

    elif(dataType=='pinhole'):
        for i in frameIDs:
            files.append(sourceDir+"/"+fPref+str(i).zfill(6)+"00.fits")

        prefix="see_"+str(frameId1).zfill(5)+"_"+str(frameId2).zfill(5)
        
    centroidFile=prefix+"_centroidsSEX.dat"

    return files,prefix,centroidFile,frameIDs

     
def loadInstParams(config):

    """

    load instrument parameters. Dummy function, update!

    """

    
    if(config=='oct18'):
        rotCent=[4691.5,3095.7]
        offset=[0,-85]
    if(config=='aug19'):
        rotCent=[4470, 2873]
        offset=[0,-85]
        
    return rotCent,offset

def getImage(filename):

    """

    Simple wrapper to read an image from file

    """
    
    image=getdata(filename)

    return image
    
def toFits(filename):

    """

    Quick routine to convert raw image to fits.

    """
    
    a=np.fromfile(filename,dtype='uint16')
    image=a.reshape([5778,8960])

    pf.writeto(filename+".fits",image)
    print(image.min(),image.mean(),image.max())


def matchAllPoints(centroids,xx,yy,tol,frameIDs):

    """

    takes a batch of centroids created by getAllCentroids, registers
    the positions for each frame to the mask, and returns a set of arrays for
    each variable, in which points that were not detected have been masked. 

    xx, yy are the mask hole positions, and should be transformed to match the
    image

    input: 
    centroidFile: file with centroid output
    xx,yy: transformed mask coordinates

    output: 
    xArray,yArray: array of registered positions
    fxArray,fyArray,backArray,peakArray: registered parameters for spots

    """

    #create arrays
    #centroids=np.loadtxt(centroidFile)

    #size of output array
    npoints=len(xx)
    nfiles=len(frameIDs)
    
    xArray=np.zeros((npoints,nfiles))
    yArray=np.zeros((npoints,nfiles))
    fxArray=np.zeros((npoints,nfiles))
    fyArray=np.zeros((npoints,nfiles))
    peakArray=np.zeros((npoints,nfiles))
    backArray=np.zeros((npoints,nfiles))
    qualArray=np.zeros((npoints,nfiles))

    #load the centroid data
    
    print(str(nfiles)+" frames. Matching ",end="")
    #for each image
    for i in range(nfiles):

        print(str(i+1)+", ",end="")

        #get spot positions, etc from a particular image

        ind=np.where(centroids[:,0]==frameIDs[i])

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

            dd=np.sqrt((xx[j]-x)**2+(yy[j]-y)**2)
            ind=np.where(dd==dd.min())


            #need to filter in case points are missing
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

    Estimate the affine transformation on a frame by frame basis, compared to reference points

    input:
    xArray,yArray: array of psotions
    fxArray,fyArray: array of FWHMs
    backArray,peakArray: array of background/peak values
    xm,ym: mask coordinates

    output: affine transformation results
    xdAll,ydAll: translations
    sxAll,syAll: scaling
    rotAll: rotation
    fxFrameAv,fyFrameAv,peakFrameAv: average in spot characteristics

    """

    #set up variables
    npoints,nframes=xArray.shape
    xdAll=[]
    ydAll=[]
    sxAll=[]
    syAll=[]
    fxFrameAv=[]
    fyFrameAv=[]
    peakFrameAv=[]
    rotAll=[]

    allTrans=[]

    print("Translating ",end="")

    for i in range(nframes):
        print(i+1,', ',end="")
        #use CV2 library to calculate the affine transformation

        transform,xd,yd,sx,sy,rotation=getTransform(xArray[:,i],yArray[:,i],xm,ym,1)
        xdAll.append(xd)
        ydAll.append(yd)
        sxAll.append(sx)
        syAll.append(sy)
        rotAll.append(rotation)

        #calculate the average values
        fxFrameAv.append(fxArray[:,i].mean())
        fyFrameAv.append(fyArray[:,i].mean())
        peakFrameAv.append(peakArray[:,i].mean())

        allTrans.append(transform)

    print()
    #convert data to numpy arrays
    xdAll=np.array(xdAll)
    ydAll=np.array(ydAll)
    fxFrameAv=np.array(fxFrameAv)
    fyFrameAv=np.array(fyFrameAv)
    peakFrameAv=np.array(peakFrameAv)
    sxAll=np.array(sxAll)
    syAll=np.array(syAll)
    rotAll=np.array(rotAll)
    
    return xdAll,ydAll,sxAll,syAll,rotAll,fxFrameAv,fyFrameAv,peakFrameAv,allTrans

def getTransform(x,y,xx,yy,getVals):

    """

    given two sets of registered points, estimate the rigid transformation
    this is in a separate routine mostly for bookkeeping purposes. 
    Returns transformation matrix, and if getVals == 1 returns the 
    extracted parameters (rotation, translation, scale) as well. 

    input:
    x,y: input positions
    xx,yy: transformed positions
    getVales: if ==1, return parameters too

    output: 
    transformation: matrix 
    xd,yd: translations
    sx,sy: scalings
    rotation: rotation (radians)

    """

    #turn data into right form
    pts1=np.zeros((1,len(x),2))
    pts2=np.zeros((1,len(x),2))

    pts1[0,:,0]=x
    pts1[0,:,1]=y

    pts2[0,:,0]=xx
    pts2[0,:,1]=yy

    #float32 is needed
    pts1=np.float32(pts1)
    pts2=np.float32(pts2)

    #calculate the transformation
    transformation = cv2.estimateRigidTransform(pts1, pts2, False)

    afCoeff,inlier=cv2.estimateAffinePartial2D(pts1, pts2)
    
    if(getVals == 0):
        return transformation
    
    if(getVals == 1):

        #extract the parameters


        #print(afCoeff)
        #print(transformation)

        sx=np.sqrt(afCoeff[0,0]**2+afCoeff[0,1]**2)
        sy=np.sqrt(afCoeff[1,0]**2+afCoeff[1,1]**2)
        
        xd=afCoeff[0,2]
        yd=afCoeff[1,2]
         
        rotation=np.arctan2(afCoeff[1,0]/np.sqrt(afCoeff[0,0]**2+afCoeff[0,1]**2),
                                    afCoeff[1,1]/np.sqrt(afCoeff[1,0]**2+afCoeff[1,1]**2))

        
        #sx=np.sqrt(transformation[0,0]**2+transformation[0,1]**2)
        #sy=np.sqrt(transformation[1,0]**2+transformation[1,1]**2)
        #xd=transformation[0,2]
        #yd=transformation[1,2]
        #rotation = np.arctan2(transformation[1,0]/sx,transformation[1,1]/sy)

        return afCoeff,xd,yd,sx,sy,rotation



def transformPointsNew(x,y,xd,yd,theta,sx,sy):

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
