
import numpy as np
import numpy.ma as ma
from scipy.stats import sigmaclip
import matplotlib.pylab as plt
import visRoutines as vis


def scatterPlot(xs,    ys,    val1,plotrange,titl,      prefix, suffix, valtype,   units,inter,rx,ry,swap,stitle=""):

    #set up plot
    fig,axes=plt.subplots()

    if(swap==1):
        xx=ys
        yy=xs
    else:
        xx=xs
        yy=ys
    
    #scatter plot. size of points optimized for file/notebooks
    sc=axes.scatter(xx,yy,c=val1,marker="o",cmap='Purples',lw=0,vmin=plotrange[0],vmax=plotrange[1])
    fig.colorbar(sc)
    #label the axes
    axes.set_xlabel("X ("+units+")")
    axes.set_ylabel("Y ("+units+")")

    plt.suptitle(valtype)

    ylim=axes.get_ylim()
    xlim=axes.get_xlim()

    if(rx==1):
        axes.set_xlim((xlim[1],xlim[0]))
    if(ry==1):
        axes.set_ylim((ylim[1],ylim[0]))
    axes.axis('equal')

    #show and save
    if(inter == 1):
        plt.show()
    plt.savefig(prefix+suffix+".png")


def pairPlot(xs,    ys,    val1,val2,           plotrange,titl,      prefix, suffix, valtype,   units,   nbins,inter,stitle=""):

    """

    plot a pair of plots, one showing the map, the other the histogram.

    Input: 
       xs,ys - coordinates of points
       val1 - value for the map, same dimensions as xs,ys
       val2 - value for the histogram
       plotrange - [min,max] for polotting range on both plots
       titl - title
       prefix - prefix for output files
       suffix - suffix for output files
       valtype - type of variable plotted (e.g., FWHM (x))
       units - unites (eg pixels)
       nbins - number of bins for histogram
       inter - interactive flag
       stitle - optional subbbtitle.

    REturns

       plots to png files, if inter=1 to screen

    """

    #set up plot
    fig,axes=plt.subplots(1,2)
    fig.set_figheight(4)
    fig.set_figwidth(10)

    print(plotrange)
    if(plotrange==None):

        mm=val1.mean()
        st=val1.std()
        plotrange=[mm-2*st,mm+2*st]
    print(plotrange)

    #scatter plot. size of points optimized for file/notebooks
    sc=axes[0].scatter(xs,ys,c=val1,marker="o",cmap='Purples',lw=0,s=20,vmin=plotrange[0],vmax=plotrange[1])
    #label the axes
    axes[0].set_xlabel("X ("+units+")")
    axes[0].set_ylabel("Y ("+units+")")

    #calculate the bins
    binsize=(plotrange[1]-plotrange[0])/nbins
    bins=np.arange(plotrange[0],plotrange[1]+binsize,binsize)

    #histogram - compressed deals correctly with masked values
    hi=axes[1].hist(val2.compressed(),bins=bins)

    #labels
    axes[1].set_xlabel(valtype)
    axes[1].set_ylabel("N")

    plt.suptitle(valtype+stitle)
    
    #show and save
    if(inter == 1):
        plt.show()
    plt.savefig(prefix+suffix+".png")
 

def checkCentroids(xc,yc,cutrange,prefix,inter):

    """

    Quick plot of centroids to check results

    input

    xc,yc: centroid coordinates
    cutrange: limit the region of plotting if needed (for bad data)
    prefix: prefix for plots

    returns: plot, to screen and file

    """

    fig,ax = plt.subplots()

    #scatter plot
    
    ax.scatter(xc,yc)
    ax.set_aspect('equal')
    
    #display and save
    if(inter == 1):
        plt.show()
    plt.savefig(prefix+"_checkpoints.png")

def checkMatched(xx,yy,xs,ys,prefix,inter):

    """

    quick plotting routine for measured centroids and pinhole coordinates

    input: 

    xx,yy mask coordiantes
    xs,ys: spot coordinates
    prefix: prefix for image files

    """

    
    fig,ax = plt.subplots()
    
    #scatter plot: centroids in circles, mask in red dots
    
    ax.scatter(xs,ys)
    ax.scatter(xx,yy,s=20,color='r')

    #save and show
    plt.savefig(prefix+"_checkpoints1.png")
    if(inter == 1):
        plt.show()

def plotVal(xs,ys,val,limit,plotrange,titl,prefix,suffix,units,inter,stitle=""):

    """
    
    routine for scatter plot of a variable

    input

    xs,ys: coordinates

    val: variable to plot
    limit: upper limit to filter out of plots
    plotrange: colour range in [vmin,vmax] format, or None for default
    title: title for plot
    prefix: prefix for output files
    suffix: suffix for output files

    """

    
    #a quick kludge to filter out bad quality points in poorly focussed images
    if(limit > 0):
        ind=np.where((val < limit) & (val > 0) & (val > 0))
    else:
        ind=np.arange(len(val))

    #scatter plot, with or without ragne limit
    
    fig, axes = plt.subplots()
    
    if(plotrange != None):
        sc=axes.scatter(xs[ind],ys[ind],c=val[ind],marker="o",cmap='Purples',lw=0,s=20,vmin=plotrange[0],vmax=plotrange[1])
    
    else:
        sc=axes.scatter(xs[ind],ys[ind],c=val[ind],marker="o",cmap='Purples',lw=0,s=20)

    fig.colorbar(sc)
    plt.title(titl+stitle)
    plt.xlabel("X ("+units+")")
    plt.ylabel("Y ("+units+")")
    if(inter == 1):
        plt.show()
    plt.savefig(prefix+suffix+".png")

def plotTransByFrame(fxFrameAv,fyFrameAv,peakFrameAv,sxAll,syAll,xdAll,ydAll,rotAll,prefix,inter,stitle=""):

    """

    plot the calculated transformations values and averages by frame number.
    takes output generated by getTransByFrame

    input:
    fxFrameAv,fyFrameAv: average FWHM by frame
    peakFrameAv: peak value by frame

    sxAll,syAll: scale in x and y direction
    xdAll,ydAll: trnaslation in x and y
    rotAll: rotation 

    output: plots

    """

    
    #get number of frames
    frames=np.arange(len(fxFrameAv))

    #first set - fwhms and translation (most useful)
    fig,axes=plt.subplots(1,2)
    fig.set_figheight(4)
    fig.set_figwidth(10)

    axes[0].plot(frames,fxFrameAv,marker='d',linestyle="-",color="#1f77b4")
    axes[0].plot(frames,fyFrameAv,marker='s',linestyle="-",color="#ff7f0e")
    axes[0].set_title("FWHM Average by Frame"+stitle)
    axes[0].set_xlabel("Frame #")
    axes[0].set_ylabel("FHWM (pixels)")

    axes[1].plot(frames,xdAll-xdAll.mean(),marker='d',linestyle="-",color="#1f77b4")
    axes[1].plot(frames,ydAll-ydAll.mean(),marker='s',linestyle="-",color="#ff7f0e")
    axes[1].set_title("Translation Average by Frame"+stitle)
    axes[1].set_xlabel("Frame #")
    axes[1].set_ylabel("Translation (pixels)")
    
    plt.savefig(prefix+"_byframe1.png")
    if(inter == 1):
        fig.show()

    #second set - peaks and bakcgrounds
    fig,axes=plt.subplots(1,2)
    fig.set_figheight(4)
    fig.set_figwidth(10)
        
    axes[0].plot(frames,peakFrameAv,marker='d',linestyle="-")
    axes[0].set_title("Peak Average by Frame"+stitle)
    axes[0].set_xlabel("Frame #")
    axes[0].set_ylabel("Peak")

    axes[1].plot(frames,peakFrameAv,marker='d',linestyle="-")
    axes[1].set_title("Back Average by Frame"+stitle)
    axes[1].set_xlabel("Frame #")
    axes[1].set_ylabel("Back")
    
    plt.savefig(prefix+"_byframe2.png")
    if(inter == 1):
        fig.show()

    #third set - scale and rotation
    fig,axes=plt.subplots(1,2)
    fig.set_figheight(4)
    fig.set_figwidth(10)
 
    axes[0].plot(frames,sxAll,marker='d',linestyle="-")
    axes[0].plot(frames,syAll,marker='d',linestyle="-")
    axes[0].set_title("Scale Average by Frame"+stitle)
    axes[0].set_xlabel("Frame #")
    axes[0].set_ylabel("Scale")
    
    axes[1].plot(frames,rotAll,marker='d',linestyle="-")
    axes[1].set_title("Rotation Average by Frame"+stitle)
    axes[1].set_xlabel("Frame #")
    axes[1].set_ylabel("Rotation (radians)")

    plt.savefig(prefix+"_byframe3.png")
    if(inter == 1):
         fig.show()

    #fourth set - nper frame
         
def plotImageStats(image,prefix,inter,stitle=""):

    """

    plot histogram of an image
    
    """

    
    back = sigmaclip(image, sigma=2, iters=2)
    backImage=back.mean()
    rmsImage=back.std()

    logbins = np.geomspace(image.min(), image.max(), 50)
    
    fig,ax = plt.subplots()
    ax.hist(image.flatten(),bins=logbins,histtype="step")
    print("here")
    plt.title("Histogram of Region of Interest"+stitle)
    plt.xlabel("Flux Value")
    plt.ylabel("N")
    plt.yscale("log")
    plt.savefig(prefix+"_stats.png")

    if(inter == 1):
        fig.show()

    return backImage,rmsImage

def plotValHist(val,plotrange,titl,prefix,suffix,valtype,inter,nbins,stitle=""):

    #routine to plot a histogram of a variable

    fig,ax = plt.subplots()

    binsize=(plotrange[1]-plotrange[0])/nbins
    bins=np.arange(plotrange[0],plotrange[1]+binsize,binsize)
    ax.hist(val,bins=bins)
    plt.title(titl+stitle)
    plt.xlabel(valtype)
    plt.ylabel("N")
    plt.savefig(prefix+"_"+suffix+".png")

    if(inter == 1):
        fig.show()


def makeReg(x,y,outfile):

    """

    Dump a series of points to ds9 region file

    """

    ff=open(outfile,"w")
    for i in range(len(x)):
        print("circle point ",x[i],y[i],file=ff)
    ff.close()

def movieCutout(files,xc,yc,sz,mmin,mmax,prefix):

    """
    
    generate a series of cutouts around an xy point

    """
    
    fig,ax=plt.subplots()

    for i in range(len(files)):
        image=getImage(files[i])
        
        ax.imshow(image[xc-sz:xc+sz,yc-sz:yc+sz],vmin=mmin,vmax=mmax)
        plt.savefig(prefix+str(i).zfill(2)+".png")
