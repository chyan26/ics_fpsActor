"""

This module contains various routines to plot overviews of the results
from the engineering run.

Many of the routines require as input a tsv spreadsheet with results
obtained from the analysis routines (average RMS, average FWHMs,
physical parameters, etc).

"""


import numpy as np
import pylab as plt
from astropy.io import fits
import visRoutines as vis

def sumPlots(data,syms,cols,var1,var2,titl,xlab,ylab,ylim,prefix,day):

    """

    Plot a summary plot for a given parameter, for one day or all days

    Input:
      data: data structure from spreadsheet
      syms, cols: symbol and colour information
      var1: x variable
      var2: y variable
      titl: title
      xlab, ylab: labels fo raxes
      ylim: y limits
      prefix: for saving files
      day #, or -1 for all

    Output
      plot file

    """

    plt.ioff()

    #create the axis, making it longer for all days and time axis
    if(day==-1):
        if(var1=='reltime'):
            fig,axes=plt.subplots(1,1,figsize=[15,4])
        else:
            fig,axes=plt.subplots(1,1,figsize=[6,4])

    else:
        fig,axes=plt.subplots(1,1,figsize=[6,4])

    #extract teh data to plot
    x=[]
    y=[]
    adj=[]
    tval=[]
    zval=[]
    nvals=0
    for i in range(len(data[var1])):
        if(day==-1):

            #the two variables
            x.append(data[var1][i])
            y.append(data[var2][i])

            #adjust the relative time for all day plots
            if(data['day'][i]<=3):
                adj.append((data['day'][i]-1)*12)
            else:
                adj.append((data['day'][i]-3)*8+24)

            #time and elevation data
            tval.append(str(int(data['t'][i])))
            zval.append(str(int(data['el'][i])))
            nvals+=1
            
        elif(data['day'][i] == day):

            #the two variables
            x.append(data[var1][i])
            y.append(data[var2][i])

            #time and elevation data

            tval.append(str(int(data['t'][i])))
            zval.append(str(int(data['el'][i])))
            adj.append(0)
            nvals+=1

    #turn into numpy arrays
    x=np.array(x)
    y=np.array(y)
    adj=np.array(adj)
    #plot the data
    ii=0

    if(ylim == None):
        ylim=[y.min(),y.max()]
    
    for i in range(nvals):

        
        if(var1=='reltime'):
            if(day==-1):
                axes.plot([x[i]+adj[i]],[y[i]],color=cols[tval[i]],marker=syms[zval[i]])
            else:
                axes.plot([x[i]],[y[i]],color=cols[tval[i]],marker=syms[zval[i]])
        else:
            axes.plot([x[i]],[y[i]],color=cols[tval[i]],marker=syms[zval[i]])

    #if(day==1):
        #axes[ii].set_xlim([-4.5,4.5])

    #set the subtitles
    if(day==-1):
        stitle=' [All Days]'
    else:
        stitle=" [Day "+str(int(day))+"]"

    #add various lines for telescope opening and closing
    if(var1=='reltime'):
        if(day==1):
            axes.axvline(x=26.53,color='blue',linestyle="-")
            axes.axvline(x=23.948,color='blue',linestyle='--')
        if(day==2):
            axes.axvline(x=22.746,color='blue',linestyle="-")
            axes.axvline(x=22.914,color='blue',linestyle="--")
            axes.axvline(x=23.082,color='blue',linestyle="-")
            axes.axvline(x=23.428,color='blue',linestyle="--")
            axes.axvline(x=25.026,color='green',linestyle="-")
            axes.axvline(x=27.209,color='green',linestyle="--")
        if(day==3):
            axes.axvline(x=22.754,color='blue',linestyle="-")
            axes.axvline(x=22.874,color='blue',linestyle="--")
            axes.axvline(x=22.989,color='blue',linestyle="-")
            axes.axvline(x=23.105,color='blue',linestyle="--")
            
            axes.axvline(x=26.881,color='green',linestyle="-")

        if(day==-1):
            axes.axvline(x=26.53,color='blue',linestyle="-")
            axes.axvline(x=23.948,color='blue',linestyle='--')
            axes.axvline(x=22.746+12,color='blue',linestyle="-")
            axes.axvline(x=22.914+12,color='blue',linestyle="--")
            axes.axvline(x=23.082+12,color='blue',linestyle="-")
            axes.axvline(x=23.428+12,color='blue',linestyle="--")
            axes.axvline(x=25.026+12,color='green',linestyle="-")
            axes.axvline(x=27.209+12,color='green',linestyle="--")
            axes.axvline(x=22.754+24,color='blue',linestyle="-")
            axes.axvline(x=22.874+24,color='blue',linestyle="--")
            axes.axvline(x=22.989+24,color='blue',linestyle="-")
            axes.axvline(x=23.105+24,color='blue',linestyle="--")
            axes.axvline(x=26.881+24,color='green',linestyle="-")

            axes.axvline(x=30,color='black',linestyle="-",linewidth=2)
            axes.axvline(x=43,color='black',linestyle="-",linewidth=2)
            axes.axvline(x=53,color='black',linestyle="-",linewidth=2)
            axes.axvline(x=63,color='black',linestyle="-",linewidth=2)


    #set the limits and labels
    axes.set_ylim(ylim)
    axes.set_ylabel(ylab)
    axes.set_xlabel(xlab)
    plt.suptitle(titl+stitle,y=0.99)

    #tight layout
    plt.tight_layout()

    #set the file name and save
    if(day==-1):
        plt.savefig(prefix+"_"+str(var1)+"_"+str(var2)+"_allday.png")
    else:
        plt.savefig(prefix+"_"+str(var1)+"_"+str(var2)+"_day"+str(int(day))+".png")

    plt.close('all')

def massageData(data):

    """
    
    Tidy the data for nicer plots

    """

    
    #adjust relative time to account for midnight
    ind=data['reltime'] < 15
    data['reltime'][ind]=data['reltime'][ind]+24

    #round the elevations
    ind=np.where(abs(data['el']-90)<5)
    data['el'][ind]=90
    ind=np.where(abs(data['el']-60)<5)
    data['el'][ind]=60
    ind=np.where(abs(data['el']-45)<5)
    data['el'][ind]=45
    ind=np.where(abs(data['el']-30)<5)
    data['el'][ind]=30
    ind=np.where(abs(data['el']-75)<5)
    data['el'][ind]=75

    #roudn the instrument rotation
    data['rot']=(np.round(data['rot']) // 5)*5

    return data

def setSymbols():

    """

    Structures with plotting symbols nad colorus

    """
    
    syms={}
    syms['90']='o'
    syms['75']='s'
    syms['60']='d'
    syms['45']='^'
    syms['30']='x'

    cols={}
    cols['0']='red'
    cols['1']='orange'
    cols['2']='yellow'
    cols['5']='green'
    cols['8']='blue'
    cols['10']='purple'
    cols['15']='purple'
    cols['30']='purple'

    return syms,cols

def oneNight(data,day,syms,cols):

    """ 

    do one set of plots for summaries, one day or all days

    """
    
    xlim=None
    suffix=''
    stitle=" "
    
    #rms, fwhms vs time
    ylim=[0,0.01]
    sumPlots(data,syms,cols,'reltime','rmsmm','Spot Motion RMS vs Time',"Time (hrs)",'Spot Motion RMS (mm)',ylim,'over',day)
    
    ylim=[1,5]
    sumPlots(data,syms,cols,'reltime','fx','FWHM(x) vs Time',"Time (hrs)",'FWHM(x)',ylim,'over',day)
    sumPlots(data,syms,cols,'reltime','fy','FWHM(y)',"Time (hrs)",'FWHM(y)',ylim,'over',day)
    
    ylim=[0,0.01]
    
    sumPlots(data,syms,cols,'id','rmsmm','Spot Motion RMS vs Frame ID',"Frame ID",'Spot Motion RMS (Pixels)',ylim,'over',day)
    
    ylim=[1,5]
    sumPlots(data,syms,cols,'id','fx','FWHM(x) vs Frame ID',"Frame ID",'FWHM(x)',ylim,'over',day)
    sumPlots(data,syms,cols,'id','fy','FWHM(y)',"Frame ID",'FWHM(y)',ylim,'over',day)
    
    
    #rms, fwhms vs elevation
    ylim=[0,0.01]
    sumPlots(data,syms,cols,'el','rmsmm','Spot Motion RMS vs Elevation',"Elevation",'Spot Motion RMS (mm)',ylim,'over',day)
    ylim=[1,5]
    sumPlots(data,syms,cols,'el','fx','FWHM(x) vs Elevation',"Elevation",'FWHM(s)',ylim,'over',day)
    sumPlots(data,syms,cols,'el','fy','FWHM(y) vs Elevation',"Elevation",'FWHM(y)',ylim,'over',day)
    
    #rms, fwhms vs exposure time
    ylim=[0,0.01]
    sumPlots(data,syms,cols,'t','rmsmm','Spot Motion RMS vs Exposure Time',"Exposure Time",'Spot Motion RMS (mm)',ylim,'over',day)
    ylim=[1,5]
    sumPlots(data,syms,cols,'t','fx','FWHM(x) vs Exposure Time',"Exposure Time",'FWHM(s)',ylim,'over',day)
    sumPlots(data,syms,cols,'t','fy','FWHM(y) vs Exposure Time',"Exposure Time",'FWHM(y)',ylim,'over',day)
    
    #fx vs rms and each other
    
    ylim=[0,0.01]
    sumPlots(data,syms,cols,'fx','rmsmm','Spot Motion RMS vs FWHM(x)',"FWHM(x)",'Spot Motion RMS (mm)',ylim,'over',day)
    sumPlots(data,syms,cols,'fy','rmsmm','Spot Motion RMS vs FWHM(y)',"FWHM(y)",'Spot Motion RMS (mm)',ylim,'over',day)
    ylim=[1,5]
    
    sumPlots(data,syms,cols,'fx','fy','FWHM(y) vs FWHM(x)',"FWHM(x)",'FWHM(y)',ylim,'over',day)
    
    ylim=None
    sumPlots(data,syms,cols,'reltime','domhum','Dome Humidity vs time','time',"Dome Humidity",ylim,'over',day)
    sumPlots(data,syms,cols,'reltime','domwnd','Dome Wind Speed vs time',"time,","Dome Wind Speed",ylim,'over',day)
    sumPlots(data,syms,cols,'reltime','domprs','Dome Pressure vs time',"time","Dome Pressure",ylim,'over',day)
    sumPlots(data,syms,cols,'reltime','domtmp','Dome Temperature vs time',"time","Dome Temperature",ylim,'over',day)
    sumPlots(data,syms,cols,'reltime','outtmp','Outside Temperature vs time',"time","Outside Temperature",ylim,'over',day)
    sumPlots(data,syms,cols,'reltime','mcm1t','Mirror Temperature vs time',"time","Mirror Temperature",ylim,'over',day)

    
def calcReltime(time_string):

    """
    create  a decimal relative time from the time string. Used when generating the spreadsheet.

    """
    
    #split the time string into fields
    fields = time_string.split(":")

    #grt hours, minutes, seconds
    hours = fields[0] 
    minutes = fields[1]
    seconds = fields[2]

    #return values
    return float(hours) + (float(minutes) / 60.0) + (float(seconds) / pow(60.0, 2))

def readSpread(fname):

    data = np.genfromtxt(fname, delimiter=' ', names=True,dtype=None)
    data = massageData(data)
    return data

def runSet(day):

    """

    wrapper to run various analysis. 

    """
    
    syms,cols=setSymbols()

    infile="rmsStatsAll.csv"
    data=readSpread(infile)

    oneNight(data,day,syms,cols)
    

if __name__ == "__main__":    

    
    #runSet(1)
    #runSet(2)
    #runSet(3)
    #runSet(4)
    #runSet(5)
    runSet(-1)

    
