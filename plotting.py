import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plotHist, createVersionDirectory, totalAreaInDSRange, isAreaSmall, plotScatter, plotKDE


def Hist_outside_DS_range(dataframe,d_space, dsRange, savePath, showFig):
    outOfRangeDS    = []
    for ind, row in dataframe.iterrows():
        if row[ 'D-Spacing(FFT, nm)' ] < dsRange[0] or row[ 'D-Spacing(FFT, nm)' ] > dsRange[1]:
            outOfRangeDS.append(row[ 'D-Spacing(FFT, nm)' ])
    
    plotHist(value = outOfRangeDS, path = savePath, filename = str(d_space)+"_outOfRangedspace.png", numBins=200, logscaling=None, xLabel= 'DSpacing out of range', yLabel='Frequency', show=showFig)

def filterThreshArea(df, params):
    for ind, row in df.iterrows():
        TorF = isAreaSmall(row['Crystal Area (nm^2)'], params, areaInPix=False)
        if TorF == True:
            df.drop(ind, inplace = True)
    
    return df


# 0: Generating Single D-Spacing plots, 1: Generating combined D-Spacing plots, 2: Generating Relative Distance Plots., 3: Generate Relative Orientation Plot
plotType        = 0                                 
csvPath         = 'Results/all/version_3'           # Compulsory to fill
filename        = 'overall_1p9.csv'                 # Compulsory to fill
filename1       = 'overall_0p7.csv'                 # Fill value only if plotType == 1
showFig         = 'no'                              # 'yes' or 'no'
# d_space         = 1.9
# dsRange         = [1.52, 2.28]

# d_space = 0.7
# dsRange = [0.56, 0.84]

d_space_string = filename[-7:-4] 
projectPath     = os.path.dirname(os.path.abspath(__file__))
df              = pd.read_csv(os.path.join(projectPath,csvPath,filename))

# df              = filterThreshArea(df, { 'Threshold area factor': 10, 'd space nm': d_space,})

if plotType == 1: df1 = pd.read_csv(os.path.join(projectPath,csvPath,filename1))

plotSavePath    = createVersionDirectory(projectPath, csvPath, 'Plot_version')

if plotType == 0:
    """ 
    plotHist(   value       = df['Crystal Area (nm^2)'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = d_space_string + "_area_bothLog.png", 
                numBins     = 400, 
                logscaling  = 'both', 
                xLabel      = 'Crystal Area (nm^2)', 
                yLabel      = 'Frequency', 
                show        = showFig)
     """
    
    plotKDE(    value       = df['Crystal Area (nm^2)'],
                wght        = None,
                path        = plotSavePath,
                filename    = d_space_string + "_area_bothLog_KDE.png",
                kernel      = 'gaussian',
                logscaling  = 'both',
                xLabel      = 'Crystal Area (nm^2)',
                yLabel      = 'Frequency',
                show        = showFig)

    plotHist(   value       = df['Crystal Area (nm^2)'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = d_space_string + "_area_logx.png", 
                numBins     = 300, 
                logscaling  = 'x', 
                xLabel      = 'Crystal Area (nm^2)', 
                yLabel      = 'Frequency', 
                show        = showFig)

    plotHist(   value       = df['Crystal Angle (zero at X-axis and clockwise positive)'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = d_space_string + "_angle.png",
                numBins     = 36, 
                logscaling  = None, 
                xLabel      = 'Crystal Angle', 
                yLabel      = 'Frequency', 
                show        = showFig)

    plotHist(   value       = df['D-Spacing(FFT, nm)'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = d_space_string + '_dspace.png', 
                numBins     = 20, 
                logscaling  = None, 
                xLabel      = 'D-Spacing(FFT, nm)', 
                yLabel      = 'Frequency', 
                show        = showFig)

    plotHist(   value       = df['D-Spacing(FFT, nm)'], 
                wght        = df['Crystal Area (nm^2)'], 
                path        = plotSavePath, 
                filename    = d_space_string + "_dspaceVsArea.png",
                numBins     = 20, 
                logscaling  = None, 
                xLabel      = 'D-Spacing(FFT, nm)', 
                yLabel      = 'Crystal Area (nm^2)', 
                show        = showFig)

elif plotType == 1:
    frames = [df, df1]
    resultDF = pd.concat(frames)

    plotHist(   value       = resultDF['D-Spacing(FFT, nm)'], 
                wght        = resultDF['Crystal Area (nm^2)'],
                path        = plotSavePath, 
                filename    = "combined_dspaceVsArea_log.png", 
                numBins     = 20, 
                logscaling  = 'y',
                xLabel      = 'D-Spacing(FFT, nm)',
                yLabel      = 'Crystal Area (nm^2)', 
                show        = showFig)

    plotHist(   value       = resultDF['D-Spacing(FFT, nm)'],
                wght        = resultDF['Crystal Area (nm^2)'],
                path        = plotSavePath,
                filename    = 'combined_dspaceVsArea.png',
                numBins     = 20,
                logscaling  = None,
                xLabel      = 'D-Spacing(FFT, nm)',
                yLabel      = 'Crystal Area (nm^2)',
                show        = showFig)

elif plotType == 2:
    plotHist(   value       = df['Distances'],
                wght        = None,
                path        = plotSavePath,
                filename    = filename[:-4] +'_logx.png',
                numBins     = 300,
                logscaling  = 'x',
                xLabel      = 'Relative Distances (nm)',
                yLabel      = 'Frequency',
                show        = showFig)

elif plotType == 3:
    plotHist(   value       = df['relModAngle'],
                wght        = None,
                path        = plotSavePath,
                filename    = 'relModAngle.png',
                numBins     = 36,
                logscaling  = None,
                xLabel      = 'Relative Modulus Angle',
                yLabel      = 'Frequency',
                show        = showFig)
    """ 
    plotHist(   value       = df['relModAngle'],
                wght        = df['distance'],
                path        = plotSavePath,
                filename    = 'relAngleVsCrystalDist.png',
                numBins     = 36,
                logscaling  = None,
                xLabel      = 'Relative Modulus Angle',
                yLabel      = 'crystal distance',
                show        = showFig) """
    
    plotScatter(value       = df['relModAngle'],
                wght        = df['distance'],
                path        = plotSavePath,
                filename    = 'relAngleVsCrystalDist.png',
                logscaling  = 'y',
                xLabel      = 'Relative Modulus Angle',
                yLabel      = 'crystal distance',
                show        = showFig)
    
