import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plotHist, createVersionDirectory,plotScatter, plotKDE, plotKDE_2D, filterThreshArea
from scipy.stats import wasserstein_distance

def Hist_outside_DS_range(dataframe,d_space, dsRange, savePath, showFig):
    outOfRangeDS    = []
    for ind, row in dataframe.iterrows():
        if row[ 'D-Spacing(FFT, nm)' ] < dsRange[0] or row[ 'D-Spacing(FFT, nm)' ] > dsRange[1]:
            outOfRangeDS.append(row[ 'D-Spacing(FFT, nm)' ])
    
    plotHist(value = outOfRangeDS, path = savePath, filename = str(d_space)+"_outOfRangedspace.png", numBins=200, logscaling=None, xLabel= 'DSpacing out of range', yLabel='Frequency', show=showFig)
    
def Hist_inside_DS_range(dataframe,d_space, dsRange, savePath, showFig):
    inRangeDS    = []
    for ind, row in dataframe.iterrows():
        if row[ 'D-Spacing(FFT, nm)' ] > dsRange[0] and row[ 'D-Spacing(FFT, nm)' ] < dsRange[1]:
            inRangeDS.append(row[ 'D-Spacing(FFT, nm)' ])
    
    # save the list of D-Spacing values in the range
    np.savetxt(os.path.join(savePath,str(d_space)+"_inRangedspace.csv"), inRangeDS, fmt='%10.5f')
    
    plotHist(value = inRangeDS, path = savePath, filename = str(d_space)+"_inRangedspace.png", numBins=200, logscaling=None, xLabel= 'DSpacing inside range', yLabel='Frequency', show=showFig)

    plotKDE(    value       = inRangeDS, 
                wght        = None,
                path        = plotSavePath, 
                filename    = d_space_string + '__inRangedspace_KDE.png', 
                kernel      = 'gaussian',
                bandwidth   = 0.1, 
                logscaling  = None, 
                xLabel      = 'D-Spacing(FFT, nm)', 
                yLabel      = 'Normalized Density', 
                show        = showFig)

""" 
Lookup table for "plotType": 
0   :   Generate KDE based single D-Spacing plots, 
    - Set <csvPath>
    - Set <filename> # <filename> = overall.csv 
    - Set <ShowFig>
    - Set <d_space>
1   :   Generate combined D-Spacing plots, 
2   :   Generate relative Distance Plots., 
3   :   Generate relative angle Plot,
-1  :   Generate histogram based single D-Spacing plots 
"""

plotType        = 0                                 
csvPath         = 'Results/all/combined_v3/version_7/1.9/CSV/'           # Compulsory to fill
filename        = 'overall.csv'                                         # Compulsory to fill
filename1       = 'overall_0p7.csv'                                     # Fill value only if plotType == 1
showFig         = 'no'                                                  # 'yes' or 'no'
d_space         = 1.9

if d_space == 1.9:
    ThresholdFactorArea = 7
elif d_space == 0.7: 
    ThresholdFactorArea = 10

# d_space_string  = filename[-7:-4]
d_space_string  = str(d_space) 
projectPath     = os.path.dirname(os.path.abspath(__file__))
df              = pd.read_csv(os.path.join(projectPath,csvPath,filename))

if plotType == 1: df1 = pd.read_csv(os.path.join(projectPath,csvPath,filename1))

plotSavePath    = createVersionDirectory(projectPath / csvPath, 'Plot_version')

if plotType == 0:
    df              = filterThreshArea(df, { 'threshold area factor': ThresholdFactorArea, 'd space nm': d_space,})
    
    # Save updated dataframe
    df.to_csv(os.path.join(plotSavePath,filename[:-4]+"_areaFiltered.csv"), index=False)
    aspectRatio = df['crystalMajorAxis_length (nm)']/df['crystalMinorAxis_length (nm)']
    
    aspectRatio.to_csv(os.path.join(plotSavePath,filename[:-4]+"_aspectRatio.csv"), index=False)
    
    df_len          = len(df['Crystal Area (nm^2)']) 
    """ # uniformDistFactor = 1
    # uniformDist     = np.random.uniform(df['Crystal Area (nm^2)'].min(), df['Crystal Area (nm^2)'].max(), uniformDistFactor*len(df['Crystal Area (nm^2)']))
    ## Data Sufficiency Test
    df_last = df['Crystal Area (nm^2)'][:int(df_len*10/100)]
    for i in np.linspace(10,100, 10,endpoint=True):
        df_current = df['Crystal Area (nm^2)'][:int(df_len*i/100)]
        plotKDE(    value       = df_current,
                    wght        = None,
                    path        = plotSavePath,
                    filename    = str(i) + "_" + d_space_string + "_area_logx_KDE.png",
                    kernel      = 'gaussian',
                    bandwidth   = 10,
                    logscaling  = 'x',
                    xLabel      = 'Crystal Area (nm^2)',
                    yLabel      = 'Normalized Density',
                    show        = showFig)
        
        if i == 00: 
            continue
        else: 
            # ws_dist = wasserstein_distance(df_last, df_current)
            # ws_dist = wasserstein_distance(df['Crystal Area (nm^2)'], df_current)
            ws_dist = wasserstein_distance(uniformDist, df_current)
            df_last = df_current
            print(ws_dist) """

    Hist_inside_DS_range(df,d_space, [1.5, 4], plotSavePath, showFig)
    
    plotKDE(    value       = df['Crystal Area (nm^2)'],
                wght        = None,
                path        = plotSavePath,
                filename    = d_space_string + "_area_logx_KDE.png",
                kernel      = 'gaussian',
                bandwidth   = None,
                logscaling  = 'x',
                xLabel      = 'Crystal Area (nm^2)',
                yLabel      = 'Normalized Density',
                show        = showFig)

    plotKDE(    value       = df['D-Spacing(FFT, nm)'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = d_space_string + '_dspace_KDE.png', 
                kernel      = 'gaussian',
                bandwidth   = 0.1, 
                logscaling  = None, 
                xLabel      = 'D-Spacing(FFT, nm)', 
                yLabel      = 'Normalized Density', 
                show        = showFig)

    plotKDE_2D( value       = df['D-Spacing(FFT, nm)'], 
                wght        = df['Crystal Area (nm^2)'], 
                path        = plotSavePath, 
                filename    = d_space_string + "_dspaceVsArea_KDE.png",
                kernel      = 'gaussian',
                bandwidth   = None, 
                logscaling  = None, 
                xLabel      = 'D-Spacing(FFT, nm)', 
                yLabel      = 'Crystal Area (nm^2)', 
                show        = showFig)
    
    plotScatter(value       = df['crystalMajorAxis_length (nm)'], 
                wght        = df['crystalMinorAxis_length (nm)'], 
                path        = plotSavePath, 
                filename    = 'crystal_MajorVsMinor_scatter.png', 
                logscaling  = None, 
                xLabel      = 'crystalMajorAxis_length (nm)', 
                yLabel      = 'crystalMinorAxis_length (nm)', 
                show        = showFig)

    plotKDE(    value       = df['crystalMajorAxis_length (nm)']/df['crystalMinorAxis_length (nm)'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = "AspectRatio_MajorbyMinor_KDE.png",
                kernel      = 'gaussian',
                bandwidth   = None,
                logscaling  = None, 
                xLabel      = 'AspectRatio_MajorbyMinor', 
                yLabel      = 'Normalized Density', 
                show        = showFig)
    
    plotKDE(    value       = df['angleDifference'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = "angleDifference_crystalOrientationVsMajorAxis_KDE.png",
                kernel      = 'gaussian',
                bandwidth   = 2,
                logscaling  = None, 
                xLabel      = 'Angle Difference B/W crystal Pattern vs Major Axis', 
                yLabel      = 'Normalized Density', 
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

    plotKDE(    value       = df['Metric Distances'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = filename[:-4] +'_MetricDistances_logx_KDE.png', 
                kernel      = 'gaussian',
                bandwidth   = 0.2, 
                logscaling  = 'x', 
                xLabel      = 'Metric Distances (nm)', 
                yLabel      = 'Normalized Density', 
                show        = showFig)

    plotHist(   value       = df['Metric Distances'],
                wght        = None,
                path        = plotSavePath,
                filename    = filename[:-4] +'_MetricDistances_logx.png',
                numBins     = 300,
                logscaling  = 'x',
                xLabel      = 'Metric Distances (nm)',
                yLabel      = 'Frequency',
                show        = showFig)

    plotKDE(    value       = df['Direct Distances'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = filename[:-4] +'_DirectDistances_logx_KDE.png', 
                kernel      = 'gaussian',
                bandwidth   = 0.9, 
                logscaling  = 'x', 
                xLabel      = 'Direct Distances (nm)', 
                yLabel      = 'Normalized Density', 
                show        = showFig)

    plotHist(   value       = df['Direct Distances'],
                wght        = None,
                path        = plotSavePath,
                filename    = filename[:-4] +'_DirectDistances_logx.png',
                numBins     = 300,
                logscaling  = 'x',
                xLabel      = 'Direct Distances (nm)',
                yLabel      = 'Frequency',
                show        = showFig)
    
    plotKDE_2D( value       = df['Direct Distances'],
                wght        = df['Relative Angle'],
                path        = plotSavePath,
                filename    = filename[:-4] +'_relAngleVsDirectDist.png',
                logscaling  = None,
                xLabel      = 'Direct Distance (nm)',
                yLabel      = 'Relative Angle',
                show        = showFig)
    
    plotKDE_2D( value       = df['Metric Distances'],
                wght        = df['Relative Angle'],
                path        = plotSavePath,
                filename    = filename[:-4] +'_relAngleVsMetricDist.png',
                logscaling  = None,
                xLabel      = 'Metric Distance (nm)',
                yLabel      = 'Relative Angle',
                show        = showFig)

elif plotType == 3:

    plotKDE(    value       = df['Relative Angle'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = 'Relative_Angle_KDE.png', 
                kernel      = 'gaussian',
                bandwidth   = 2.55, 
                logscaling  = None, 
                xLabel      = 'Relative Angle', 
                yLabel      = 'Normalized Density', 
                show        = showFig)
    
    plotHist(   value       = df['Relative Angle'],
                wght        = None,
                path        = plotSavePath,
                filename    = 'Relative_Angle.png',
                numBins     = 36,
                logscaling  = None,
                xLabel      = 'Relative Angle',
                yLabel      = 'Frequency',
                show        = showFig)
    
    plotKDE_2D( value       = df['Relative Angle'],
                wght        = df['Direct Distances'],
                path        = plotSavePath,
                filename    = 'Relative_AngleVsDirect_Distance.png',
                logscaling  = None,
                xLabel      = 'Relative Angle',
                yLabel      = 'Direct Distance',
                show        = showFig)
    
elif plotType == -1:
    df              = filterThreshArea(df, { 'threshold area factor': 10, 'd space nm': d_space,})
    
    plotHist(   value       = df['Crystal Area (nm^2)'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = d_space_string + "_area_logx.png", 
                numBins     = 300, 
                logscaling  = 'x', 
                xLabel      = 'Crystal Area (nm^2)', 
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

    plotHist(   value       = df['crystalMajorAxis_length (nm)']/df['crystalMinorAxis_length (nm)'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = "AspectRatio_MajorbyMinor_histogram.png",
                numBins     = 50,
                logscaling  = None, 
                xLabel      = 'AspectRatio_MajorbyMinor', 
                yLabel      = 'Frequency', 
                show        = showFig)
    
    plotHist(   value       = df['angleDifference'], 
                wght        = None,
                path        = plotSavePath, 
                filename    = "angleDifference_crystalOrientationVsMajorAxis_Histogram.png",
                numBins     = 36,
                logscaling  = None, 
                xLabel      = 'Angle Difference B/W crystal Pattern vs Major Axis', 
                yLabel      = 'Normalized Density', 
                show        = showFig)
