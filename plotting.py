import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plotfig, totalAreaInDSRange

def Old_filterThresholdArea(dataframe, factor, d_space):
    width           = factor*d_space
    height          = d_space
    areaThresh      = width*height

    dropCount       = 0
    for ind, row in dataframe.iterrows():
        if row['Crystal Area (nm^2)'] < areaThresh or row[ 'D-Spacing(FFT, nm)' ] == 0:
            print("Drop ind   :", ind)
            dropCount += 1
            dataframe = dataframe.drop([ind])
    return dataframe, dropCount

def plotsExceptRelDistance(df, factor, d_space, dSpace_range, csvPath):
    TotDspaceArea   = 0
    dropCount       = 0
    dsRangeCount    = 0

    print("D Spacing                            :", d_space)
    print("Original Number of Detections        :", df.shape[0])

    df, dropCount = Old_filterThresholdArea(df, factor, d_space)

    print("Detections with below Threshold Area :", dropCount)
    # print("After Area Filtering row count       :", df.shape[0])

    TotDspaceArea, dsRangeCount = totalAreaInDSRange(df, dSpace_range)

    print("DS Range Count                       :", dsRangeCount)
    print("Area in Range                        :", round(TotDspaceArea,2))

    plotfig(df['Crystal Area (nm^2)'], csvPath, "area.png", 200, semilog=1)
    plotfig(df['Crystal Angle (zero at X-axis and clockwise positive)'],  csvPath, "angle.png", 150, semilog=0)
    plotfig(df['D-Spacing(FFT, nm)'], csvPath, "dspace.png", 200, semilog=1)
    plotfig(df['D-Spacing(FFT, nm)'], csvPath, "dspaceVsArea.png", 200, df['Crystal Area (nm^2)'], semilog=1)
    plotfig(df['Crystal Area (nm^2)'], csvPath, "AreaVsdspace.png", 200, df['D-Spacing(FFT, nm)'], semilog=1)

csvPath         = 'Results/all/fixedDspacing/overall'
filename        = 'dspacing_1p9.csv'
plotting_relDist = 0    # 0: Generating other plots, 1: Generating Relative Distance Plots.

factor  = 4

d_space = 1.9
dsRange = [1.8, 2.5]

# d_space =  0.7
# dsRange = [0.5, 1.5]

projectPath     = os.path.dirname(os.path.abspath(__file__))
df              = pd.read_csv(os.path.join(projectPath,csvPath,filename))

if plotting_relDist == 1:
    plotfig( df['Distances'], csvPath, "Relative_Distances.png", 300, semilog=1)
elif plotting_relDist == 0:
    plotsExceptRelDistance(df, factor, d_space, dsRange, csvPath)