import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils_1 import createVersionDirectory, plotKDE_2D

csvPath         = 'Results/all/combined_v3/version_7/'           # Compulsory to fill
filename        = 'sameDSpacingInfo.csv'                                         # Compulsory to fill
d_space         = 1.9
showFig         = 'no'                                                  # 'yes' or 'no'

d_space_string  = str(d_space) 
projectPath     = os.path.dirname(os.path.abspath(__file__))
df              = pd.read_csv(os.path.join(projectPath,csvPath,filename))

plotSavePath    = createVersionDirectory(projectPath, csvPath, 'Plot_version')

plotKDE_2D( value       = df['Direct Distances'],
                wght        = df['Relative Angle'],
                path        = plotSavePath,
                filename    = filename[:-4] +'_relAngleVsDirectDist.png',
                logscaling  = None,
                xLabel      = 'Direct Distance (nm)',
                yLabel      = 'Relative Angle',
                show        = showFig)