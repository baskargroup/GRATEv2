import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join, splitext
from utils import plotHist, filterThreshArea, getAngleDifference

def listIntersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def numericFromString(str, pix2nm):
    str = str[1:-1]
    num = str.split(", ")
    num = list(map(int, num))
    num[0] = num[0]/pix2nm
    num[1] = num[1]/pix2nm
    return num

def centroidDist(cnt1, cnt2):
    np1 = np.array(cnt1)
    np2 = np.array(cnt2)
    return np.linalg.norm(np1-np2)

def radFromArea(area):
    rad = np.sqrt(area/np.pi)
    return rad

projectPath     = os.path.dirname(os.path.abspath(__file__))
dataDir         = 'Results/all/combined_v3/version_8'           ## Set this value
d_space         = 0.7                                           ## Set this value
ThresholdFactorArea = 10                                        ## Set this value (:7 for 1.9nm crystal detection, :10 for 0.7nm crystal detect)
d_spaceDir1     = join(projectPath, dataDir, str(d_space), 'CSV')
pix2nm          = 78.5

files1 = [f for f in listdir(d_spaceDir1) if splitext(f)[1] == ".csv"]

print("len files1:  ",len(files1))

distances = []

MetricDistances = []
DirectDistances = []
ModRelAngle     = []

for filename in files1:
    if filename == "overall.csv":
        continue
    print("File name:   ", filename)
    df1 = pd.read_csv(join(d_spaceDir1, filename))
    df1 = filterThreshArea(df1, { 'Threshold area factor': ThresholdFactorArea, 'd space nm': d_space})

    for ind1, row1 in df1.iterrows():
        # print("ind1     :", ind1)
        ang1       = float(row1['Crystal Angle (zero at X-axis and clockwise positive)'])
        area1       = float(row1['Crystal Area (nm^2)'])
        rad1        = radFromArea(area1)
        centroid1   = numericFromString(row1['Centroid'], pix2nm)

        for ind2, row2 in df1.iterrows():
            # print("ind2     :", ind2)
            if ind1 == ind2:
                continue
            else:
                ang2        = float(row2['Crystal Angle (zero at X-axis and clockwise positive)'])
                area2       = float(row2['Crystal Area (nm^2)'])
                rad2        = radFromArea(area2)
                centroid2   = numericFromString(row2['Centroid'], pix2nm)
                
                CCdist      = centroidDist(centroid1, centroid2)
                MetricDist   = CCdist/ (rad1 + rad2)
                
                MetricDistances.append(MetricDist)
                DirectDistances.append(CCdist)
                ModRelAngle.append(getAngleDifference(ang1, ang2))
            
print("length:   ", len(ModRelAngle))

df_dist = pd.DataFrame(list(zip(MetricDistances, DirectDistances, ModRelAngle)), columns=['Metric Distances','Direct Distances','Relative Angle'])
df_dist = df_dist.round(2)
df_dist.to_csv(join(projectPath, dataDir, "sameDSpacingInfo.csv"))