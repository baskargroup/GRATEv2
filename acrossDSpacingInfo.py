import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join, splitext
from utils import getAngleDifference, filterThreshArea

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
dataDir         = 'Results/all/combined_v3'
d_spaceDir1     = join(projectPath, dataDir, "version_7/1.9/CSV")       ## Set 1.9nm Directory Here
d_spaceDir2     = join(projectPath, dataDir, "version_8/0.7/CSV")       ## Set 0.7nm Directory Here
pix2nm          = 78.5

files1 = [f for f in listdir(d_spaceDir1) if splitext(f)[1] == ".csv"]
files2 = [f for f in listdir(d_spaceDir2) if splitext(f)[1] == ".csv"]
commonCSV   = listIntersection(files1, files2)

print("len files1:  ",len(files1))
print("len files2:  ",len(files2))
print("len commonCSV", len(commonCSV))

MetricDistances = []
DirectDistances = []
ModRelAngle     = []

for filename in commonCSV:
    if filename == "overall.csv":
        continue
    # print("File name:   ", filename)
    df1 = pd.read_csv(join(d_spaceDir1, filename))
    df1 = filterThreshArea(df1, { 'Threshold area factor': 7, 'd space nm': 1.9})
    
    df2 = pd.read_csv(join(d_spaceDir2, filename))
    df2 = filterThreshArea(df2, { 'Threshold area factor': 10, 'd space nm': 0.7})

    for ind1, row1 in df1.iterrows():
        centroid1   = numericFromString(row1['Centroid'], pix2nm) 
        area1       = float(row1['Crystal Area (nm^2)'])
        rad1        = radFromArea(area1)
        ang1        = float(row1['Crystal Angle (zero at X-axis and clockwise positive)'])

        for ind2, row2 in df2.iterrows():
            centroid2   = numericFromString(row2['Centroid'], pix2nm) 
            area2       = float(row2['Crystal Area (nm^2)'])
            rad2        = radFromArea(area2)
            ang2        = float(row2['Crystal Angle (zero at X-axis and clockwise positive)'])

            CCdist      = centroidDist(centroid1, centroid2)
            MetricDist   = CCdist/ (rad1 + rad2)
            
            MetricDistances.append(MetricDist)                              ## Metric Distance
            DirectDistances.append(CCdist)                                  ## Direct Distance
            ModRelAngle.append(getAngleDifference(ang1, ang2))              ## Absolute Value of the Angle Difference

print("len distances:   ", len(DirectDistances))

df_dist = pd.DataFrame(list(zip(MetricDistances, DirectDistances, ModRelAngle)), columns=['Metric Distances','Direct Distances','Relative Angle'])
df_dist = df_dist.round(2)
df_dist.to_csv(join(projectPath, dataDir, "acrossDSpacingInfo.csv"))