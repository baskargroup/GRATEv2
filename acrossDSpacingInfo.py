import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join, splitext

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
dataDir = 'Results/all'
d_spaceDir1 = join(projectPath, dataDir, "version_4/0.7")
d_spaceDir2 = join(projectPath, dataDir, "version_3/1.9")
pix2nm      = 78.5

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
    print("File name:   ", filename)
    df1 = pd.read_csv(join(d_spaceDir1, filename))
    df2 = pd.read_csv(join(d_spaceDir2, filename))

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
            
            MetricDistances.append(MetricDist)              ## Metric Distance
            DirectDistances.append(CCdist)                  ## Direct Distance
            ModRelAngle.append(abs(ang1 - ang2))            ## Absolute Value of the Angle Difference 

print("len distances:   ", len(DirectDistances))

df_dist = pd.DataFrame(list(zip(MetricDistances, DirectDistances, ModRelAngle)), columns=['Metric Distances','Direct Distances','Modulus Relative Angle'])
df_dist.to_csv(join(projectPath, dataDir, "acrossDSpacingInfo.csv"))