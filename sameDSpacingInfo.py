import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join, splitext
from utils import plotHist

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
dataDir         = 'Results/all'
d_spaceDir1     = join(projectPath, dataDir, "version_3/1.9")
pix2nm          = 78.5

files1 = [f for f in listdir(d_spaceDir1) if splitext(f)[1] == ".csv"]

print("len files1:  ",len(files1))

modRelAngle = []
distances = []

for filename in files1:
    if filename == "overall.csv":
        continue
    print("File name:   ", filename)
    df1 = pd.read_csv(join(d_spaceDir1, filename))

    for ind1, row1 in df1.iterrows():
        ang1       = float(row1['Crystal Angle (zero at X-axis and clockwise positive)'])
        centroid1   = numericFromString(row1['Centroid'], pix2nm)

        for ind2, row2 in df1.iterrows():
            ang2        = float(row2['Crystal Angle (zero at X-axis and clockwise positive)'])
            if abs(ang1 - ang2) == 0:
                continue
            else:
                centroid2   = numericFromString(row2['Centroid'], pix2nm)
                CCdist      = centroidDist(centroid1, centroid2)
                distances.append(CCdist)
                modRelAngle.append(abs(ang1 - ang2))
            
print("length:   ", len(modRelAngle))

df_dist = pd.DataFrame(list(zip(modRelAngle, distances)), columns=['modRelAngle', 'distance'])
df_dist.to_csv(join(projectPath, dataDir, "relAngleAndDist.csv"))