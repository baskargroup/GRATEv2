import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csvPath = '/media/dhruv/data/Dhruv/ISU/PhD/Projects/GRATE/GRATE_for_PennState/Results/all/skimage_Library/1.9/'
df = pd.read_csv(csvPath +'overall.csv')
area = df['Crystal Area (nm^2)']
angle = df['Crystal Angle (zero at X-axis and clockwise positive)']
dspace = df['D-Spacing(FFT, nm)']

_ = plt.hist(area, bins=100)
plt.title("area")
plt.semilogx()
plt.show()

_ = plt.hist(angle, bins=100)
plt.title("angle")
plt.show()

_ = plt.hist(dspace, bins=200)
plt.title("dspace")
plt.semilogx()
plt.show()