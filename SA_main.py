from utils import *
from ops import *
from grate import *
 
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skan import skeleton_to_csgraph
from skimage import io, morphology
from plantcv import plantcv as pcv
import math
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
from scipy import spatial
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table, EllipseModel
from skimage.transform import rotate
import pandas as pd 
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import random
from statistics import mean 
import gc
from descartes import PolygonPatch
import alphashape
from sklearn.decomposition import PCA
import os
from os import listdir
from os.path import isfile, join
import sys

projectPath = '/media/dhruv/data/Dhruv/ISU/PhD/Projects/GRATE/GRATE_for_PennState/'       # Local
# projectPath = '/work/adarsh/Dhruv/GRATE/'                                                    # NOVA
dataDir = 'sampleData/sensitivityAnalysis/'
resultDir = 'Results/sensitivityAnalysis/'

AnnotationDir = 'Annotations/'
onlyfiles = [f for f in listdir(projectPath+dataDir) if isfile(join(projectPath+dataDir, f))]

resultDir = join(resultDir, str(sys.argv[1]))
if os.path.isdir(join(projectPath, resultDir)) == False:
    os.mkdir(join(projectPath, resultDir))


dspace_nm = 1.9 # 1.9nm, 7A == 0.7nm, 4A == 0.4nm 
pix2nm = 78.5

dspace_pix = int(dspace_nm*pix2nm)

blur_iteration          = 200                   # prev = 200, 150
dspace_frac_Blur_kernel = 0.09                  # Best till now 0.09, Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Kernel Size
opening_k_size          = 17                    # Kernel Size
pixThresh               = int(1.25*dspace_pix)  # 1.25*dspace_pix,Threshold number of pixels consituting polymers
ellipse_len             = int(1.5*dspace_pix)   # Old = 160,Breaking polymer into this size before constructing ellipse 
ellipseAspectRatio      = 5                     # Threshold aspect Ratio of the ellipse
thresh_dist             = int(1.35*dspace_pix)  # Old = 250, Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 10                    # old = 10, Threshold Crystal cluster size 


debug = 0                               # To print images: 1, Not to:0
saveImg = 0                             # To intermediate step images: 1, Not to:0
save_BB = 0                             # To save Bounding box coordinates: 1, Not to: 0
ResultDisp = 0                          # To display final result in notebook: 1, Not to:0
image_scale_percent = 50                # Scaling the image before display

if sys.argv[1] == "blur_iteration":
    domain = np.arange(100, 310, 10)
elif sys.argv[1] == "dspace_frac_Blur_kernel":
    domain = np.arange(0.05, 0.21, 0.01)
elif sys.argv[1] == "closing_k_size":
    domain = np.arange(9, 21, 2)
elif sys.argv[1] == "opening_k_size":
    domain = np.arange(9, 21, 2)
elif sys.argv[1] == "pixThresh":
    domain = np.arange(0.8, 1.8, 0.1)
elif sys.argv[1] == "ellipse_len":
    domain = np.arange(1.0, 2.1, 0.1)
elif sys.argv[1] == "ellipseAspectRatio":
    domain = np.arange(3, 10, 1)
elif sys.argv[1] == "thresh_dist":
    domain = np.arange(1, 2, 0.1)
elif sys.argv[1] == "thresh_theta":
    domain = np.arange(5, 16, 1)
elif sys.argv[1] == "clusterSize":
    domain = np.arange(5, 16, 1)


for val in domain: 
    # blur_iteration = val

    if sys.argv[1] == "blur_iteration":
        blur_iteration = val
    elif sys.argv[1] == "dspace_frac_Blur_kernel":
        dspace_frac_Blur_kernel = val
    elif sys.argv[1] == "closing_k_size":
        closing_k_size = val
    elif sys.argv[1] == "opening_k_size":
        opening_k_size = val
    elif sys.argv[1] == "pixThresh":
        pixThresh = int(val*dspace_pix)
    elif sys.argv[1] == "ellipse_len":
        ellipse_len = int(val*dspace_pix)
    elif sys.argv[1] == "ellipseAspectRatio":
        ellipseAspectRatio = val
    elif sys.argv[1] == "thresh_dist":
        thresh_dist = int(val*dspace_pix)
    elif sys.argv[1] == "thresh_theta":
        thresh_theta = val
    elif sys.argv[1] == "clusterSize":
        clusterSize = val

    print("Value:", val)
    valDir = join(resultDir, str(val))
    if os.path.isdir(join(projectPath, valDir)) == False:
        os.mkdir(join(projectPath, valDir))
    parameters = {'d space pix':dspace_pix, 
                 'blur iterations': blur_iteration, 
                 'blur k size': dspace_frac_Blur_kernel,
                 'closing k size': closing_k_size,
                 'opening k size': opening_k_size, 
                 'backbone threshold length': pixThresh, 
                 'ellipse pixel size': ellipse_len,
                 'ellipse threshold aspect ratio': ellipseAspectRatio, 
                 'adjacency threshold distance': thresh_dist,
                 'adjacency threshold angle': thresh_theta, 
                 'cluster threshold size': clusterSize,
                 'show intermediate images': debug,
                 'save intermediate images': saveImg,
                 'save bounding box': save_BB,
                 'show final image': ResultDisp,
                 'display image scaling': image_scale_percent}


    df_overall = pd.DataFrame(columns =["Centroid", 'Crystal Area (pixel^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, pixels)'])

    for f in onlyfiles:
        if f[-4:] == ".tif":#and f == "FoilHole_21832497_Data_21829764_21829765_20200122_1417.tif":
            print("Image Path:", join(projectPath,dataDir,f))    
            df_crystalProps = GRATE(projectPath, dataDir, f, valDir, AnnotationDir, parameters)
            df_overall = df_overall.append(df_crystalProps, ignore_index=True,)

    df_overall.to_csv(join(projectPath, valDir,'overall.csv'))
    # print("Final dataframe",df_overall)

    df = df_overall 
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df[['Crystal Area (pixel^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, pixels)']])

    plt.scatter(principalComponents[:,0],principalComponents[:,1] )
    plt.savefig(join(projectPath, valDir,'pca_'+str(val) +'.png'))
    # plt.show()
    plt.close()
    plt.clf()
    print("=====================================================================================")