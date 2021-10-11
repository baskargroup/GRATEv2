from utils import *
from ops import *
from grate import *

import sys
import os
import re
from os import listdir
from os.path import isfile

projectPath     = os.path.dirname(os.path.abspath(__file__))
dataDir         = 'DATA/sampleData/'
BaseResultDir   = 'Results/temp/'

ResultImageDir      = 'Images'
ResultCSVDir        = 'CSV'
ResultAnnotationDir = 'Annotations'

BaseResultDir   = createVersionDirectory(projectPath, BaseResultDir, 'version')

'''
Command Line Arguments:
sys.argv[1] : The D-Spacing value (eg. 1.9, 0.7) at which the algorithm will run. 
'''

# dataDir     = join(dataDir, str(sys.argv[1]))
resultDir   = join(BaseResultDir, str(sys.argv[1]))
dspace_nm   = float(sys.argv[1]) # 1.9nm, 7A == 0.7nm, 4A == 0.4nm

pix2nm      = 78.5
dspace_pix  = int(dspace_nm*pix2nm)

blur_iteration          = 15                    # Number of Blur Iteration 
Blur_kernel             = int(0.15*dspace_pix)  # Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Closing Kernel Size
opening_k_size          = 17                    # Opening Kernel Size
pixThresh               = int(0.625*dspace_pix) # Threshold number of pixels consituting Backbone
ellipse_len             = int(1.5*dspace_pix)   # Breaking Backbone into uniform size before constructing ellipse 
ellipseAspectRatio      = 5                     # Threshold ellipse aspect Ratio 
thresh_dist             = int(2*dspace_pix)     # Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 7                     # Threshold ellipse in Crystal cluster
powSpec_peak_thresk     = 1.15                  # 1.20 works for all
Thresh_area_factor      = 4


debug               = 0     # To show images: 1, Not to:0
saveImg             = 0     # To save intermediate step images: 1, Not to:0
save_BB             = 0     # To save Bounding box coordinates: 1, Not to: 0
ResultDisp          = 0     # To display final result in notebook: 1, Not to:0
image_scale_percent = 50    # Scaling the image before display

parameters = {'d space nm':                     dspace_nm,
             'd space pix':                     dspace_pix, 
             'pix to nm':                       pix2nm,
             'blur iterations':                 blur_iteration, 
             'blur k size':                     Blur_kernel,
             'closing k size':                  closing_k_size,
             'opening k size':                  opening_k_size, 
             'backbone threshold length':       pixThresh, 
             'ellipse pixel size':              ellipse_len,
             'ellipse threshold aspect ratio':  ellipseAspectRatio,
             'adjacency threshold distance':    thresh_dist,
             'adjacency threshold angle':       thresh_theta, 
             'cluster threshold size':          clusterSize,
             'pow spec peak vs mean factor':    powSpec_peak_thresk,
             'show intermediate images':        debug,
             'save intermediate images':        saveImg,
             'save bounding box':               save_BB,
             'show final image':                ResultDisp,
             'display image scaling':           image_scale_percent,
             'Threshold area factor':           Thresh_area_factor}


df_overall = pd.DataFrame(columns =['Image Name', 'Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)'])

if os.path.isdir(join(projectPath, resultDir)) == False: os.mkdir(join(projectPath, resultDir))
if os.path.isdir(join(projectPath, resultDir, ResultCSVDir)) == False: os.mkdir(join(projectPath, resultDir, ResultCSVDir))
if os.path.isdir(join(projectPath, resultDir, ResultImageDir)) == False: os.mkdir(join(projectPath, resultDir, ResultImageDir))

if save_BB == 1: 
    if os.path.isdir(join(projectPath, resultDir, ResultAnnotationDir)) == False: os.mkdir(join(projectPath, resultDir, ResultAnnotationDir))

onlyfiles = [f for f in listdir(join(projectPath,dataDir)) if isfile(join(projectPath,dataDir,f))]

print("\nd space:", dspace_nm)

for f in onlyfiles:

    if f[-4:] == ".tif" :#and f == "FoilHole_21830219_Data_21829764_21829765_20200122_1016.tif":
        print("Img Name: ", f, "\n")
        t0 = time.time()    
        df_crystalProps = GRATE(projectPath, dataDir, f, resultDir, ResultImageDir, ResultCSVDir, ResultAnnotationDir, parameters)
        t1 = time.time()
        total = t1-t0
        print("Overall GRATE Time:", round(total,2), "\n")
        df_overall = df_overall.append(df_crystalProps, ignore_index=True,)

df_overall.to_csv(join(projectPath, resultDir, ResultCSVDir, 'overall.csv'))