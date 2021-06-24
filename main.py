from utils import *
from ops import *
from grate import *

import sys
import os
from os import listdir
from os.path import isfile

projectPath = '/media/dhruv/data/Dhruv/ISU/PhD/Projects/GRATE/GRATE_for_PennState/' 
dataDir = 'sampleData/'
resultDir = 'Results/temp/'

# projectPath = '/work/adarsh/Dhruv/GRATE/'
# dataDir = 'DATA/'
# resultDir = 'Results/all/skimage_Library/'

AnnotationDir = 'Annotations/'

dataDir = join(dataDir, str(sys.argv[2]))

onlyfiles = [f for f in listdir(join(projectPath,dataDir)) if isfile(join(projectPath,dataDir,f))]

resultDir = join(resultDir, str(sys.argv[2]),str(sys.argv[1]))
if os.path.isdir(join(projectPath, resultDir)) == False:
    os.mkdir(join(projectPath, resultDir))

dspace_nm = float(sys.argv[1]) # 1.9nm, 7A == 0.7nm, 4A == 0.4nm
pix2nm = 78.5
print("\n d space:", dspace_nm)
dspace_pix = int(dspace_nm*pix2nm)
# print("dspace_pix:", dspace_pix)

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


debug               = 0     # To show images: 1, Not to:0
saveImg             = 0     # To save intermediate step images: 1, Not to:0
save_BB             = 0     # To save Bounding box coordinates: 1, Not to: 0
ResultDisp          = 0     # To display final result in notebook: 1, Not to:0
image_scale_percent = 50    # Scaling the image before display

parameters = {'d space pix':                    dspace_pix, 
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
             'show intermediate images':        debug,
             'save intermediate images':        saveImg,
             'save bounding box':               save_BB,
             'show final image':                ResultDisp,
             'display image scaling':           image_scale_percent}


df_overall = pd.DataFrame(columns =['Image Name', 'Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)'])

for f in onlyfiles:
    if f[-4:] == ".tif" and f == "FoilHole_21836544_Data_21829764_21829765_20200123_0129.tif":
        print("Img Name: ", f, "\n")
        # print("Full Img Path: ",join(projectPath,dataDir,f))
        t0 = time.time()    
        df_crystalProps = GRATE(projectPath, dataDir, f, resultDir, AnnotationDir, parameters)
        t1 = time.time()
        total = t1-t0
        print("Overall GRATE Time:", round(total,2), "\n")

        df_overall = df_overall.append(df_crystalProps, ignore_index=True,)

df_overall.to_csv(join(projectPath, resultDir,'overall.csv'))