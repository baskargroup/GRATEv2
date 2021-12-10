from utils import *
from ops import *
from grate import *

import sys
import os
import re
from os import listdir
from os.path import isfile
import io, libconf

'''
Command Line Arguments:
sys.argv[1] : .cfg file name present inside the configFiles directory.  
'''

projectPath     = os.path.dirname(os.path.abspath(__file__))

with io.open(join(projectPath,'configFiles', sys.argv[1])) as f:
    config = libconf.load(f)

dataDir             = config['dataDir']
BaseResultDir       = config['BaseResultDir']
ResultImageDir      = config['ResultImageDir']
ResultCSVDir        = config['ResultCSVDir']
ResultAnnotationDir = config['ResultAnnotationDir']

dspace_nm           = config['dspace_nm']                       # 1.9nm, 7A == 0.7nm, 4A == 0.4nm
pix2nm              = config['pix2nm']

Blur_kernel_propCons    = config['Blur_kernel_propCons'] 
pixThresh_propCons      = config['pixThresh_propCons']
ellipse_len_propCons    = config['ellipse_len_propCons'] 
thresh_dist_propCons    = config['thresh_dist_propCons']

dspace_pix              = int(dspace_nm*pix2nm)
Blur_kernel             = int(Blur_kernel_propCons*dspace_pix)      # Fraction of dspace for the blur kernel size
pixThresh               = int(pixThresh_propCons*dspace_pix)        # Threshold number of pixels consituting Backbone
ellipse_len             = int(ellipse_len_propCons*dspace_pix)      # Breaking Backbone into uniform size before constructing ellipse
thresh_dist             = int(thresh_dist_propCons*dspace_pix)      # Distance threshold for adjacency matrix

BaseResultDir   = createVersionDirectory(projectPath, BaseResultDir, 'version')
resultDir       = join(BaseResultDir, str(dspace_nm))

parameters = {'d space nm':                     dspace_nm,
             'd space pix':                     dspace_pix, 
             'pix to nm':                       pix2nm,
             'blur iterations':                 config['blur_iteration'], 
             'blur k size':                     Blur_kernel,
             'closing k size':                  config['closing_k_size'],
             'opening k size':                  config['opening_k_size'], 
             'backbone threshold length':       pixThresh, 
             'ellipse pixel size':              ellipse_len,
             'ellipse threshold aspect ratio':  config['ellipseAspectRatio'],
             'adjacency threshold distance':    thresh_dist,
             'adjacency threshold angle':       config['thresh_theta'], 
             'cluster threshold size':          config['clusterSize'],
             'pow spec peak vs mean factor':    config['powSpec_peak_thresh'],
             'show intermediate images':        config['debug'],
             'save intermediate images':        config['saveImg'],
             'save bounding box':               config['save_BB'],
             'show final image':                config['ResultDisp'],
             'display image scaling':           config['image_scale_percent'],
             'Threshold area factor':           config['Thresh_area_factor']}

df_overall = pd.DataFrame(columns =['Image Name', 'Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)'])

if os.path.isdir(join(projectPath, resultDir)) == False: os.mkdir(join(projectPath, resultDir))
if os.path.isdir(join(projectPath, resultDir, ResultCSVDir)) == False: os.mkdir(join(projectPath, resultDir, ResultCSVDir))
if os.path.isdir(join(projectPath, resultDir, ResultImageDir)) == False: os.mkdir(join(projectPath, resultDir, ResultImageDir))

if parameters['save bounding box'] == 1: 
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
