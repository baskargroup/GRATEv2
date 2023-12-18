from numpy import copy
from utils import *
from ops import *
from grate import *

import sys
import os
import re
from os import listdir
from os.path import isfile
import io, libconf
from shutil import copy2

'''
Command Line Arguments:
sys.argv[1] : .cfg file name present inside the configFiles directory.  
'''

projectPath     = os.path.dirname(os.path.abspath(__file__))

with io.open(join(projectPath,'configFiles', sys.argv[1])) as f:
    config = libconf.load(f)

dataDir             = config['dataDir']
BaseResultDir       = config['BaseResultDir']

ResultImageDir      = "Images";         # Directory name storing the output images, created inside the BaseResultDir/version_#/<dspace_nm>/
ResultCSVDir        = "CSV";            # Directory name storing the CSV output, created inside the BaseResultDir/version_#/<dspace_nm>/
ResultAnnotationDir = "Annotations";    # Directory name storing the annotations, created inside the BaseResultDir/version_#/<dspace_nm>/
ResultBackboneCoordDir = "BackboneCoord";    # Directory name storing the BackboneCoord, created inside the BaseResultDir/version_#/<dspace_nm>/

dspace_pix              = int(config['dspace_nm']*config['pix2nm'])

BaseResultDir   = createVersionDirectory(projectPath, BaseResultDir, 'version')
copy2(join(projectPath,'configFiles', sys.argv[1]), join(projectPath, BaseResultDir)) # Copying the .cfg file to the result directory.
resultDir       = join(BaseResultDir, str(config['dspace_nm']))

parameters = {'d space nm':                     config['dspace_nm'],
             'd space pix':                     dspace_pix, 
             'pix to nm':                       config['pix2nm'],
             'blur iterations':                 config['blur_iteration'], 
             'blur k size':                     int( config['Blur_kernel_propCons'] * dspace_pix ),
             'closing k size':                  config['closing_k_size'],
             'opening k size':                  config['opening_k_size'], 
             'backbone threshold length':       int( config['pixThresh_propCons']   * dspace_pix ), 
             'ellipse pixel size':              int( config['ellipse_len_propCons'] * dspace_pix ),
             'ellipse threshold aspect ratio':  config['ellipseAspectRatio'],
             'adjacency threshold distance':    int( config['thresh_dist_propCons'] * dspace_pix ),
             'adjacency threshold angle':       config['thresh_theta'], 
             'cluster threshold size':          config['clusterSize'],
             'pow spec peak vs mean factor':    config['powSpec_peak_thresh'],
             'debug':                           config['debug'],
             'save bounding box':               config['save_BB'],
             'save backbone coords':            config['save_backbone_coords'],
             'show final image':                config['ResultDisp'],
             'display image scaling':           config['image_scale_percent'],
             'Threshold area factor':           config['Thresh_area_factor'],
             'Project path':                    projectPath,
             'result directory'     :           resultDir, 
             'result image directory':          ResultImageDir,
             'result CSV directory':            ResultCSVDir,
             'result annotation directory':     ResultAnnotationDir,
             'result backbone coords':          ResultBackboneCoordDir,
             'Data directory':                  config['dataDir'],
             'Base result directory':           config['BaseResultDir']}

CreateDirectories(parameters)
print("\nd space:", config['dspace_nm'])

df_overall = pd.DataFrame(columns =['Image Name', 'Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)'])
onlyfiles = [f for f in listdir(join(projectPath,dataDir)) if isfile(join(projectPath,dataDir,f))]

# if parameters['debug'] == 1: 
#     onlyfiles = onlyfiles[0:1]
#     print("DEBUG MODE ON")

for f in onlyfiles:
    acceptedFormats = ['.tif', '.tiff', '.png']
    
    if f.endswith(tuple(acceptedFormats)):
    # if f[-5:] == ".tiff":
        print("Img Name: ", f, "\n")
        parameters['img name'] = f
        t0 = time.time()    
        df_crystalProps = GRATE(f, parameters)
        t1 = time.time()
        total = t1-t0
        print("Overall GRATE Time:", round(total,2), "\n")
        df_overall = df_overall.append(df_crystalProps, ignore_index=True,)

df_overall.to_csv(join(projectPath, resultDir, ResultCSVDir, 'overall.csv'))
