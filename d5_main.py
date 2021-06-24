from utils import *
from ops import *
from grate import *

import sys
import os
from os import listdir
from os.path import isfile

# projectPath = '/media/dhruv/data/Dhruv/ISU/PhD/Projects/GRATE/GRATE_for_PennState/'       # Local
projectPath = '/work/adarsh/Dhruv/GRATE/'                                                    # NOVA 
dataDir = 'DATA/GridSquare_21826475/'
resultDir = 'Results/delta5/'

AnnotationDir = 'Annotations/'
onlyfiles = [f for f in listdir(projectPath+dataDir) if isfile(join(projectPath+dataDir, f))]

resultDir = join(resultDir, str(sys.argv[2]),str(sys.argv[1]))
if os.path.isdir(join(projectPath, resultDir)) == False:
    os.mkdir(join(projectPath, resultDir))

dspace_nm = float(sys.argv[1]) # 1.9nm, 7A == 0.7nm, 4A == 0.4nm
pix2nm = 78.5
print("\n d space:", dspace_nm)
dspace_pix = int(dspace_nm*pix2nm)
# print("dspace_pix:", dspace_pix)

blur_iteration          = 15                    # Best = 4 prev Best = 200
dspace_frac_Blur_kernel = int(0.15*dspace_pix)  # Best = 0.5, prev Best 0.09, Fraction of dspace for the blur kernel size 
closing_k_size          = 15                    # Kernel Size
opening_k_size          = 17                    # Kernel Size
pixThresh               = int(0.625*dspace_pix) # 1.25*dspace_pix,Threshold number of pixels consituting polymers
ellipse_len             = int(1.5*dspace_pix)   # Old = 160,Breaking polymer into this size before constructing ellipse 
ellipseAspectRatio      = 5                     # Threshold aspect Ratio of the ellipse
thresh_dist             = int(2*dspace_pix)     # Old = int(1.35*dspace_pix), Distance threshold for adjacency matrix 
thresh_theta            = 10                    # delta Theta threshold for adjacency matrix 
clusterSize             = 7                    # old = 10, Threshold Crystal cluster size 


debug = 0                               # To print images: 1, Not to:0
saveImg = 0                             # To intermediate step images: 1, Not to:0
save_BB = 0                             # To save Bounding box coordinates: 1, Not to: 0
ResultDisp = 0                          # To display final result in notebook: 1, Not to:0
image_scale_percent = 50                # Scaling the image before display

if sys.argv[2] == "blur_iteration":
    domain = [14,15,16]
elif sys.argv[2] == "dspace_frac_Blur_kernel":
    domain = [0.142, 0.15, 0.158]
elif sys.argv[2] == "closing_k_size":
    domain = [14,15,17]
elif sys.argv[2] == "opening_k_size":
    domain = [16,17,18]
elif sys.argv[2] == "pixThresh":
    domain = [0.594,0.625,0.656]
elif sys.argv[2] == "ellipse_len":
    domain = [1.42,1.5,1.58]
elif sys.argv[2] == "ellipseAspectRatio":
    domain = [4.75,5,5.25]
elif sys.argv[2] == "thresh_dist":
    domain = [1.9,2,2.1]
elif sys.argv[2] == "thresh_theta":
    domain = [9.5,10,10.5]
elif sys.argv[2] == "clusterSize":
    domain = [6,7,8]


for val in domain: 
    # blur_iteration = val

    if sys.argv[2] == "blur_iteration":
        blur_iteration = val
    elif sys.argv[2] == "dspace_frac_Blur_kernel":
        dspace_frac_Blur_kernel = int(val*dspace_pix)
    elif sys.argv[2] == "closing_k_size":
        closing_k_size = val
    elif sys.argv[2] == "opening_k_size":
        opening_k_size = val
    elif sys.argv[2] == "pixThresh":
        pixThresh = int(val*dspace_pix)
    elif sys.argv[2] == "ellipse_len":
        ellipse_len = int(val*dspace_pix)
    elif sys.argv[2] == "ellipseAspectRatio":
        ellipseAspectRatio = val
    elif sys.argv[2] == "thresh_dist":
        thresh_dist = int(val*dspace_pix)
    elif sys.argv[2] == "thresh_theta":
        thresh_theta = val
    elif sys.argv[2] == "clusterSize":
        clusterSize = val

    print("Value:", val)
    valDir = join(resultDir, str(round(val,3)))
    if os.path.isdir(join(projectPath, valDir)) == False:
        os.mkdir(join(projectPath, valDir))
    parameters = {'d space pix':                    dspace_pix,
                 'pix to nm':                       pix2nm, 
                 'blur iterations':                 blur_iteration, 
                 'blur k size':                     dspace_frac_Blur_kernel,
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
        if f[-4:] == ".tif":#and f == "FoilHole_21832497_Data_21829764_21829765_20200122_1417.tif":
            print("Img Name: ", f, "\n")
            # print("Full Img Path: ",join(projectPath,dataDir,f))    
            df_crystalProps = GRATE(projectPath, dataDir, f, valDir, AnnotationDir, parameters)
            df_overall = df_overall.append(df_crystalProps, ignore_index=True,)

    df_overall.to_csv(join(projectPath, valDir,'overall.csv'))
    """ df = df_overall 
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df[['Crystal Area (pixel^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, pixels)']])

    plt.scatter(principalComponents[:,0],principalComponents[:,1] )
    plt.savefig(join(projectPath, valDir,'pca_'+str(val) +'.png'))
    # plt.show()
    plt.close()
    plt.clf() """
    print("=====================================================================================")