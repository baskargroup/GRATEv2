from utils import *
from ops import *
import time

def GRATE(projectPath, dataDir, imgName, resultDir, annotationDir, parameters):
    img = cv2.imread(join(projectPath, dataDir, imgName),0)
    
    timeCode = 1
    
    t0 = time.time()
    thresh = BlurThresh(img, parameters)
    if timeCode == 1:
        t1 = time.time()
        total = t1-t0
        print("Blurring and Thresholding Time           :", round(total,2))

    # t0 = time.time()
    if parameters['d space pix'] < 78:
        skeleton = Skeletonize(thresh, parameters)
    else:
        closing = Closing(thresh, parameters)
        opening = Opening(closing, parameters)
        skeleton = Skeletonize(opening, parameters)
    
    if timeCode == 1:
        t0 = time.time()
        total = t0-t1
        print("Closing Opening and Skeletonization Time :", round(total,2))

    # t0 = time.time()
    skeleton = BreakBraches(skeleton, parameters)
    if timeCode == 1:
        t1 = time.time()
        total = t1-t0
        print("Breaking Banches Time                    :", round(total,2))
    
    t0 = time.time()
    # _, temp = SkeletonSegmentation(skeleton)
    temp = SkeletonSegmentation(skeleton)
    if timeCode == 1:
        t1 = time.time()
        total = t1-t0
        print("Segmentation Time                        :", round(total,2))

    Broken_backbone_img = Filtered_Uniform_BB(img, temp, parameters)
    if timeCode == 1:
        t0 = time.time()
        total = t0-t1
        print("Uniform BB Time                          :", round(total,2))

    ######################### MODIFY Filtered_Uniform_BB TO REMOVE RESIDUAL ALSO ################################
#     _, filteredPolys = SkeletonSegmentation(Broken_backbone_img_temp)
#     Broken_backbone_img = Remove_small_BB(Broken_backbone_img_temp, filteredPolys, int(0.5*ellipse_len))
    
#     _, filteredPolys1 = SkeletonSegmentation(Broken_backbone_img)
    
#     restructuredFP = RemovingDim(filteredPolys1)
    
    # t0 = time.time()
    bb_ellipse1, bb_ellipse_props = EllipseConstruction(Broken_backbone_img, parameters)
    if timeCode == 1:
        t1 = time.time()
        total = t1-t0
        print("Ellipse Construction Time                :", round(total,2))


    # t0 = time.time()
    adjacencyMat = AdjacencyMat(Broken_backbone_img, bb_ellipse_props, parameters)
    if timeCode == 1:
        t0 = time.time()
        total = t0-t1
        print("Adjacency Matrix Time                    :", round(total,2))

    # t0 = time.time()
    ellipseCluster, AllClusterPointCloud, crystalAngles = ConnecComp(Broken_backbone_img, adjacencyMat, bb_ellipse_props, parameters)
    if timeCode == 1:
        t1 = time.time()
        total = t1-t0
        print("Connected Component Time                 :", round(total,2))

    # t0 = time.time()
    crystalArea, centroid, crystalAngles_final, dspaces, df_boundBox = PlottingAndSaving(img, AllClusterPointCloud, projectPath, resultDir, imgName, crystalAngles, parameters)
    if timeCode == 1:
        t0 = time.time()
        total = t0-t1
        print("Eval D-Spacing, Plotting and Saving Time :", round(total,2))


    if len(centroid) == 0:
        imgNamelist = [imgName]
    else:
        imgNamelist = [None]*len(centroid)
        imgNamelist[0] = imgName
    # df = pd.DataFrame(list(zip(imgNamelist, centroid, crystalArea, crystalAngles, ds)), columns=['Image Name','Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)'])
    df = pd.DataFrame(list(zip(imgNamelist, centroid, crystalArea, crystalAngles_final, dspaces)), columns=['Image Name','Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)'])
    df = df.round(2)

#     print("Saving results to: ",join(projectPath, resultDir,imgName[:-4]+'.csv'))
    df.to_csv(join(projectPath, resultDir,imgName[:-4]+'.csv'))
    
    if parameters['save bounding box'] == 1: 
#         print("Saving BB Annotations to: ",join(projectPath, annotationDir, imgName[:-4]+'.csv'))
        df_BB = pd.DataFrame( list( zip( df_boundBox ) ) , columns = [ "Top Left(x_y) Bottom Right(x_y)" ] )
        df_BB.to_csv(join(projectPath, annotationDir, imgName[:-4]+'.csv'))
    
    return df 