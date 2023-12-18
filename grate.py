from utils import *
from ops import *
import time
from skimage import io

def print_time(process_name, start_time):
    total = time.time() - start_time
    print(f"{process_name} Time: {round(total, 2)}")
    
def process_image(img, parameters):
    timeCode = parameters.get('timeCode', 0)
    start_time = time.time()
    result = BlurThresh(img, parameters)
    if timeCode:
        print_time("Blurring and Thresholding", start_time)
    return result

def process_skeleton(thresh, parameters):
    timeCode = parameters.get('timeCode', 0)
    start_time = time.time()

    if parameters['d space pix'] < 78:
        result = Skeletonize(thresh, parameters)
    else:
        closing = Closing(thresh, parameters)
        opening = Opening(closing, parameters)
        result = Skeletonize(opening, parameters)

    if timeCode:
        print_time("Closing Opening and Skeletonization", start_time)

    return result

def process_break_branches(skeleton, parameters):
    timeCode = parameters.get('timeCode', 0)
    start_time = time.time()
    result = BreakBranches(skeleton, parameters)
    if timeCode:
        print_time("Breaking Branches", start_time)

    return result

def process_skeleton_segmentation(skeleton, parameters):
    timeCode = parameters.get('timeCode', 0)
    start_time = time.time()
    temp = SkeletonSegmentation(skeleton)
    if timeCode:
        print_time("Segmentation", start_time)
    return temp

def process_filtered_uniform_bb(img, temp, parameters):
    timeCode = parameters.get('timeCode', 0)
    start_time = time.time()
    Broken_backbone_img = Filtered_Uniform_BB(img, temp, parameters)
    if timeCode:
        print_time("Uniform BB", start_time)
    return Broken_backbone_img

def process_ellipse_construction(Broken_backbone_img, parameters):
    timeCode = parameters.get('timeCode', 0)
    start_time = time.time()
    bb_ellipse1, bb_ellipse_props = EllipseConstruction(Broken_backbone_img, parameters)
    if timeCode:
        print_time("Ellipse Construction", start_time)
    return bb_ellipse1, bb_ellipse_props

def GRATE(img_path, parameters):
    # img = cv2.imread(join(imgName),0)
    img = io.imread(img_path)
    img = img.astype('float64')
    timeCode = 0
    
    thresh = process_image(img, parameters)
    
    skeleton = process_skeleton(thresh, parameters)
    
    skeleton = process_break_branches(skeleton, parameters)
    
    temp = process_skeleton_segmentation(skeleton, parameters)
    
    Broken_backbone_img = process_filtered_uniform_bb(img, temp, parameters)
    
    bb_ellipse1, bb_ellipse_props = process_ellipse_construction(Broken_backbone_img, parameters)
    
    # t0 = time.time()
    # thresh = BlurThresh(img, parameters)
    # if timeCode == 1:
    #     t1 = time.time()
    #     total = t1-t0
    #     print("Blurring and Thresholding Time           :", round(total,2))

    # # t0 = time.time()
    # if parameters['d space pix'] < 78:
    #     skeleton = Skeletonize(thresh, parameters)
    # else:
    #     closing = Closing(thresh, parameters)
    #     opening = Opening(closing, parameters)
    #     skeleton = Skeletonize(opening, parameters)
    
    # if timeCode == 1:
    #     t0 = time.time()
    #     total = t0-t1
    #     print("Closing Opening and Skeletonization Time :", round(total,2))

    # t0 = time.time()
    # skeleton = BreakBranches(skeleton, parameters)
    # if timeCode == 1:
    #     t1 = time.time()
    #     total = t1-t0
    #     print("Breaking Banches Time                    :", round(total,2))
    
    # t0 = time.time()
    # # _, temp = SkeletonSegmentation(skeleton)
    # temp = SkeletonSegmentation(skeleton)
    # if timeCode == 1:
    #     t1 = time.time()
    #     total = t1-t0
    #     print("Segmentation Time                        :", round(total,2))

    # Broken_backbone_img = Filtered_Uniform_BB(img, temp, parameters)
    # if timeCode == 1:
    #     t0 = time.time()
    #     total = t0-t1
    #     print("Uniform BB Time                          :", round(total,2))

    ######################### MODIFY Filtered_Uniform_BB TO REMOVE RESIDUAL ALSO ################################
#     _, filteredPolys = SkeletonSegmentation(Broken_backbone_img_temp)
#     Broken_backbone_img = Remove_small_BB(Broken_backbone_img_temp, filteredPolys, int(0.5*ellipse_len))
    
#     _, filteredPolys1 = SkeletonSegmentation(Broken_backbone_img)
    
#     restructuredFP = RemovingDim(filteredPolys1)
    
    # t0 = time.time()
    # bb_ellipse1, bb_ellipse_props = EllipseConstruction(Broken_backbone_img, parameters)
    # if timeCode == 1:
    #     t1 = time.time()
    #     total = t1-t0
    #     print("Ellipse Construction Time                :", round(total,2))


    # t0 = time.time()
    adjacencyMat = AdjacencyMat(Broken_backbone_img, bb_ellipse_props, parameters)
    if timeCode == 1:
        t0 = time.time()
        total = t0-t1
        print("Adjacency Matrix Time                    :", round(total,2))

    # t0 = time.time()
    ellipseCluster, AllClusterPointCloud, crystalAngles = ConnecComp(Broken_backbone_img, adjacencyMat, bb_ellipse_props, parameters, img_path)
    if timeCode == 1:
        t1 = time.time()
        total = t1-t0
        print("Connected Component Time                 :", round(total,2))

    # t0 = time.time()
    crystalArea, centroid, crystalAngles_final, dspaces, df_boundBox, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference = PlottingAndSaving(img, AllClusterPointCloud, img_path, crystalAngles, parameters)
    if timeCode == 1:
        t0 = time.time()
        total = t0-t1
        print("Eval D-Spacing, Plotting and Saving Time :", round(total,2))


    if len(centroid) == 0:
        imgNamelist = [img_path.name]
    else:
        imgNamelist = [None]*len(centroid)
        imgNamelist[0] = img_path.name
    
    df = pd.DataFrame(list(zip(imgNamelist, centroid, crystalArea, crystalAngles_final, dspaces, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference)), columns=['Image Name','Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)', 'crystalMajorAxis_length (nm)', 'crystalMinorAxis_length (nm)', 'MajorAxisAngle', 'angleDifference'])
    df = df.round(2)

#     print("Saving results to: ",join(projectPath, resultDir,imgName[:-4]+'.csv'))
    df.to_csv(join(parameters['result CSV directory'], img_path.stem+'.csv'))
    
    if parameters['save bounding box'] == 1: 
#         print("Saving BB Annotations to: ",join(projectPath, annotationDir, imgName[:-4]+'.csv'))
        df_BB = pd.DataFrame( list( zip( df_boundBox ) ) , columns = [ "Top Left(x_y) Bottom Right(x_y)" ] )
        df_BB.to_csv(join(parameters['result annotation directory'], img_path.stem+'.csv'))
    
    return df 