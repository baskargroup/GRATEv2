from utils import *
from ops import *
import time
from skimage import io

    
def process_image(img, parameters):
    result = BlurThresh(img, parameters)
    return result

def process_skeleton(thresh, parameters):
    if parameters['d space pix'] < 78:
        result = Skeletonize(thresh, parameters)
    else:
        closing = Closing(thresh, parameters)
        opening = Opening(closing, parameters)
        result = Skeletonize(opening, parameters)
        
    return result

def process_break_branches(skeleton, parameters):
    result = BreakBranches(skeleton, parameters)
    
    return result

def process_skeleton_segmentation(skeleton, parameters):
    temp = SkeletonSegmentation(skeleton)
    return temp

def process_filtered_uniform_bb(img, temp, parameters):
    Broken_backbone_img = Filtered_Uniform_BB(img, temp, parameters)
    return Broken_backbone_img

def process_ellipse_construction(Broken_backbone_img, parameters):
    bb_ellipse1, bb_ellipse_props = EllipseConstruction(Broken_backbone_img, parameters)
    return bb_ellipse1, bb_ellipse_props

def process_adjacency_mat(Broken_backbone_img, bb_ellipse_props, parameters):
    adjacencyMat = AdjacencyMat(Broken_backbone_img, bb_ellipse_props, parameters)
    return adjacencyMat

def process_connec_comp(Broken_backbone_img, adjacencyMat, bb_ellipse_props, parameters, img_path):
    ellipseCluster, AllClusterPointCloud, crystalAngles = ConnecComp(Broken_backbone_img, adjacencyMat, bb_ellipse_props, parameters, img_path)
    return ellipseCluster, AllClusterPointCloud, crystalAngles

def process_plotting_and_saving(img, AllClusterPointCloud, img_path, crystalAngles, parameters):
    crystal_props = PlottingAndSaving(img, AllClusterPointCloud, img_path, crystalAngles, parameters)
    return crystal_props

def process_and_save_dataframe(img_path, parameters, centroid, crystalArea, crystalAngles_final, dspaces, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference):
    
    if len(centroid) == 0:
        imgNamelist = [img_path.name]
    else:
        imgNamelist = [None] * len(centroid)
        imgNamelist[0] = img_path.name

    df = pd.DataFrame(list(zip(imgNamelist, centroid, crystalArea, crystalAngles_final, dspaces, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference)), 
                      columns=['Image Name', 'Centroid', 'Crystal Area (nm^2)', 'Crystal Angle (zero at X-axis and clockwise positive)', 'D-Spacing(FFT, nm)', 'crystalMajorAxis_length (nm)', 'crystalMinorAxis_length (nm)', 'MajorAxisAngle', 'angleDifference'])
    df.round(2)
    
    csv_file_path = Path(parameters['result CSV directory']) / f"{img_path.stem}.csv"
    df.to_csv(csv_file_path)
    
    return df

def process_save_backbone_coords(img_path, parameters, df_boundBox):
    
    if parameters['save bounding box'] == 1:
        df_BB = pd.DataFrame( list( zip( df_boundBox ) ) , columns = [ "Top Left(x_y) Bottom Right(x_y)" ] )
        df_BB.to_csv(join(parameters['result annotation directory'], img_path.stem+'.csv'))
    

@timeit
def GRATE(img_path, parameters):
    # img = cv2.imread(join(imgName),0)
    img = io.imread(img_path)
    img = img.astype('float64')
    
    # Check if the file is already present in the result image directory else convert gray to RGB and save it
    if (parameters['result image directory'] / (img_path.stem+'.png')).is_file():
        print("Image already present in the result image directory")
    else:
        print("Saving image as png to the result image directory")
        cv2.imwrite(str(parameters['result image directory'] / (img_path.stem+'.png')), cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2RGB))
        
    thresh = process_image(img, parameters)
    
    skeleton = process_skeleton(thresh, parameters)
    
    skeleton = process_break_branches(skeleton, parameters)
    
    temp = process_skeleton_segmentation(skeleton, parameters)
    
    Broken_backbone_img = process_filtered_uniform_bb(img, temp, parameters)
    
    bb_ellipse1, bb_ellipse_props = process_ellipse_construction(Broken_backbone_img, parameters)
    
    adjacencyMat = process_adjacency_mat(Broken_backbone_img, bb_ellipse_props, parameters)
    
    ellipseCluster, AllClusterPointCloud, crystalAngles = process_connec_comp(Broken_backbone_img, adjacencyMat, bb_ellipse_props, parameters, img_path)
    
    crystalArea, centroid, crystalAngles_final, dspaces, df_boundBox, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference = process_plotting_and_saving(img, AllClusterPointCloud, img_path, crystalAngles, parameters)
    
    df = process_and_save_dataframe(img_path, parameters, centroid, crystalArea, crystalAngles_final, dspaces, crystalMajorAxis_length, crystalMinorAxis_length, crystalMajorAxisAngle, angleDifference)
    
    process_save_backbone_coords(img_path, parameters, df_boundBox)
    
    return df