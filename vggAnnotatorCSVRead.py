import csv
import json
import cv2
import numpy as np
import pathlib as pl
from bayesianOpt import CreateMaskFromAnnotatedImagesInsideDir

def compute_iou(detectedDir_fpath, groundTruthDir_fpath):
    imagesNames = [file_path.name for file_path in detectedDir_fpath.iterdir() 
                    if file_path.is_file() and file_path.suffix in '.png']
    
    IoU = {}
    for imageName in imagesNames:
        detected = cv2.imread(str(detectedDir_fpath / imageName), cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(str(groundTruthDir_fpath / imageName), cv2.IMREAD_GRAYSCALE)
        
        # Ensure that images are read correctly
        if detected is None or ground_truth is None:
            print(f"Warning: Could not read images {imageName}. Skipping.")
            continue
        
        # Convert images to binary (if necessary)
        _, detected_bin = cv2.threshold(detected, 127, 1, cv2.THRESH_BINARY)
        _, ground_truth_bin = cv2.threshold(ground_truth, 127, 1, cv2.THRESH_BINARY)
        
        intersection = np.logical_and(detected_bin, ground_truth_bin)
        union = np.logical_or(detected_bin, ground_truth_bin)
        
        if np.sum(union) == 0:
            print(f"Warning: Union of detected and ground truth is zero for {imageName}. Skipping.")
            continue
        
        iou = np.sum(intersection) / np.sum(union)
        
        IoU[imageName] = iou
    
    if len(IoU) == 0:
        print("Warning: No valid IoU values computed. Returning zero.")
        return 0.0  # or handle this case as needed
    
    return IoU

def read_via_annotations(csv_file_path):
    """
    Reads annotations from a VIA (VGG Image Annotator) CSV file and returns a list of annotations.

    Args:
        csv_file_path (str): Path to the VIA annotations CSV file.

    Returns:
        List[dict]: A list of dictionaries containing annotation data.
    """
    annotations = []
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse the JSON strings in 'region_shape_attributes' and 'region_attributes'
            shape_attrs = json.loads(row['region_shape_attributes']) if row['region_shape_attributes'] else {}
            
            # Store the x, y coordinates of the polygon in a list of tuples
            region_attrs = {}
            if shape_attrs['name'] == 'polygon':
                x_coords = shape_attrs['all_points_x']
                y_coords = shape_attrs['all_points_y']
                region_attrs['polygon'] = list(zip(x_coords, y_coords))
            
            # For annotation, just store the filename and the polygon coordinates
            annotation = {
                'filename': row['filename'],
                'region_attributes': region_attrs
            }
            
            annotations.append(annotation)
    return annotations

def save_image_with_polygon(annotations, 
                            project_dir_fpath, 
                            input_image_dir_rpath, 
                            save_image_dir_rpath):
    currentFilename = ''
    lastFilename = ''
    for annotation in annotations:
        
        currentFilename = annotation['filename']
        if lastFilename != currentFilename:
            img = cv2.imread(str(project_dir_fpath / input_image_dir_rpath / currentFilename))
        else:
            img = cv2.imread(str(project_dir_fpath / save_image_dir_rpath / currentFilename))
        
        polygon = annotation['region_attributes']['polygon']
        for i in range(len(polygon)):
            cv2.line(img, polygon[i], polygon[(i+1)%len(polygon)], (0, 255, 0), 5)
        cv2.imwrite(str(project_dir_fpath / save_image_dir_rpath / currentFilename), img)
        lastFilename = currentFilename
    
def write_iou_to_file(iou, 
                      fpath, 
                      grateOutput_Masks_dir_fpath, 
                      groundTruth_Masks_dir_fpath):
    with open (fpath, 'w') as f:
        f.write('IOU\n')
        f.write('Ground Truth Masks Directory   : ' + str(groundTruth_Masks_dir_fpath) + '\n')
        f.write('GRATE Masks Directory          : ' + str(grateOutput_Masks_dir_fpath) + '\n')
        
        for key, value in iou.items():
            f.write(key + ' : ' + str(value) + '\n')
        
        f.write('\n')
        f.write('Avg IOU: ' + str(np.mean(list(iou.values()))))
        f.write('\n')
        
def plot_VGG_annotations_on_image(project_dir_fpath, 
                                  annotations_csv_rpath, 
                                  input_image_dir_rpath, 
                                  save_image_dir_rpath):
    
    annotations = read_via_annotations(project_dir_fpath / annotations_csv_rpath)
    save_image_with_polygon(annotations, project_dir_fpath, input_image_dir_rpath, save_image_dir_rpath)

def create_gt_BO_manual_masks(  project_dir_fpath,
                                groundTruth_dir_rpath,
                                BO_grateOutput_dir_rpath,
                                manual_grateOutput_dir_rpath):
    CreateMaskFromAnnotatedImagesInsideDir(project_dir_fpath / groundTruth_dir_rpath / 'Images', 
                                           project_dir_fpath / groundTruth_dir_rpath / 'Masks')
    CreateMaskFromAnnotatedImagesInsideDir(project_dir_fpath / BO_grateOutput_dir_rpath / 'Images',
                                           project_dir_fpath / BO_grateOutput_dir_rpath / 'Masks')
    CreateMaskFromAnnotatedImagesInsideDir(project_dir_fpath / manual_grateOutput_dir_rpath / 'Images',
                                           project_dir_fpath / manual_grateOutput_dir_rpath / 'Masks')

def compute_ioU_and_write_to_file(project_dir_fpath, 
                                  groundTruth_dir_rpath, 
                                  BO_grateOutput_dir_rpath, 
                                  manual_grateOutput_dir_rpath):
    # calculate BO and manual IOU
    print('Calculating BO IOU...')
    BO_iou = compute_iou(project_dir_fpath / BO_grateOutput_dir_rpath / 'Masks', 
                         project_dir_fpath / groundTruth_dir_rpath / 'Masks')
    
    print('Calculating Manual IOU...')
    manual_iou = compute_iou(project_dir_fpath / manual_grateOutput_dir_rpath / 'Masks', 
                             project_dir_fpath / groundTruth_dir_rpath / 'Masks')
    
    # Write IOU to file
    write_iou_to_file(  BO_iou, 
                        project_dir_fpath / BO_grateOutput_dir_rpath / 'iou_log.txt', 
                        project_dir_fpath / BO_grateOutput_dir_rpath / 'Masks',
                        project_dir_fpath / groundTruth_dir_rpath / 'Masks')
    write_iou_to_file(  manual_iou, 
                        project_dir_fpath / manual_grateOutput_dir_rpath / 'iou_log.txt',
                        project_dir_fpath / manual_grateOutput_dir_rpath / 'Masks',
                        project_dir_fpath / groundTruth_dir_rpath / 'Masks')

if __name__ == '__main__':
    
    project_dir_fpath = pl.Path(__file__).parent.resolve()
    VGG_annotations_csv_rpath   = 'DATA/BO/validation/groundTruth/csv/via_project_28Nov2024_20h54m_csv.csv'
    unannotated_image_dir_rpath = 'DATA/BO/validation/input/png'
    annotated_image_dir_rpath   = 'DATA/BO/validation/groundTruth/images'
    
    # Below directories should containing the "Images" and "Masks" directories
    groundTruth_dir_rpath           = 'DATA/BO/validation/groundTruth'
    BO_grateOutput_dir_rpath        = 'DATA/BO/validation/output/BO_para/version_2_subset_soloPlot'
    manual_grateOutput_dir_rpath    = 'DATA/BO/validation/output/manual_para/version_2_subset_soloPlot'
    
    plot_VGG_annotations_on_image(project_dir_fpath, VGG_annotations_csv_rpath, unannotated_image_dir_rpath, annotated_image_dir_rpath)
    
    # Create Masks for groundTruth, BO and manual
    create_gt_BO_manual_masks(  project_dir_fpath,
                                groundTruth_dir_rpath,
                                BO_grateOutput_dir_rpath,
                                manual_grateOutput_dir_rpath)
    
    compute_ioU_and_write_to_file(project_dir_fpath, 
                                  groundTruth_dir_rpath, 
                                  BO_grateOutput_dir_rpath, 
                                  manual_grateOutput_dir_rpath)
    
    print('Done!')
