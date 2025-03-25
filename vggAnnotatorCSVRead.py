import csv
import json
import cv2
import numpy as np
import pathlib as pl

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
                            input_image_dir_fpath, 
                            save_image_dir_fpath):
    currentFilename = ''
    lastFilename = ''
    
    for annotation in annotations:    
        currentFilename = annotation['filename']
        if lastFilename != currentFilename:
            img = cv2.imread(str(input_image_dir_fpath / currentFilename))
        else:
            img = cv2.imread(str(save_image_dir_fpath / currentFilename))
        
        polygon = annotation['region_attributes']['polygon']
        for i in range(len(polygon)):
            cv2.line(img, polygon[i], polygon[(i+1)%len(polygon)], (0, 255, 0), 5)
        cv2.imwrite(str(save_image_dir_fpath / currentFilename), img)
        lastFilename = currentFilename
    

def plot_VGG_annotations_on_image(base_dir_fpath, 
                                  annotations_csv_fname):
    
    annotations = read_via_annotations(base_dir_fpath / 'groundTruth' / annotations_csv_fname)
    
    save_image_dir_fpath = base_dir_fpath / 'groundTruth' / 'Images'
    save_image_dir_fpath.mkdir(parents=True, exist_ok=True)
    
    save_image_with_polygon(annotations,
                            base_dir_fpath / 'input', 
                            save_image_dir_fpath)

if __name__ == '__main__':
    
    project_dir_fpath = pl.Path(__file__).parent.resolve()
    base_dir_rpath = 'Example/validation'   # should have input and groundTruth directories inside
    annotation_csv_fname = 'annotation.csv'
    
    # base dir should have the following structure:
    #|--base_dir/
    #   |--groundTruth/
    #       |--annotation_csv_fname
    #   |--input/
    #       |--img1.png
    #       |--img2.png
    #       |--...
    
    # check base directory
    if not (project_dir_fpath / base_dir_rpath).exists():
        print('Error: Directory not found:', base_dir_rpath)
        exit()
        
    # check annotation file
    if not (project_dir_fpath / base_dir_rpath / 'groundTruth' / annotation_csv_fname).exists():
        print('Error: Annotation file not found:', annotation_csv_fname)
        exit()
        
    # check input image directory
    if not (project_dir_fpath / base_dir_rpath / 'input').exists():
        print('Error: Directory not found: input')
        exit()
    
    plot_VGG_annotations_on_image(project_dir_fpath / base_dir_rpath, 
                                  annotation_csv_fname) 
    print('Done!')
