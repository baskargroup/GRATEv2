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
            # region_attrs = json.loads(row['region_attributes']) if row['region_attributes'] else {}
            
            # Exaple region_shape_attributes: {"name":"polygon","all_points_x":[4088,3322,2769,2641,3022,3250,3687,4084],"all_points_y":[1050,862,1262,2132,2509,2761,2934,3118]}
            # Store the x, y coordinates of the polygon in a list of tuples
            region_attrs = {}
            if shape_attrs['name'] == 'polygon':
                x_coords = shape_attrs['all_points_x']
                y_coords = shape_attrs['all_points_y']
                region_attrs['polygon'] = list(zip(x_coords, y_coords))
            # elif shape_attrs['name'] == 'rect':
            #     x = shape_attrs['x']
            #     y = shape_attrs['y']
            #     width = shape_attrs['width']
            #     height = shape_attrs['height']
            #     region_attrs['rect'] = (x, y, width, height)
            # elif shape_attrs['name'] == 'circle':
            #     cx = shape_attrs['cx']
            #     cy = shape_attrs['cy']
            #     r = shape_attrs['r']
            #     region_attrs['circle'] = (cx, cy, r)


            # annotation = {
            #     'filename': row['filename'],
            #     'file_size': int(row['file_size']) if row['file_size'] else None,
            #     'file_attributes': json.loads(row['file_attributes']) if 'file_attributes' in row and row['file_attributes'] else {},
            #     'region_count': int(row['region_count']) if 'region_count' in row and row['region_count'] else 0,
            #     'region_id': int(row['region_id']) if 'region_id' in row and row['region_id'] else None,
            #     'region_shape_attributes': shape_attrs,
            #     'region_attributes': region_attrs
            # }
            
            # For annotation, just store the filename and the polygon coordinates
            annotation = {
                'filename': row['filename'],
                'region_attributes': region_attrs
            }
            
            annotations.append(annotation)
    return annotations

if __name__ == '__main__':
    
    project_dir_fpath = pl.Path(__file__).parent.resolve()
    annotations_csv_rpath = 'DATA/BO/validation/groundTruth/csv/via_project_28Nov2024_20h54m_csv.csv'
    input_image_dir_rpath = 'DATA/BO/validation/input/png'
    save_image_dir_rpath = 'DATA/BO/validation/groundTruth/images'

    annotations = read_via_annotations(project_dir_fpath / annotations_csv_rpath)

    # Print file name and polygon coordinates for each annotation
    for annotation in annotations:
        print(annotation['filename'])
        print(annotation['region_attributes']['polygon'])
        print()
        
    # Read the image from annotations filename and save the image with the polygon overlay
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
