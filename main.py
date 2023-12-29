import sys
import time
import pandas as pd
import io
import libconf
from shutil import copy2
from pathlib import Path
from utils import createVersionDirectory, CreateDirectories
from grate import ImageProcessor
from concurrent.futures import ProcessPoolExecutor

'''
Command Line Arguments:
sys.argv[1] : .cfg file name present inside the configFiles directory.  
sys.argv[2] : (Optional) dspace_nm value. If not provided, then the value from the .cfg file is used.
'''

# Constants
ACCEPTED_FORMATS = ['.tif', '.tiff', '.png']

def load_config():
    """Load the configuration file."""
    
    project_path = Path(__file__).parent.resolve()
    config_file_path = project_path / 'configFiles' / sys.argv[1]
    
    config = {}
    
    with config_file_path.open() as f:
        config = libconf.load(f)
    
    if len(sys.argv) > 2:
        try:
            float(sys.argv[2])
            print("Overriding dspace_nm with command line argument.")
            config['dspace_nm'] = float(sys.argv[2])
        except ValueError:
            print("Invalid dspace_nm argument. Please provide a float number.")
            sys.exit(1)
        
    return config, project_path
    
def calculate_pixel_size(value, factor):
    return int(value * factor)
    
def prepare_parameters(config, project_path, base_result_dir):
    """Prepare parameters for processing."""
    
    dspace_pix = calculate_pixel_size(config['dspace_nm'], config['pix_2_nm'])
    
    result_dir  = base_result_dir / str(config['dspace_nm'])
    data_dir    = project_path / str(config['data_dir'])
    
    resolution_params = {
        'd space nm'        : config['dspace_nm'],
        'd space pix'       : dspace_pix,
        'pix to nm'         : config['pix_2_nm'],
    }
    
    image_processing_params = {
        'blur iterations'   : config['blur_iteration'],
        'blur k size'       : calculate_pixel_size(config['Blur_kernel_propCons'], dspace_pix),
        'closing k size'    : config['closing_k_size'],
        'opening k size'    : config['opening_k_size'],
        'backbone threshold length'     : calculate_pixel_size(config['pixThresh_propCons'], dspace_pix),
        'ellipse pixel size'            : calculate_pixel_size(config['ellipse_len_propCons'], dspace_pix),
        'ellipse threshold aspect ratio': config['ellipse_aspect_ratio'],
        'adjacency threshold distance'  : calculate_pixel_size(config['thresh_dist_propCons'], dspace_pix),
        'adjacency threshold angle'     : config['thresh_theta'],
        'cluster threshold size'        : config['cluster_size'],
        'pow spec peak vs mean factor'  : config['powSpec_peak_thresh'],
        'Threshold area factor'         : config['Thresh_area_factor'],
    }

    filesystem_params = {
        'Project path'          : project_path,
        'result directory'      : result_dir,
        'result image directory': result_dir / "Images",
        'result CSV directory'  : result_dir / "CSV",
        'result annotation directory': result_dir / "Annotations",
        'result backbone coords': result_dir / "BackboneCoord",
        'Data directory'        : data_dir,
        'Base result directory' : base_result_dir,
        'save image format'     : '.png'
    }

    miscellaneous_params = {
        'debug': config['debug'],
        'save bounding box'     : config['save_BB'],
        'save backbone coords'  : config['save_backbone_coords'],
        'show final image'      : config['result_display'],
        'display image scaling' : config['image_scale_percent']
    }

    all_params = {**resolution_params, **image_processing_params, **filesystem_params, **miscellaneous_params}
    
    return all_params, result_dir, data_dir
    
def setup_directories_and_parameters(project_path, config):
    """Setup directories and prepare parameters."""
    
    base_result_dir = createVersionDirectory(project_path / str(config['base_result_dir']), 'version')

    with open(base_result_dir / 'config.cfg', 'w') as config_file:
        libconf.dump(config, config_file)
    
    parameters, result_dir, data_dir = prepare_parameters(config, project_path, base_result_dir)
    CreateDirectories(parameters)
    
    return parameters, result_dir, data_dir

def process_image(file_path, parameters):
    try:
        print("Processing image:", file_path.name)
        processor = ImageProcessor(file_path, parameters)
        df_crystal_props = processor.GRATE()
        return df_crystal_props
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame or appropriate error indicator

def process_images(data_dir, parameters):
    """Process each image in the specified directory."""
    
    df_overall = pd.DataFrame(columns=['Image Name', 'Centroid', 'Crystal Area (nm^2)', 
                                       'Crystal Angle (zero at X-axis and clockwise positive)', 
                                       'D-Spacing(FFT, nm)'])
    image_files = [file_path for file_path in data_dir.iterdir() 
                   if file_path.is_file() and file_path.suffix in ACCEPTED_FORMATS]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_image, image_files, [parameters] * len(image_files)))

    for result in results:
        df_overall = df_overall.append(result, ignore_index=True)
    
    return df_overall

def main():
    
    config, project_path = load_config()
    
    print("\nd space:", config['dspace_nm'])

    parameters, result_dir, data_dir = setup_directories_and_parameters(project_path, config)
    
    df_overall = process_images(data_dir, parameters)
    df_overall.to_csv(result_dir / 'overall.csv')
    
if __name__ == "__main__":
    main()
