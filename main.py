import sys
import time
import pandas as pd
import io
import libconf
from shutil import copy2
from pathlib import Path
from utils import createVersionDirectory, CreateDirectories
from grate import GRATE

'''
Command Line Arguments:
sys.argv[1] : .cfg file name present inside the configFiles directory.  
'''

# Constants
ACCEPTED_FORMATS = ['.tif', '.tiff', '.png']

def load_config(config_file_path):
    """Load the configuration file."""
    
    with config_file_path.open() as f:
        return libconf.load(f)
    
def calculate_pixel_size(value, factor):
    return int(value * factor)
    
def prepare_parameters(config, project_path, result_dir):
    """Prepare parameters for processing."""
    
    dspace_pix = calculate_pixel_size(config['dspace_nm'], config['pix2nm'])
    
    return {
        'd space nm':                      config['dspace_nm'],
        'd space pix':                     dspace_pix, 
        'pix to nm':                       config['pix2nm'],
        'blur iterations':                 config['blur_iteration'], 
        'blur k size':                     calculate_pixel_size(config['Blur_kernel_propCons'], dspace_pix),
        'closing k size':                  config['closing_k_size'],
        'opening k size':                  config['opening_k_size'], 
        'backbone threshold length':       calculate_pixel_size(config['pixThresh_propCons'], dspace_pix),
        'ellipse pixel size':              calculate_pixel_size(config['ellipse_len_propCons'], dspace_pix),
        'ellipse threshold aspect ratio':  config['ellipseAspectRatio'],
        'adjacency threshold distance':    calculate_pixel_size(config['thresh_dist_propCons'], dspace_pix),
        'adjacency threshold angle':       config['thresh_theta'], 
        'cluster threshold size':          config['clusterSize'],
        'pow spec peak vs mean factor':    config['powSpec_peak_thresh'],
        'debug':                           config['debug'],
        'save bounding box':               config['save_BB'],
        'save backbone coords':            config['save_backbone_coords'],
        'show final image':                config['ResultDisp'],
        'display image scaling':           config['image_scale_percent'],
        'Threshold area factor':           config['Thresh_area_factor'],
        'Project path':                    project_path,
        'result directory'     :           result_dir, 
        'result image directory':          result_dir / "Images",
        'result CSV directory':            result_dir / "CSV",
        'result annotation directory':     result_dir / "Annotations",
        'result backbone coords':          result_dir / "BackboneCoord",
        'Data directory':                  config['dataDir'],
        'Base result directory':           config['BaseResultDir']
    }
    
def setup_directories_and_parameters(project_path, config):
    """Setup directories and prepare parameters."""
    
    base_result_dir = createVersionDirectory(project_path, config['BaseResultDir'], 'version')
    copy2(project_path / 'configFiles' / sys.argv[1], base_result_dir)
    result_dir = base_result_dir / str(config['dspace_nm'])
    parameters = prepare_parameters(config, project_path, result_dir)
    CreateDirectories(parameters)
    
    data_dir = project_path / parameters['Data directory']
    
    return parameters, result_dir, data_dir

def process_image(file_path, parameters):
    """Process a single image."""
    
    print("Img Name: ", file_path.name, "\n")
    
    parameters['img path'] = file_path
    start_time = time.time()
    df_crystal_props = GRATE(file_path, parameters)
    print("Overall GRATE Time:", round(time.time() - start_time, 2), "\n")
    
    return df_crystal_props

def process_images(data_dir, parameters):
    """Process each image in the specified directory."""
    
    df_overall = pd.DataFrame(columns=['Image Name', 'Centroid', 'Crystal Area (nm^2)', 
                                       'Crystal Angle (zero at X-axis and clockwise positive)', 
                                       'D-Spacing(FFT, nm)'])
    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ACCEPTED_FORMATS:
            df_crystal_props = process_image(file_path, parameters)
            df_overall = df_overall.append(df_crystal_props, ignore_index=True)
    
    return df_overall

def main():
    
    project_path = Path(__file__).parent.resolve()
    config = load_config(project_path / 'configFiles' / sys.argv[1])
    print("\nd space:", config['dspace_nm'])

    parameters, result_dir, data_dir = setup_directories_and_parameters(project_path, config)
    
    df_overall = process_images(data_dir, parameters)
    df_overall.to_csv(result_dir / 'overall.csv')
    
if __name__ == "__main__":
    main()
