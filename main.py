import sys
import pandas as pd
import libconf
from pathlib import Path
from utils import createVersionDirectory, CreateDirectories, calculate_pixel_size, timeit, pick_unique_colors
from grate import ImageProcessor
from concurrent.futures import ProcessPoolExecutor
import random

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
    
def prepare_parameters(config, project_path, version_result_dir, dspace_nm, crys_color):
    """Prepare parameters for processing."""
    
    dspace_pix = calculate_pixel_size(dspace_nm, config['pix_2_nm'])
    
    data_dir    = project_path / str(config['data_dir'])
    
    resolution_params = {
        'd space nm'        : dspace_nm,
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
        'd space bandpass'              : config['dspace_bandpass'],
        'pow spec peak vs mean factor'  : config['powSpec_peak_thresh'],
        'threshold area factor'         : config['thresh_area_factor'],
        'crystal color'                 : crys_color,
        'alpha shape factor'            : config['alpha_shape_factor'],
    }

    filesystem_params = {
        'Project path'          : project_path,
        'result directory'      : version_result_dir,
        'result image directory': version_result_dir / "Images",
        'result CSV directory'  : version_result_dir / "CSV",
        'result annotation directory': version_result_dir / "Annotations",
        'result backbone coords': version_result_dir / "BackboneCoord",
        'Data directory'        : data_dir,
        'Mask directory'        : version_result_dir / 'Masks',
        # 'Base result directory' : version_result_dir,
        'save image format'     : '.png'
    }

    miscellaneous_params = {
        'debug': config['debug'],
        'save bounding box'     : config['save_BB'],
        'save backbone coords'  : config['save_backbone_coords'],
        'show final image'      : config['result_display'],
        'display image scaling' : config['image_scale_percent'],
        'bayesian opt run'      : config['bayesian_opt_run'],
    }

    parameters = {**resolution_params, **image_processing_params, **filesystem_params, **miscellaneous_params}
    
    directories = [
        parameters['result directory'],
        parameters['result CSV directory'],
        parameters['result image directory'],
        parameters['result backbone coords'] if parameters['save backbone coords'] == 1 else None,
        parameters['result annotation directory'] if parameters['save bounding box'] == 1 else None,
        parameters['Mask directory']
    ]
    
    return parameters, data_dir, directories
    
def setup_directories_and_parameters(project_path, config, dspace_nm, crys_color):
    """Setup directories and prepare parameters."""
    
    version_result_dir = createVersionDirectory(project_path / str(config['base_result_dir']), 
                                                'version')
    
    parameters, data_dir, directories = prepare_parameters(config, 
                                                           project_path, 
                                                           version_result_dir, 
                                                           dspace_nm, 
                                                           crys_color)
    
    CreateDirectories(directories)
    
    with open(version_result_dir / 'config.cfg', 'w') as config_file:
        libconf.dump(config, config_file)
        
    return parameters, data_dir

def run_image_processor(file_path, parameters, last_run):
    try:
        print("Processing image:", file_path.name)
        processor = ImageProcessor(file_path, parameters, last_run)
        df_crystal_props = processor.GRATE()
        return df_crystal_props
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame or appropriate error indicator
            
def write_to_overallCSV(result_dir, dspace_nm, df_overall):
    
    csvFileName = f"overall_dspace_{dspace_nm}.csv"
    
    overall_csv_path = result_dir / csvFileName
    if overall_csv_path.exists():
        print("Appending to overall.csv")
        df_overall.to_csv(overall_csv_path, mode='a', header=False, index=False)
    else:
        print("Creating overall.csv")
        df_overall.to_csv(overall_csv_path, mode='w', index=False)
        
def write_to_color(result_dir, dspace_nm, crystal_color):
    
    color_path = result_dir / 'color.txt'
    if color_path.exists():
        print("Appending to readme.txt")
        with open(color_path, 'a') as readme_file:
            readme_file.write(f"\n\nD-Space: {dspace_nm}\nCrystal Color: {crystal_color}")
    else:
        print("Creating readme.txt")
        with open(color_path, 'w') as readme_file:
            readme_file.write(f"D-Space: {dspace_nm}\nCrystal Color: {crystal_color}")
            
def process_image(data_dir, parameters, last_run, run_parallel=True):
    
    df_overall = pd.DataFrame(columns=[ 'Image Name', 
                                        'Centroid', 
                                        'Crystal Area (nm^2)', 
                                        'Crystal Angle (zero at X-axis and clockwise positive)', 
                                        'D-Spacing(FFT, nm)' 
                                        'crystalMajorAxis_length (nm)', 
                                        'crystalMinorAxis_length (nm)', 
                                        'MajorAxisAngle', 
                                        'angleDifference'])
    
    image_files = [file_path for file_path in data_dir.iterdir() 
                   if file_path.is_file() and file_path.suffix in ACCEPTED_FORMATS]
    
    if run_parallel:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(run_image_processor, 
                                        image_files, 
                                        [parameters] * len(image_files), 
                                        [last_run] * len(image_files)))
            executor.shutdown( wait=True )

        for result in results:
            df_overall = pd.concat([df_overall, result], ignore_index=True)
    else:
        for file_path in image_files:
            df_crystal_props = run_image_processor(file_path, 
                                                   parameters, 
                                                   last_run)
            df_overall = pd.concat([df_overall, df_crystal_props], 
                                   ignore_index=True)
    
    df_overall = df_overall.round(2)
        
    return df_overall
            
@timeit
def main():
    
    config, project_path = load_config()
    
    first_run = True
    last_run = False
    dspace_nm_list = config['dspace_nm']
    print("d space list:", dspace_nm_list)
    
    crys_colors = pick_unique_colors(len(dspace_nm_list))
    
    for i, dspace_nm in enumerate(dspace_nm_list):
        print("\nd space:", dspace_nm)
        
        if first_run:
            parameters, data_dir = setup_directories_and_parameters(project_path, 
                                                                    config, 
                                                                    dspace_nm, 
                                                                    crys_colors[i])
            first_run = False
        else:
            parameters, data_dir = prepare_parameters(config, 
                                                      project_path, 
                                                      parameters['result directory'], 
                                                      dspace_nm, 
                                                      crys_colors[i])
            
        if dspace_nm == dspace_nm_list[-1]:
            last_run = True
            
        df_overall = process_image(data_dir, 
                                   parameters, 
                                   last_run, 
                                   run_parallel=True)
        
        write_to_overallCSV(parameters['result directory'], 
                            parameters['d space nm'], 
                            df_overall)
        write_to_color(parameters['result directory'], 
                       parameters['d space nm'], 
                       parameters['crystal color'])
    
if __name__ == "__main__":
    main()
