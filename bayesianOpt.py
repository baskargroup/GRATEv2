import numpy as np
import glob
import cv2
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
import matplotlib.pyplot as plt
import pathlib as pl
import libconf
import subprocess

# Constants
ACCEPTED_FORMATS = ['.tif', '.tiff', '.png']

param_space = [
    Integer(5   ,   20  ,   name='blur_iteration'       ,   dtype= int),
    Real(   0.1 ,   0.5 ,   name='Blur_kernel_propCons' ,   dtype= float),
    Integer(1   ,   20  ,   name='closing_k_size'       ,   dtype= int),
    Integer(1   ,   20  ,   name='opening_k_size'       ,   dtype= int),
    Real(   0.0 ,   1.0 ,   name='pixThresh_propCons'   ,   dtype= float),
    Real(   0.5 ,   5.0 ,   name='ellipse_len_propCons' ,   dtype= float),
    Real(   2.0 ,   7.0 ,   name='ellipse_aspect_ratio' ,   dtype= float),
    Real(   1.0 ,   5.0 ,   name='thresh_dist_propCons' ,   dtype= float),
    Real(   5.0 ,   15.0,   name='thresh_theta'         ,   dtype= float),
    Integer(1   ,   10  ,   name='cluster_size'         ,   dtype= int),
    Real(   0.1 ,   0.5 ,   name='dspace_bandpass'      ,   dtype= float),
    Real(   1.0 ,   1.5 ,   name='powSpec_peak_thresh'  ,   dtype= float),
    Real(   1.0 ,   5.0 ,   name='thresh_area_factor'   ,   dtype= float),
    ]

pathsDict = {
    'projectDirFPath'       : pl.Path(__file__).parent.resolve(),
    'inputImgDirRPath'      : 'DATA/BO/input/',
    'grateOutputDirRPath'   : 'DATA/BO/BOTraining/',
    'groundTruthDirRPath'   : 'DATA/BO/groundTruth/',
    'detectionDirName'      : 'Images',
    'masksDirName'          : 'Masks',
    'grateRunDirTemplate'   : 'version_{}',
    "latestRunDirIndex"     : 0,
    'configFileRPath'       : 'configFiles/BO.cfg',
    }

def createConfigFile(configFilePath, 
                     configDict):
    configDict['data_dir']          = pathsDict['inputImgDirRPath']
    configDict['base_result_dir']   = pathsDict['grateOutputDirRPath']
    
    configDict['dspace_nm'] = [1.9]
    configDict['pix_2_nm']  = 78.5
    
    configDict['debug']                 = 0
    configDict['save_BB']               = 0
    configDict['save_backbone_coords']  = 0
    configDict['result_display']        = 0
    configDict['image_scale_percent']   = 50
    configDict['bayesian_opt_run']      = True
    configDict['alpha_shape_factor']    = 0.002
    
    with open(configFilePath, 'w') as configFile:
        libconf.dump(configDict, 
                     configFile)
        configFile.close()

def extract_and_fill_annotations(image, 
                                 output_path, 
                                 threshold_value=10):
    """
    Extracts colored annotations (crystal outlines) from an RGB TEM image with a grayscale background,
    fills the inside of the annotations, and outputs a binary image where the annotations are white (filled)
    and the background is black.

    Parameters:
    - image (numpy.ndarray): The input RGB TEM image array with crystal annotations.
    - output_path (str): Path to save the output binary image.
    - threshold_value (int): Threshold value for detecting non-grayscale pixels.

    Returns:
    - binary_filled (numpy.ndarray): The binary image array with filled annotations as white and background as black.
    """

    if image is None:
        raise ValueError(f"Error: Could not read the image.")

    image_float = image.astype(np.float32)

    B, G, R = cv2.split(image_float)

    diff_RG = cv2.absdiff(R, G)
    diff_RB = cv2.absdiff(R, B)
    diff_GB = cv2.absdiff(G, B)

    max_diff = cv2.max(diff_RG, cv2.max(diff_RB, diff_GB))

    _, binary_image = cv2.threshold(max_diff, threshold_value, 255, cv2.THRESH_BINARY)

    binary_image = binary_image.astype(np.uint8)

    _, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask of the same size as the image, initialized to white (255)
    mask = np.ones_like(binary_image) * 255  # White background

    # Fill the contours on the mask
    cv2.drawContours(mask, contours, -1, color=0, thickness=cv2.FILLED)
    
    # Invert the binary image so that annotations are white (255) and background is black (0)    
    mask = cv2.bitwise_not(mask)

    # Save the filled binary image
    cv2.imwrite(str(output_path), mask)

    return mask

def CreateMaskFromAnnotatedImagesInsideDir(inputDirPath, 
                                           outputDirPath, 
                                           threshold_value=10):
    annotatedImagesPath = [file_path for file_path in inputDirPath.iterdir() 
                   if file_path.is_file() and file_path.suffix in ACCEPTED_FORMATS]
    
    for i, annotatedImagePath in enumerate(annotatedImagesPath):
        annotatedImg = cv2.imread(str(annotatedImagePath))
        maskImagePath = outputDirPath / annotatedImagePath.name
        extract_and_fill_annotations(annotatedImg, maskImagePath, threshold_value)

# Placeholder functions for loading data
def load_images(image_folder):
    image_files = glob.glob(f"{image_folder}/*.png")  # Adjust extension if needed
    images = [cv2.imread(f) for f in image_files]
    return images

def load_annotations(annotation_folder):
    annotation_files = glob.glob(f"{annotation_folder}/*.png")
    annotations = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in annotation_files]
    return annotations

def updateTemplateIndex(baseDirPathObj, 
                        versionDirTemplate, 
                        versionIndex):
    if versionIndex <= 0:
        versionIndex = 0
        while True:
            versionIndex += 1
            if not (baseDirPathObj / versionDirTemplate.format(versionIndex)).exists():
                break
    return versionIndex - 1

# # Previous Loss function used for run_3
# def compute_iou(detectedDirPath, groundTruthDirPath):
    
#     imagesNames = [file_path.name for file_path in detectedDirPath.iterdir() 
#                     if file_path.is_file() and file_path.suffix in '.png']
    
#     IoU = []
#     for imageName in imagesNames:
#         detected = cv2.imread(str(detectedDirPath / imageName), cv2.IMREAD_GRAYSCALE)
#         ground_truth = cv2.imread(str(groundTruthDirPath / imageName), cv2.IMREAD_GRAYSCALE)
#         intersection = np.logical_and(detected, ground_truth)
#         union = np.logical_or(detected, ground_truth)
#         iou = np.sum(intersection) / np.sum(union)
#         IoU.append(iou)
    
#     return np.mean(IoU)

def compute_iou(detectedDir_fpath, 
                groundTruthDir_fpath):
    imagesNames = [file_path.name for file_path in detectedDir_fpath.iterdir() 
                    if file_path.is_file() and file_path.suffix in '.png']
    
    IoU = []
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
        
        IoU.append(iou)
    
    if len(IoU) == 0:
        print("Warning: No valid IoU values computed. Returning zero.")
        return 0.0  # or handle this case as needed
    
    return np.mean(IoU)


@use_named_args(param_space)
def objective(**params):
    """
    Objective function for Bayesian Optimization.
    Evaluates the algorithm's performance on the annotated images.
    """
    
    def run_algorithm():
        command = ['python', 'main.py', 'BO.cfg']
        subprocess.run(command)
    
    createConfigFile(pathsDict['projectDirFPath'] / pathsDict['configFileRPath'], 
                     params)
    run_algorithm()
    
    latestRunDirIndex = updateTemplateIndex(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'], 
                                            pathsDict['grateRunDirTemplate'], 
                                            0)
    
    # Create Masks for the latest run
    CreateMaskFromAnnotatedImagesInsideDir(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / pathsDict['grateRunDirTemplate'].format(latestRunDirIndex) / pathsDict['detectionDirName'], 
                                           pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / pathsDict['grateRunDirTemplate'].format(latestRunDirIndex) / pathsDict['masksDirName'], 
                                           threshold_value = 10)
    
    score = compute_iou(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / pathsDict['grateRunDirTemplate'].format(latestRunDirIndex) / pathsDict['masksDirName'], 
                        pathsDict['projectDirFPath'] / pathsDict['groundTruthDirRPath'] / pathsDict['masksDirName'])
    
    return -score

if __name__ == "__main__":
    # # Prepare the gound truth masks
    # CreateMaskFromAnnotatedImagesInsideDir(pathsDict['projectDirFPath'] / pathsDict['groundTruthDirRPath'] / pathsDict['detectionDirName'], 
    # pathsDict['projectDirFPath'] / pathsDict['groundTruthDirRPath'] / pathsDict['masksDirName'], 
    # threshold_value=10)
    
    # Save checkpoints using callbacks
    checkpoint_callback = skopt.callbacks.CheckpointSaver(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / 'checkpoint.pkl')
    
    # Run Bayesian Optimization
    res = gp_minimize(
        func=objective,
        dimensions=param_space,
        acq_func='EI',      # Expected Improvement
        n_calls=200,         # Number of evaluations of the objective function
        n_initial_points=10,# Number of initial random evaluations
        random_state=42,     # For reproducibility
        callback=[checkpoint_callback]
    )

    # Print the best found parameters and the corresponding score
    print("Best parameters found:")
    for name, value in zip([dim.name for dim in param_space], res.x):
        print(f"{name}: {value}")

    print(f"Best objective value: {-res.fun}")
    
    # Store the results and convergence plot
    with open(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / 'results.txt', 'w') as f:
        f.write(f"Best parameters found:\n")
        f.write(f"Best objective value: {-res.fun}\n")
        best_eval = res.func_vals.argmin() + 1
        f.write(f"Best evaluation number: {best_eval}\n")
    
    with open(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / 'convergence.csv', 'w') as f:
        f.write(f"Evaluations, Objective Value, Min Objective Value\n")
        min_val = res.func_vals[0]
        for i, val in enumerate(res.func_vals):
            min_val = min(min_val, val)
            f.write(f"{i+1}, {val}, {min_val}\n")
            
    plot_convergence(res)
    plt.savefig(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / 'convergence_plot.png')
    
    # Save the best parameters to a config file
    best_params = dict(zip([dim.name for dim in param_space], res.x))
    createConfigFile(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / 'best_params.cfg', best_params)
    
    plot_evaluations(res)
    plt.savefig(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / 'evaluations_plot.png')
    
    plot_objective(res)
    plt.savefig(pathsDict['projectDirFPath'] / pathsDict['grateOutputDirRPath'] / 'objective_plot.png')
    