import numpy as np
import glob
import cv2
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import pathlib as pl

# Define the parameter space
# Replace 'param1', 'param2', ..., 'param15' with actual parameter names
# Define their types (Real, Integer) and bounds accordingly


'''
Config file example:

## Directory paths
data_dir         = "DATA/sampleData/";     # Data directory wrt project path 
base_result_dir  = "Results/temp/";        # Base result directory wrt project path

## Parameters
dspace_nm   = [1.9];                # The D-Spacing value (eg. 1.9, 0.7) at which the algorithm will run.
pix_2_nm    = 78.5;                 # Image resolution, number of pixels per nanometer

blur_iteration          = 15;       # Number of Blur Iteration
Blur_kernel_propCons    = 0.15;     # Proportionality constant of d-spacing (in pixel) for the blur kernel size
closing_k_size          = 15;       # Closing Kernel Size
opening_k_size          = 17;       # Opening Kernel Size
pixThresh_propCons      = 0.625;    # Proportionality constant of d-spacing (in pixel) for the threshold number of pixels consituting Backbone
ellipse_len_propCons    = 1.5;      # Proportionality constant of d-spacing (in pixel) for the breaking Backbone into uniform size before constructing ellipse
ellipse_aspect_ratio    = 5;        # Threshold ellipse aspect Ratio 
thresh_dist_propCons    = 2;        # Proportionality constant of d-spacing (in pixel) for the distance threshold for adjacency matrix
thresh_theta            = 10;       # delta Theta threshold for adjacency matrix
cluster_size            = 7;        # Threshold ellipse in Crystal cluster
dspace_bandpass         = 0.2;      # Bandpass filter range across d-spacing
powSpec_peak_thresh     = 1.15;     # 1.20 works for all
Thresh_area_factor      = 4;        # Cut off area factor of d-spacing^2

## Modes
debug               = 1;            # To run on single image and save intermediate steps
save_BB             = 1;            # To save Bounding box coordinates: 1, Not to: 0
save_backbone_coords= 1;            # To save backbone coordinates: 1, Not to: 0
result_display      = 0;            # To display final result in notebook: 1, Not to: 0
image_scale_percent = 50;           # Scaling the image before display

'''
# Constants
ACCEPTED_FORMATS = ['.tif', '.tiff', '.png']

param_space = [
    Real(0.0, 1.0, name='param1'),      # Example: A real-valued parameter between 0 and 1
    Integer(1, 10, name='param2'),      # Example: An integer parameter between 1 and 10
    Real(0.1, 5.0, name='param3'),
    Integer(0, 100, name='param4'),
    Real(0.0, 1.0, name='param5'),
    Real(0.0, 10.0, name='param6'),
    Integer(1, 20, name='param7'),
    Real(0.0, 100.0, name='param8'),
    Real(0.0, 1.0, name='param9'),
    Integer(0, 50, name='param10'),
    Real(0.0, 5.0, name='param11'),
    Integer(1, 15, name='param12'),
    Real(0.0, 1.0, name='param13'),
    Real(0.0, 10.0, name='param14'),
    Integer(1, 100, name='param15')
]

def extract_and_fill_annotations(image, output_path, threshold_value=10):
    """
    Extracts colored annotations (crystal outlines) from an RGB TEM image with a grayscale background,
    fills the inside of the annotations, and outputs a binary image where the annotations are black (filled)
    and the background is white.

    Parameters:
    - image_path (str): Path to the input annotated TEM image.
    - output_path (str): Path to save the output binary image.
    - threshold_value (int): Threshold value for detecting non-grayscale pixels.

    Returns:
    - binary_filled (numpy.ndarray): The binary image array with filled annotations as black and background as white.
    """
    # Load the RGB image
    # image = cv2.imread(image_path)

    # Check if image was loaded successfully
    if image is None:
        raise ValueError(f"Error: Could not read the image.")

    # Convert image to float32 for precision in calculations
    image_float = image.astype(np.float32)

    # Split the image into B, G, R channels
    B, G, R = cv2.split(image_float)

    # Calculate the absolute differences between the channels
    diff_RG = cv2.absdiff(R, G)
    diff_RB = cv2.absdiff(R, B)
    diff_GB = cv2.absdiff(G, B)

    # Combine the differences to get the maximum difference per pixel
    max_diff = cv2.max(diff_RG, cv2.max(diff_RB, diff_GB))

    # Apply threshold to detect non-grayscale pixels (annotations)
    _, binary_image = cv2.threshold(max_diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert binary_image to uint8 type
    binary_image = binary_image.astype(np.uint8)

    # Invert the binary image so that annotations are white (255) and background is black (0)
    # binary_inverted = cv2.bitwise_not(binary_image)

    # Find contours in the binary image
    _, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask of the same size as the image, initialized to white (255)
    mask = np.ones_like(binary_image) * 255  # White background

    # Fill the contours on the mask
    cv2.drawContours(mask, contours, -1, color=0, thickness=cv2.FILLED)
    
    mask = cv2.bitwise_not(mask)

    # Save the filled binary image
    cv2.imwrite(str(output_path), mask)

    return mask

# Placeholder functions for loading data
def load_images(image_folder):
    image_files = glob.glob(f"{image_folder}/*.png")  # Adjust extension if needed
    images = [cv2.imread(f) for f in image_files]
    return images

def load_annotations(annotation_folder):
    annotation_files = glob.glob(f"{annotation_folder}/*.png")
    annotations = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in annotation_files]
    return annotations

# Define the objective function to minimize (negative performance metric)
@use_named_args(param_space)
def objective(**params):
    """
    Objective function for Bayesian Optimization.
    Evaluates the algorithm's performance on the annotated images.
    """
    # Extract parameter values
    # For example: param1 = params['param1'], param2 = params['param2'], etc.
    # Use these parameters in your algorithm

    # Placeholder for algorithm execution
    # Replace this with actual code to run your algorithm
    def run_algorithm(images, params):
        results = []
        # for image in images:
            # Apply preprocessing if needed
            # processed_image = preprocess_image(image, params)
            # Detect crystals
            # detected_crystals = detect_crystals(processed_image, params)
            # results.append(detected_crystals)
        return results

    results = run_algorithm(images, params)

    def compute_iou(detected, ground_truth):
        # Ensure binary images
        detected_binary = (detected > 0).astype(np.uint8)
        ground_truth_binary = (ground_truth > 0).astype(np.uint8)
        intersection = np.logical_and(detected_binary, ground_truth_binary)
        union = np.logical_or(detected_binary, ground_truth_binary)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    
    
    def evaluate_performance(results, annotations):
        scores = []
        for detected, ground_truth in zip(results, annotations):
            # Compute IoU or another metric
            iou_score = compute_iou(detected, ground_truth)
            scores.append(iou_score)
        return np.mean(scores)

    score = evaluate_performance(results, annotations)

    # Since gp_minimize minimizes the objective, return the negative score
    return -score

def CreateMaskFromAnnotatedImagesInsideDir(inputDirPath, outputDirPath, threshold_value=10):
    annotatedImagesPath = [file_path for file_path in inputDirPath.iterdir() 
                   if file_path.is_file() and file_path.suffix in ACCEPTED_FORMATS]
    
    for i, annotatedImagePath in enumerate(annotatedImagesPath):
        annotatedImg = cv2.imread(str(annotatedImagePath))
        maskImagePath = outputDirPath / annotatedImagePath.name
        extract_and_fill_annotations(annotatedImg, maskImagePath, threshold_value)

if __name__ == "__main__":
# def bayesianOpt(image_folder, annotation_folder):
    projectDirPath      = pl.Path(__file__).parent.resolve()
    inputDirPath        = projectDirPath / 'DATA/BO/input/'
    groundTruthDirPath  = projectDirPath /  'DATA/BO/groundTruth/'
    
    # createAnnotations('Data/BO/ground_truth_images', 'Data/BO/annotations', threshold_value=10)
    CreateMaskFromAnnotatedImagesInsideDir(groundTruthDirPath/'images', groundTruthDirPath/'mask', threshold_value=10)
    
    # images = load_images('/media/dgamdha/data/Dhruv/ISU/PhD/Projects/GRATE/GRATE_for_PennState/DATA/BO/results/version_1/Images/')
    # annotations = load_annotations('Data/BO/annotations')
    
    # # image_path = '/media/dgamdha/data/Dhruv/ISU/PhD/Projects/GRATE/GRATE_for_PennState/DATA/BO/results/version_3/Images/FoilHole_21830223_Data_21829764_21829765_20200122_1019.png'
    
    # for i, image in enumerate(images):
    #     # cv2.imwrite(f"Data/BO/annotations/annotation_{i}.png", image)
    #     output_path = f"/media/dgamdha/data/Dhruv/ISU/PhD/Projects/GRATE/GRATE_for_PennState/DATA/BO/results/temp/binary_annotations_filled_{i}.png" 
    #     extract_and_fill_annotations(images[i], output_path, threshold_value=10)
    
    # # Run Bayesian Optimization
    # res = gp_minimize(
    #     func=objective,
    #     dimensions=param_space,
    #     acq_func='EI',      # Expected Improvement
    #     n_calls=50,         # Number of evaluations of the objective function
    #     n_initial_points=10,  # Number of initial random evaluations
    #     random_state=42     # For reproducibility
    # )

    # # Print the best found parameters and the corresponding score
    # print("Best parameters found:")
    # for name, value in zip([dim.name for dim in param_space], res.x):
    #     print(f"{name}: {value}")

    # print(f"Best objective value: {-res.fun}")

    # # Plot convergence
    # plot_convergence(res)
    # plt.show()
