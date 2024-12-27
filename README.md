# GRATEv2: Automated HRTEM Image Analysis Framework 
[**Paper Link**](https://arxiv.org/abs/2411.03474)

**GRATEv2** (GRaph-based Analysis of TEM) is an open-source computational framework designed for automated analysis and detection of crystalline structures in High-Resolution Transmission Electron Microscopy (HRTEM) images. This tool helps material scientists, chemists, and engineers study the microstructure of conjugated polymers and other materials at the nanoscale.



## Key Features

1. **Image Processing-Based Analysis**  
   - **Parameterized Pipeline**: `main.py` provides a configurable image processing pipeline that detects and characterizes crystals in HRTEM images.  
   - **Configurable Parameters**: Users control algorithm behavior via a config file (default location: `configFiles/`). Parameters include morphological kernel sizes, threshold values, d-spacing references, etc.

2. **Bayesian Optimization**  
   - **Automated Parameter Tuning**: `bayesianOpt.py` leverages Bayesian optimization to find optimal parameters, minimizing manual tuning efforts.  
   - **IoU-Based or Custom Loss**: By default, uses Intersection over Union (IoU) to evaluate performance. Users can extend the loss function to incorporate orientation angles, centroid alignment, etc.

3. **Comprehensive Structural Feature Extraction**  
   - **Crystal Properties**: Extracts crystal attributes such as d-spacing, orientation angles, aspect ratios, and intercrystalline correlations.  
   - **Graph-Based Detection**: Once skeleton-based features are extracted, the pipeline constructs a graph to cluster connected skeletal segments into distinct crystals.

4. **Batch-Based Incremental Analysis**  
   - **Iterative Updates**: Integrates Bayesian optimization with iterative parameter updates, using chosen metrics (e.g., IoU) for detection performance.  
   - **Wasserstein Distance Thresholds**: When analyzing data in batches, different batch sizes can be set, affecting how quickly the distribution converges.


## Repository Structure

- **`main.py`**  
    - **Purpose**: Primary script for image processing analysis.  
    - **Usage**: Requires a config file (e.g., `BO_200Evals.cfg`) specifying input paths, output paths, and algorithm parameters.  
    - **Notes**: Processes HRTEM images and outputs detected crystals along with computed features.  

- **`bayesianOpt.py`**  
    - **Purpose**: Performs Bayesian optimization to identify optimal parameter sets for `main.py`.  
    - **Paths**: Users should update path variables inside `bayesianOpt.py` (e.g., `inputImgDirRPath`, `grateOutputDirRPath`, `groundTruthDirRPath`) before running.  
    - **Loss/Metric**: By default uses IoU, but can be changed in the `objective()` function if a more sophisticated metric is desired.  
    - **Iteration Controls**: Modify `n_calls` and `n_initial_points` to set the number of optimization steps and initial random evaluations.

- **`requirements.txt`**  
    - **Purpose**: Lists Python dependencies for creating a reproducible environment.  
    - **Usage**: `pip install -r requirements.txt`

- **`vggAnnotatorCSVRead.py`**  
    - **Purpose**: Converts VGG Image Annotator CSV outputs into binary masks for ground truth.  
    - **Usage**: Adjust path variables (e.g., `base_dir_rpath`, `annotation_csv_fname`) to point to your annotation data.  
    - **Result**: Produces masks used in evaluating detection performance or training Bayesian optimization.

- **Configuration Files** (e.g., `BO_200Evals.cfg`, `manual.cfg`)  
    - **Purpose**: Specify algorithm parameters and paths.  
    - **Usage**: 
        - `data_dir` and `base_result_dir`: For input and output directories.  
        - Algorithm parameters (e.g., morphological settings, d-spacing references).  
        - Optional modes (`debug`, `save_BB`, `result_display`) to control additional outputs.  
    - **Recommendation**: Rename or create new config files to differentiate between manual selection and Bayesian-optimized parameters (e.g., `manual.cfg`, `BO_200Evals.cfg`).


## Installation

1. **Clone the repository**  
   *(Update the following commands if the repository URL or local path changes.)*
   ```bash
   git clone https://<YourUsername>@bitbucket.org/baskargroup/gratev2.git GRATEv2
   cd GRATEv2
   ```

2.	Create and activate a virtual environment
    ```bash
    python3 -m venv gratev2_venv
    source gratev2_venv/bin/activate
    ```

3.	Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
## Usage

1. Running the Image Processing Algorithm
    1.	Update Config File
        -	Choose a config file inside the configFiles directory (e.g., BO_200Evals.cfg).
        -	Edit data_dir (path to input images) and base_result_dir (path to store output) to point to your data.
        -	If you have Bayesian-optimized parameters, you can keep them in the config file; otherwise, you can manually adjust them.
    2.	Run Analysis
          ```bash
          python main.py BO_200Evals.cfg
          ```
        -	The script processes images found in data_dir, detects crystals, and saves results under base_result_dir.

2. Performing Bayesian Optimization
    1.	Adjust bayesianOpt.py
        -	Update path variables for inputImgDirRPath, grateOutputDirRPath, groundTruthDirRPath.
        -	Modify n_calls (total optimization steps) and n_initial_points (random initial exploration) in the gp_minimize function if needed.
    2.	Run Bayesian Optimization
          ```bash
          python bayesianOpt.py
          ```

        -	Tests various parameter sets, calling main.py for each set.
        -	Saves best parameter sets, convergence plots, and results in the specified output directory.

    3.	(Optional) Custom Loss
        -	In bayesianOpt.py, the objective() function uses an IoU-based metric by default.
        -	Users can extend or replace it with orientation-based or centroid-based measures, especially if crystals exhibit complex overlaps or require finer detection fidelity.

3. Preparing Ground Truth Annotations
    1.	Annotation Tool
        -	Use the VGG Image Annotator to label crystals in HRTEM images.
        -	Save the resulting CSV in the relevant directory.
    2.	Convert CSV Annotations
          ```bash
          python vggAnnotatorCSVRead.py
          ```
        -	Modifies the CSV into binary masks for evaluating detection performance.
        -	Update base_dir_rpath and annotation_csv_fname in vggAnnotatorCSVRead.py to point to your data.

    3.	Validation and Testing
        -	Place generated masks in the correct groundTruth location.
        -	Use bayesianOpt.py or manual evaluation to measure performance on annotated data.

## Thresholds and Batch Sizes in Bayesian Optimization
-	When using Bayesian optimization with incremental dataset additions, the batch size influences how frequently and significantly the Wasserstein distance (or other metrics) changes.
-	Larger batch sizes may produce more pronounced distribution shifts, potentially requiring a higher threshold, while smaller batch sizes update incrementally and might justify a lower threshold for stopping criteria.

## Customization
1.	Loss Function
	-	Extend or replace the default IoU metric in the objective() function of bayesianOpt.py.
	-	Incorporate orientation angles, area mismatch, centroid differences, etc., for more sophisticated tasks.
2.	Modes and Parameters
	-	In the config file, debug, save_BB, result_display, etc., can be toggled to produce additional outputs or intermediate debug data.

## Contributing

We welcome contributions, such as issues, pull requests, or feature suggestions. Please open an issue to discuss proposed changes before submitting a PR.

## License

This project is licensed under the [MIT LICENSE](LICENSE), making it freely available and modifiable.

## Citation

If you use GRATEv2 in your research, please cite our work:
 ```bibtex
@article{gamdha2024computational,
title={GRATEv2: Computational Tools for Real-time Analysis of High-throughput High-resolution TEM (HRTEM) Images of Conjugated Polymers},
author={Gamdha, Dhruv and Fair, Ryan and Krishnamurthy, Adarsh and Gomez, Enrique and Ganapathysubramanian, Baskar},
journal={arXiv preprint arXiv:2411.03474},
year={2024}
}
```

GRATEv2 aims to provide a robust, user-friendly framework for advanced HRTEM analysis, fostering deeper insights into crystalline microstructures. Please reach out with any questions or suggestions. Happy analyzing!

