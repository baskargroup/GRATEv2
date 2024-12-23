
# GRATEv2: Automated HRTEM Image Analysis Framework {[Paper Link](https://arxiv.org/abs/2411.03474)}

**GRATEv2** (GRaph-based Analysis of TEM) is an open-source computational framework designed for automated analysis and detection of crystalline structures in High-Resolution Transmission Electron Microscopy (HRTEM) images. This tool helps material scientists, chemists, and engineers study the microstructure of conjugated polymers and other materials at the nanoscale.

## Key Features
- **Image Processing-Based Analysis:** 📷  
  `main.py` provides a parameterized image processing pipeline to detect and characterize crystals in HRTEM images based on parameters specified in a configuration file located inside `configFiles` directory.
- **Bayesian Optimization:** 🔍  
  `bayesianOpt.py` leverages Bayesian optimization to find the optimal parameters for the image processing algorithm, reducing manual tuning and ensuring reproducibility.
- **Comprehensive Structural Feature Extraction:** 🔬  
  Extracts crystal properties such as d-spacing, orientation angles, aspect ratios, and intercrystalline correlations.
- **Batch-Based Incremental Analysis:** 🔄  
  Integrates Bayesian optimization with iterative parameter updates, guided by a chosen metric (e.g., Intersection over Union) for quantifying detection performance.

## Repository Structure
- **`main.py`**:  
  The primary script for analysis. Requires a config file to be present inside `configFiles` directory specifying input data directory path, output results directory path, and algorithm parameters. Processes HRTEM images and outputs detected crystals with evaluated features.
  
- **`bayesianOpt.py`**:  
  Performs Bayesian optimization to determine optimal image processing parameters. Interacts with `main.py` to evaluate candidate parameters on a training dataset. Users can adjust paths, iteration counts (`n_calls`, `n_initial_points`), and even the objective function if desired for custom loss metrics.
 💡 **TODO:** Elaborate on the paths to be set by the user.

- **`requirements.txt`**:  
  Lists Python dependencies. Use `pip install -r requirements.txt` to set up a consistent environment.
  
- **`vggAnnotatorCSVRead.py`**:  
  A helper script for converting CSV annotations (created using the VGG Image Annotator) into the required format for training and evaluation of crystal detection performance. This script is used to generate ground truth annotations for training and validation.

- **Configuration Files (e.g., `BO_run3_200Evals.cfg`)**:  
  Specify directories, parameters, and modes. Once optimal parameters are found via Bayesian optimization, users mainly need to adjust `data_dir` and `base_result_dir`. The modes are optional but can enable debugging or other features. 💡 **TODO:** Need to rename the config files to ManualSelection and BO

## Installation
1. **Clone the repository:** 💡 **TODO:** May need to update it after transfering the repository to the BGLab
   ```bash
   git clone https://github.com/YourUsername/GRATEv2.git
   cd GRATEv2
   ```
   
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv gratev2_venv
   source gratev2_venv/bin/activate
   ```
   
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Image Processing Algorithm
1. **Update Config File:**  
   Open `BO_run3_200Evals.cfg` (or another config file) and update `data_dir` and `base_result_dir`. The algorithm parameters are already set to optimal values found via Bayesian optimization after 200 evaluations.

2. **Run the Analysis:**
   ```bash
   python main.py BO_run3_200Evals.cfg
   ```

   The script will look for the config file inside the `configFiles` directory and will process image present in `data_dir`, detect crystals, and save the results, including crystal properties, in the specified `base_result_dir` location.

### Performing Bayesian Optimization
1. **Update `bayesianOpt.py`:**  

   Adjust `inputImgDirRPath`, `grateOutputDirRPath`, and `groundTruthDirRPath`. Modify `n_calls` and `n_initial_points` if needed. Consider changing the loss function if you want to incorporate additional features beyond IoU.

2. **Run the Optimization:**
   ```bash
   python bayesianOpt.py
   ```
  
   The script tests various parameter sets, runs `main.py` to evaluate performance, and converges on optimal parameters. Results (convergence plots, best parameters) are saved in the output directory.

### Preparing Ground Truth Annotations
- **Annotation Tool:** Use the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) to annotate crystals in HRTEM images.
- **Convert CSV Annotations:**
  ```bash
  python vggAnnotatorCSVRead.py
  ```

  This generates appropriate ground truth masks for evaluating detection performance.

## Thresholds and Batch Sizes in Bayesian Optimization
When using Bayesian optimization, data is processed incrementally in batches. The batch size affects how the Wasserstein distance values scale. Larger batches introduce more substantial changes per increment, potentially justifying a slightly higher threshold, while smaller batches produce subtler changes and may warrant a lower threshold. Consider these factors when selecting a stopping criterion.

## Customization
- **Loss Function:**  
  Edit the objective function in `bayesianOpt.py` to integrate additional crystal features into the loss function if desired.
  
- **Modes and Parameters:**
  Adjust modes in the config file (e.g., `debug`, `save_BB`, `result_display`) to control outputs and intermediate debugging steps.
  
## Contributing
We welcome contributions in the form of issue reports, feature requests, or pull requests. Please open an issue to discuss proposed changes before submitting a PR.

## License
(Include your chosen license here, for example: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.)

## Citation
If you use GRATEv2 in your research, please cite our related publications:

```bibtex
@article{gamdha2024computational,
title={GRATEv2: Computational Tools for Real-time Analysis of High-throughput High-resolution TEM (HRTEM) Images of Conjugated Polymers},
author={Gamdha, Dhruv and Fair, Ryan and Krishnamurthy, Adarsh and Gomez, Enrique and Ganapathysubramanian, Baskar},
journal={arXiv preprint arXiv:2411.03474},
year={2024}
}
```



<!-- Information missing 
1. Input data information (image format and parameters such as image resolution pix2nm, d-spacing to search)
2. Inputs to bayesian optimization file
  - training data
  - ground truth data
  - directory paths

-->
