# GRATEv2: Automated HRTEM Image Analysis Framework [Paper Link](https://arxiv.org/abs/2411.03474)

**GRATEv2** helps automate the detection of crystalline structures in High-Resolution Transmission Electron Microscopy (HRTEM) images. It aids material scientists, chemists, and engineers in studying nanoscale microstructures.

## Key Features
- **Image Processing Pipeline (main.py)**  
  Parameter-based detection of crystals in HRTEM images, using a config file in `configFiles`.
- **Bayesian Optimization (bayesianOpt.py)**  
  Tunes parameters automatically for improved detection using the `gp_minimize` function.
- **Structural Feature Extraction**  
  Outputs d-spacing, orientation angles, aspect ratios, and intercrystalline correlations.
- **Incremental Analysis**  
  Integrates parameter updates within batch processing, guided by a detection performance metric.

## Repository Structure
- **`main.py`**: Processes HRTEM images based on a chosen config file. Outputs detected crystals and features.  
- **`bayesianOpt.py`**: Optimizes parameters by running `main.py` iteratively. Users can edit paths, iteration counts, and the objective function.
- **`requirements.txt`**: Lists Python dependencies.  
- **`vggAnnotatorCSVRead.py`**: Converts CSV annotations (from VGG Image Annotator) into ground truth masks for training and validation.
- **Configuration Files**: Contain directory paths and algorithm parameters. Modes allow debugging or specialized features.

## Installation
1. Clone the repo:
   ```
   git clone https://<YourUsername>@bitbucket.org/baskargroup/gratev2.git GRATEv2
   cd GRATEv2
   ```
2. Create and activate a virtual environment:
   ```
   python3 -m venv gratev2_venv
   source gratev2_venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. **Edit Config File:** Update `data_dir` and `base_result_dir` in `BO_200Evals.cfg` (or another `.cfg` file in `configFiles`).
2. **Run Analysis:**
   ```
   python main.py BO_200Evals.cfg
   ```
3. **Perform Bayesian Optimization:**  
   Update paths in `bayesianOpt.py`, then:
   ```
   python bayesianOpt.py
   ```

## Ground Truth Annotations
- Use the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) for crystal annotations.
- Convert CSV to annotations with:
   ```
   python vggAnnotatorCSVRead.py
   ```
- Adjust the script’s `base_dir_rpath` and `annotation_csv_fname` as needed.

## Thresholds and Batch Sizes
When running Bayesian optimization, consider how batch size influences threshold setting for the Wasserstein distance.

## Customization
- **Loss Function:** Modify `objective()` in `bayesianOpt.py` to include other crystal metrics.
- **Parameters and Modes:** Adjust these in your config file for debugging or custom visualization.

## Contributing
Open issues or submit pull requests for bug reports or enhancements.

## License
Please refer to the [LICENSE](LICENSE) file here.

## Citation
If you use GRATEv2 in research, please cite:
```bibtex
@article{gamdha2024computational,
title={GRATEv2: Computational Tools for Real-time Analysis of High-throughput High-resolution TEM (HRTEM) Images of Conjugated Polymers},
author={Gamdha, Dhruv and Fair, Ryan and Krishnamurthy, Adarsh and Gomez, Enrique and Ganapathysubramanian, Baskar},
journal={arXiv preprint arXiv:2411.03474},
year={2024}
}
```
