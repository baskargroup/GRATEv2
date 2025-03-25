## Bayesian Optimization Training Steps

- Sample images from the dataset (`.tif` format)
    - 70% training set (not necessary)
    - 30% validation set (not necessary)
- Convert the `.tif` images to .png format (use `helper/tif2png.py`)
- Can brighten the .png images using `helper/brightenTEM.py` (this can be helpful for manual annotation) 
- Use VGG annotator to annotate the `.png` images (both train and test)
    - Create annotations as convex polygons
    - Export the annotations as a .csv file 
- Use `vggAnnotatorCSVRead.py` (set variable values inside, i.e. name of annotations .csv file, and path to base directory containing the images) to mark the .csv annotations to `.png` images and save (These are the ground truth for the training and validation sets)
    - Note: The `bayesianOpt.py` will create the binary masks from the ground truth images and put them inside the `Masks` folder
- Create the below folder structure in the `DATA/BO/ver#` folder
    ```bash
    Expected directory structure:
    |--training/
        |--groundTruth/             
            |--Images/      : (.png format)             
            |--Masks/       : (.png format)
        |--input/           : (.tif format)
            |--img1.tif
            |--img2.tif
            |--...
        |--evaluations/
            
    |--validation/
        |--groundTruth/             
            |--Images/      : (.png format)
            |--Masks/       : (.png format)
        |--input/           : (.tif format)
            |--img1.tif
            |--img2.tif
            |--...
        |--output/                  
            |--BO_para/             
            |--manual_para/
    ``` 
- Create the train and validation distribution of the input (`.tif`) and ground truth annotated images (`.png`) and put them inside the `training` and `validation` folders respectively
    - train:
        - `.tif` images in `training/input`
        - annotated ground truth `.png` images in `training/groundTruth/Images`
    - validation:
        - `.tif` images in `validation/input`
        - annotated ground truth `.png` images in `validation/groundTruth/Images`
- Move the `ver#` folder from local to the NOVA and place inside `DATA/BO/` folder
- Get interactive node on NOVA and activate `hrtem` venv 
- Set below variables inside `bayesianOpt.py`
    - `pths['trn_rpth']`: `relative path to ver#/training/`
    - `pths['val_rpth']`: `relative path to ver#/validation/`
    - `gp_minimize(..., n_calls=#, n_initial_points=#, ...)`: `number of iterations for the bayesian optimization` and `number of initial points for the bayesian optimization`
- Run `bayesianOpt.py` to get the optimized parameters for the training set
    - `traininig/evaluations` will contain the evaluation results for each iteration and the optimized parameters
    - `validation/output/BO_para` will contain the dectections for the validation set using the optimized parameters
    - `validation/output/manual_para` will contain the detections for the validation set using the `manual.cfg` parameters