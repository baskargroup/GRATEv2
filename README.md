# Guide to setup and use the project
## Create python environment using requirements.txt
### Using virtualenv

```python
mkdir -p ~/venv # create virtualenv directory to store virtualenvs, ignore if it already exists and continue 
cd ~/venv # change to virtualenv directory
pip install virtualenv # Install virtualenv, ignore if already installed
virtualenv hrtem # Create a new virtualenv
source hrtem/bin/activate # Activate the virtualenv
pip install -r <path>/<to>/<project>/<dir>/requirements.txt # Install all the packages in requirements.txt
```

### Using conda

```python
conda create -n hrtem python=3.6 # Create a new conda environment
source activate hrtem # Activate the conda environment
conda install -r <path>/<to>/<project>/<dir>/requirements.txt # Install all the packages in requirements.txt
```

## Setup the config files.
- Refer the [para.cfg](configFiles/para.cfg) file for as an example of the configuration file.
- Modify the configuration file by
	- Setting the path to the data directory.
	- Setting the path to the result directory.
	- Finetuning the parameters.
	- Turning on/off the modes.

## Run the project
```python
python main.py para.cfg # Run the project
```