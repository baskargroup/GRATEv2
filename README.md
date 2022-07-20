## Create python environment using requirements.txt
### Using virtualenv
[]: # Language: python
[]: # Path: virtualenv.py
[]: # Code:

```python
mkdir -p ~/venv # create virtualenv directory to store virtualenvs, ignore if it already exists and continue 
cd ~/venv # change to virtualenv directory
pip install virtualenv # Install virtualenv, ignore if already installed
virtualenv hrtem # Create a new virtualenv
source hrtem/bin/activate # Activate the virtualenv
pip install -r <path>/<to>/<project>/<dir>/requirements.txt # Install all the packages in requirements.txt
```

### using conda
[]: # Language: python
[]: # Path: conda.py
[]: # Code:

```python
conda create -n hrtem python=3.6 # Create a new conda environment
source activate hrtem # Activate the conda environment
conda install -r <path>/<to>/<project>/<dir>/requirements.txt # Install all the packages in requirements.txt
```
