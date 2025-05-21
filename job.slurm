#!/bin/bash
# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)
#SBATCH --time=12:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks-per-node=36  # 36 processor core(s) per node
#SBATCH --mem=185G       # maximum memory per node
#SBATCH --job-name="gratev2"


# Load necessary modules
source $VENV/hrtem/bin/activate
cd /work/mech-ai-scratch/dgamdha/projects/hrtem_analysis
python bayesianOpt.py > bayesianOpt.log 2>&1