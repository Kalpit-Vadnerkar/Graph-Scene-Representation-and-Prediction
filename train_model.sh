#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=graph_model
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kvadner@clemson.edu

module load anaconda3
source activate pytorch

python Run_Model.py --mode train