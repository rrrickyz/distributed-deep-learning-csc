#!/bin/bash
#SBATCH --account=project_2006142
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10

module load tensorflow
module list

export DATAIR=/scratch/project_2006142/
export KERAS_HOME=/scratch/project_2006142/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_2006142/transformers-cache

set -xv
python3 $*
