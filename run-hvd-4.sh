#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2006142

module load tensorflow
module list

export DATADIR=/scratch/project_2006142/
export KERAS_HOME=/scratch/project_2006142/keras-cache
export TRANSFORMERS_CACHE=/scratch/project_2006142/transformers-cache

set -xv
srun python3 $*
                 
