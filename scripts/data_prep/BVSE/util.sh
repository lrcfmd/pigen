#!/bin/bash

#SBATCH --job-name=BVSEutil
#SBATCH --output=../../outputs/slurm/data_prep/util.out
#SBATCH -e ../../outputs/slurm/data_prep/util.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00         # Hours:Mins:Secs

hostname

source /home/u6fo/andrij.u6fo/miniforge3/etc/profile.d/conda.sh
conda activate pigen

srun python ../../scripts/data_prep/BVSE/disorder_order_tranform_full.py
