#!/bin/bash

#SBATCH --job-name=EMD
#SBATCH --output=outputs/slurm/EMD.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=0:5:00         # Hours:Mins:Secs

hostname
nvidia-smi --list-gpus

#module load cray-python
source /home/u6fo/andrij.u6fo/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
which python
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)"
srun python3 compute_EMD.py
