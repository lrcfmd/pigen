#!/bin/bash

#SBATCH --job-name=PDD
#SBATCH --output=outputs/slurm/PDDtrain.out
#SBATCH --nodes=4
#SBATCH --gpus=16
##SBATCH --nodes=1
##SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00         # Hours:Mins:Secs

hostname

#module load cray-python
module load gcc-native/13.2
module load cuda/12.6

source /home/u6fo/andrij.u6fo/miniforge3/etc/profile.d/conda.sh
conda activate pigen

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus

python -c "import torch; print(torch.__version__, torch.version.cuda)"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
srun python pigen/train.py --experiment='PDDTRAIN1000' --data_name PDD --prop PDDMIN 
#--ckpt_path="/home/u6fo/andrij.u6fo/pigen/outputs/logs/PDDTRAIN/2026-05-05-11-46-40/epoch=126-step=178689.ckpt"
