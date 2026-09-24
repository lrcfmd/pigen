#!/bin/bash

#SBATCH --job-name=BVSE
#SBATCH --output=outputs/slurm/BVSE_PSItrain.out
#SBATCH -e outputs/slurm/BVSE_PSItrain.err
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
srun python pigen/fine_tune.py --cmpt_mode none --cmpt_target compactness_av --experiment='BVSE_PSI' --data_name BVSE_PSI --prop 'psi_nso_300K' --ckpt_path="/lus/lfs1aip2/projects/u6fo/pigen/outputs/logs/PCD/2026-05-27-12-24-56/epoch=999-step=352000.ckpt"
#srun python pigen/fine_tune.py --cmpt_mode none --experiment='BVSE' --data_name BVSE --prop 'fom_nso' --ckpt_path="/lus/lfs1aip2/projects/u6fo/pigen/outputs/logs/PCD/2026-05-27-12-24-56/epoch=999-step=352000.ckpt"
