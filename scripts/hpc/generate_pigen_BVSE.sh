#!/bin/bash

#SBATCH --job-name=FOMNSO4
#SBATCH --output=outputs/slurm/NSO_gen15.out
#SBATCH -e outputs/slurm/NSO_gen.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
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
srun python pigen/generate.py \
        --model_path /lus/lfs1aip2/projects/u6fo/pigen/outputs/logs/BVSE/2026-07-14-13-59-47 \
        --save_path  /lus/lfs1aip2/projects/u6fo/pigen/generated/FOMNSO/ \
        --targets 0.15  \
	--guidance 2.0
