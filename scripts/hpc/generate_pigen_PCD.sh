#!/bin/bash

#SBATCH --job-name=PCD
#SBATCH --output=outputs/slurm/PCD_gen.out
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
        --model_path /home/u6fo/andrij.u6fo/pigen/outputs/logs/PCD/2026-05-27-12-24-56/ \
        --save_path /home/u6fo/andrij.u6fo/pigen/generated/PCD/ \
        --targets 9 24 \
	--guidance 3.0


#priority_ood = ['tP24', 'hP24', 'mP24', 'oP24',  # high score, Wyckoff-valid
#                'hP22', 'mP22', 'oP22',            # even, Wyckoff-reachable
#                'cI24',                             # direct Wyckoff multiplicity
#                'cI4', 'cI6']
#
#                tP: idx=9
