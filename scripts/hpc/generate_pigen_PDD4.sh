#!/bin/bash

#SBATCH --job-name=PDD4
##SBATCH --output=outputs/slurm/PDDgen4_det.out
#SBATCH --output=outputs/slurm/PDDgen4.out
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
        --model_path /home/u6fo/andrij.u6fo/pigen/outputs/logs/PDDTRAIN1000/2026-05-19-07-53-30/ \
        --save_path /home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN4 \
        --targets 4.0
        #--model_path /home/u6fo/andrij.u6fo/pigen/outputs/logs/PDDTRAIN_detach/2026-05-19-08-05-35/ \
        #--save_path /home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN4_det \
