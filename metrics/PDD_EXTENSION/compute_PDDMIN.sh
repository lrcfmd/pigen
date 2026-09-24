#!/bin/bash

#SBATCH --job-name=PDD3
#SBATCH --output=../../outputs/slurm/metrics/PDDMIN_L1.out
#SBATCH -e ../../outputs/slurm/metrics/PDDMIN.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:20:00         # Hours:Mins:Secs

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
#srun python compute_PDDMIN_toRef.py 
srun python compute_PDDMIN_toRef_L1.py
#'/home/u6fo/andrij.u6fo/pigen/generated/PDDMIN/gen_PDDMIN3/denovo_[3.0]_2.0_compact5141_spp.csv_AMD.pickle'
