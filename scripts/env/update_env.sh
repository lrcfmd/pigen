#!/bin/bash

#SBATCH --job-name=EMD
#SBATCH --output=outputs/slurm/ENVINSTAL.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=1:5:00         # Hours:Mins:Secs

hostname
nvidia-smi --list-gpus

#module load cray-python
source /home/u6fo/andrij.u6fo/miniforge3/etc/profile.d/conda.sh
#conda activate pytorch_env
conda activate pigen
#conda env update -f small_env.yml --prune
python -c "import torch; print(torch.__version__, torch.version.cuda)"
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

module load gcc-native/13.2
module load cuda/12.6

export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

echo "== Toolchain =="
echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
gcc --version | sed -n '1p'
g++ --version | sed -n '1p'
nvcc --version | sed -n '1,4p'

python - <<'PY'
import platform
import torch
print("machine =", platform.machine())
print("torch =", torch.__version__)
print("torch.cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device =", torch.cuda.get_device_name(0))
PY

python -m pip install ninja
python -m pip install --no-cache-dir --no-build-isolation --verbose torch_scatter
python -m pip install --no-cache-dir --no-build-isolation --verbose torch_sparse

#
#machine = aarch64
#torch = 2.6.0+cu126 cuda = 12.6
#torch_scatter = 2.1.2
#torch_sparse = 0.6.18
#wheel tag = cp311-cp311-linux_aarch64

python - <<'PY'
import torch_scatter
import torch_sparse
print("torch_scatter =", torch_scatter.__file__)
print("torch_sparse =", torch_sparse.__file__)
PY

pip list >> PACKAGES_INSTALLED
