**Physics-Informed Generative AI framework for crystal structure prediction and design.**

## Physics Informed Generation (PIGEN) of Crystal Structures 

### Setup environment
```bash
conda env create -f environment.yml
conda activate pigen
```

All dependencies are managed via conda; setup.py is only for local package registration:

### Install package (editable mode)
```bash
pip install -e .
```

### Data used for training can be accessed at
[https://huggingface.co/datasets/UoLiverpool/Alex_MP_20_M_LED/](https://huggingface.co/datasets/UoLiverpool/Alex_MP_20_M_LED/)

### Code Base and Attribution
This repository builds on the foundation of the open-source project https://github.com/jiaor17/DiffCSP, originally implementing Denoising Diffusion Probabilistic Model for crystal structure prediction.

### Enhancements in This Project
We extend and adapt the original DiffCSP codebase with several key contributions:
 - Physics-informed logic integrated into the sampling process
 - Conditional generation with target-guided control via classification-free guidance
 - Featurised dataset with local chemical and structural environment feature, enabling out-of-distributiion extrapolation
 - Chemistry-informed structure evaluation tools
 - Modular refactoring for better reproducibility and configuration management.
- Support for PyTorch Distributed Data Parallel to accelerate large-scale training across multiple GPUs or nodes

These additions are primarily implemented in:
assets/diffusion_pi.py — physics-aware sampling logic
train.py and generate.py — training and inference entry points
sample.py - inference with multi-optimisation of properties via classifier-free guidance
partial_sample.py - inference with partially defined chemical composition, e.g., for Li-based materials
eval/ - chemistry-informed structure featurisation and evaluation tools
common/utils.py — shared utility functions

### Tests
Download model checkpoint: [huggingface.co/DeepDrew/PIGEN/](https://huggingface.co/DeepDrew/PIGEN/)
```bash
wget https://huggingface.co/DeepDrew/PIGEN/pigen_CE_step215150.ckpt
```
After downloading, place the checkpoint file in:
```bash
checkpoints/
```
This ensures test_dummy_generate.py and generation scripts can locate it.
*Note: Large files (>100 MB) are stored externally to keep the repository lightweight.
By default, the test_dummy_generate.py test is skipped if the checkpoint is not found*

Run:
```bash
pytest tests --disable-warnings
```
### Run with Docker
*Note: The Docker image is intentionally left with a flexible entry point (/bin/bash) to allow the user to either train or generate as needed, following the instructions below. This design choice supports both CPU and GPU environments.*

```bash
docker build -t pigen .
docker run --rm pigen
```

### Model training
For training with 


### Project structure

├── checkpoints
├── data
│   └── Alex_MP_20_M_LED/
├── environment.yml
├── log
├── pigen
│   ├── __init__.py
│   ├── assets/
│   ├── common/
│   ├── eval/
│   ├── generate.py
│   ├── normalization
│   ├── partial_sample.py
│   ├── settings.py
│   └── train.py
├── README.md
├── setup.py
├── tests
│   ├── dummy_data/
│   ├── dummy_logs/
│   ├── fixtures/
│   ├── conftest.py
│   ├── test_dependecies.py
│   ├── test_dummy_generate.py
│   ├── test_dummy_training.py
│   ├── test_pd_structure_parsing.py
│   └── test_torch_installation.py
└── verify_environment_installs.py


### License and Credit
The original repository DiffCSP licensed under the MIT License.
We retain this license and clearly mark any modified components.
We gratefully acknowledge the authors of DiffCSP for their contribution to the research and open-source community.
