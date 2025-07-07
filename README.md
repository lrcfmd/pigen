**Physics-Informed Generative AI framework for crystal structure prediction and design.**

## Physics Informed Generation (PIGEN) of Crystal Structures 

### Setup environment
conda env create -f environment.yml
conda activate pigen

### Install package (editable mode)
pip install -e .

### Data used for training and the trained model checkpoint can be accessed at
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
training.py and generation.py — training and inference entry points
partial_csp.py - inference with partially defined chemical composition, e.g., for Li-based materials
eval/ - chemistry-informed structure featurisation and evaluation tools
common/utils.py — shared utility functions

### Tests
Download model checkpoint: [huggingface.co/DeepDrew/PIGEN/](https://huggingface.co/DeepDrew/PIGEN/)

```bash
wget https://huggingface.co/your-username/your-model-repo/resolve/main/dummy_ckpt
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
## Run with Docker

```bash
docker build -t myproject .
docker run myproject
```

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
