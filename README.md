## Physics Informed Generation (PIGEN) of Crystal Structures 

## Setup environment
conda env create -f environment.yml
conda activate pigen

## Install package (editable mode)
pip install -e .

## Data used for training can be accessed at
https://huggingface.co/datasets/UoLiverpool/Alex_MP_20_M_LED/tree/main

## Code Base and Attribution
This repository builds on the foundation of the open-source project https://github.com/jiaor17/DiffCSP, originally implementing Denoising Diffusion Probabilistic Model for crystal structure prediction.

## Enhancements in This Project
We extend and adapt the original DiffCSP codebase with several key contributions:

 - Physics-informed logic integrated into the sampling process

 - Conditional generation with target-guided control via classification-free guidance

 - Featurised dataset with local chemical and structural environment feature, enabling out-of-distributiion extrapolation

 - Chemistry-informed structure evaluation tools

 - Modular refactoring for better reproducibility and configuration management.

These additions are primarily implemented in:

assets/diffusion_pi.py — physics-aware sampling logic

training.py and generation.py — training and inference entry points

partial_csp.py - inference with partially defined chemical composition, e.g., for Li-based materials

metrics/ - chemistry-informed structure featurisation and evaluation tools

common/utils.py — shared utility functions

## License and Credit
The original repository DiffCSP licensed under the MIT License.
We retain this license and clearly mark any modified components.
We gratefully acknowledge the authors of DiffCSP for their contribution to the research and open-source community.
