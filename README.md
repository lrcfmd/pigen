## Physics Informed Generation (PIGEN) of Crystal Structures 
PIGEN (Physics Informed Generation) is a framework for generating novel crystal structures by integrating physics-informed sampling, chemically guided control, and structural evaluation into a denoising diffusion model. Building on DiffCSP, PIGEN enables targeted generation beyond known chemical spaces and supports out-of-distribution extrapolation. This enables the generation of chemically and structurally diverse, physically plausible crystal candidates, yielding a higher fraction of stable structures per batch and achieving greater chemical and structural diversity than frameworks such as DiffCSP or MatterGen (as demonstrated in our benchmarks).

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


### Model training
For re-training the model with the default dataset - Alex_MP_20_MLED, run:

```bash
python pigen/train.py
```

This will use the default data and conditioning properties and is equivalent to

```bash
python pigen/train.py --data_name Alex_MP_20_M_LED --prop entropy_sum target_energy
```

Compactness-loss flags (`train.py` and `fine_tune.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--cmpt_mode {none,types,full,draft}` | `types` | `none`: loss only logged, no gradient · `types`: gradient through atom types · `full`: through atom types and lattice · `draft`: legacy draft loss (no gradient) |
| `--cmpt_tau FLOAT` | `0.1` | softmax temperature of the atom-type estimate |
| `--cost_cmpt FLOAT` | `1.0` | weight of the compactness loss |
| `--cmpt_target COLUMN` | `target_energy` | CSV column with the compactness target |

```bash
python pigen/fine_tune.py --cmpt_mode none --cmpt_target compactness_av \
       --data_name BVSE_PSI --prop psi_nso_300K --ckpt_path <pretrained.ckpt>
```

### Model Inference
You can use your trained model or download the model's checkpoint from:
 [huggingface.co/DeepDrew/PIGEN/](https://huggingface.co/DeepDrew/PIGEN/)

After downloading, place the checkpoint file in:
```bash
checkpoints/
```
This ensures pigen/generate.py can locate it.

Run
```bash
cd pigen
python generate.py
```

### Code Base and Key Contributions

This repository builds on [DiffCSP](https://github.com/jiaor17/DiffCSP), an open-source implementation of denoising diffusion probabilistic models for crystal structure prediction. We have further developed and extended it as described below.

 - Physics-informed logic integrated into the sampling process
 - Conditional generation with target-guided control via classifier-free guidance
 - Featurised dataset with local chemical and structural environment feature, enabling out-of-distribution extrapolation
 - Chemistry-informed structure evaluation tools
 - Modular refactoring for better reproducibility and configuration management.
- Support for PyTorch Distributed Data Parallel to accelerate large-scale training across multiple GPUs or nodes

### Tests
Run:
```bash
pytest tests 
```
### Run with Docker
*Note: The Docker image is intentionally left with a flexible entry point (/bin/bash) to allow the user to either train or generate as needed, following the instructions below. This design choice supports both CPU and GPU environments.*

```bash
docker build -t pigen .
```

### To run with CPU only:
```bash
docker run --rm pigen
```
### To run with GPU:
```bash
docker run --rm --gpus all pigen
```
### Project structure
Code is tracked in git; `data/`, `generated/`, `outputs/` and `analysis/` hold data and results and are
git-ignored (each has its own README).
```text
├── analysis                    # results of metric/analysis runs (git-ignored)
│   ├── VAL/
│   └── examples/
├── checkpoints                 # pretrained checkpoint + settings.yaml
├── data                        # input datasets (git-ignored)
│   ├── Alex_MP_20_M_LED/
│   ├── BVSE/
│   └── BVSE_PSI/
├── environment.yml
├── generated                   # generated structures (git-ignored)
│   ├── FOMNSO/
│   ├── PCD/
│   └── PDDMIN/
├── metrics                     # evaluation metrics
│   ├── compute_chgnet_energy.py
│   ├── compute_compactness.py
│   ├── compute_mled.py
│   ├── new_compositions.py
│   ├── spp_error.py
│   ├── SPP_collected.json
│   ├── PDD_EXTENSION/
│   └── VAL/
├── outputs                     # run artefacts (git-ignored)
│   ├── logs/                   # training runs (settings.LOG_DIR)
│   └── slurm/                  # SLURM stdout/stderr
├── pigen
│   ├── __init__.py
│   ├── assets/                 # cspnet.py, diffusion_pi.py (model, --cmpt_mode), simple_dataset.py
│   ├── common/
│   ├── eval/
│   ├── fine_tune.py
│   ├── generate.py
│   ├── normalization
│   ├── partial_sample.py
│   ├── settings.py
│   └── train.py
├── README.md
├── scripts
│   ├── data_prep/              # dataset construction (splits, BVSE transforms, lattice scaler)
│   ├── env/                    # environment setup, verify_environment_installs.py
│   └── hpc/                    # SLURM jobs; submit from the repo root
├── setup.py
├── tests
│   ├── dummy_data/
│   ├── dummy_logs/
│   ├── fixtures/
│   ├── conftest.py
│   ├── test_dependecies.py
│   ├── test_dummy_training.py
│   ├── test_pd_structure_parsing.py
│   └── test_torch_installation.py
└── versions                    # overlays of alternative code versions (aug26, draft)
```

### License and Credit
The original repository DiffCSP licensed under the MIT License.
We retain this license and clearly mark any modified components.
We gratefully acknowledge the authors of DiffCSP for their contribution to the research and open-source community.
