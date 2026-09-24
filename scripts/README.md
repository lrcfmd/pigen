# scripts/

- `hpc/`: SLURM jobs for Isambard-AI (train / fine-tune / generate). Submit from the repo root: `sbatch scripts/hpc/generate_pigen_BVSE.sh`. Logs go to `outputs/slurm/`.
- `data_prep/`: dataset construction (splits, BVSE order/disorder transforms, lattice scaler). See `data/README.md`.
- `env/`: environment setup (`small_env.yml`, `update_env.sh`, `verify_environment_installs.py`, installed-package record).
