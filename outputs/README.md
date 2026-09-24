# outputs/: run artefacts (not tracked in git)

- `logs/<experiment>/<timestamp>/`: training / fine-tuning runs (checkpoints, `settings.yaml`, `training.log`). `pigen.settings.Paths.LOG_DIR` points here.
- `slurm/`: SLURM stdout/stderr from `scripts/hpc/*.sh` (`metrics/` and `data_prep/` for jobs submitted from those folders).
