# analysis/: results of metric/analysis code (not tracked in git)

- `VAL/`: Volume-Allocation-Likelihood study: ICSD Voronoi stats, fitted per-element GMMs (`icsd_phi_gmms_*.json`), scored ICSD/LiNbN dataframes, plots, `lowVALcifs/`. Code is in `metrics/VAL/`; the scripts use cwd-relative paths, so run them from here: `cd analysis/VAL && python ../../metrics/VAL/plot_val.py`. See `VAL/README` for correlation results.
- `examples/MgV2O5/`: worked example input (`.cif`) and metric outputs.
