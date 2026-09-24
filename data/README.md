# data/ — input datasets (not tracked in git)

`pigen.settings.Paths.DATA_DIR` points here; training looks for `data/<data_name>/{train,val,test}.csv`
and caches processed tensors as `data/<data_name>/*_ori_<prop>.pt`.

| Folder | Contents |
|---|---|
| `Alex_MP_20_M_LED/` | Alex-MP-20 with M-LED / compactness targets (`full`, `train`, `val`, `test`, `small_test`). |
| `BVSE/` | BVSE dataset (`fom_nso`), ordered/disordered splits and cached `.pt` tensors. |
| `BVSE_PSI/` | BVSE + psi/NSO 300 K dataset used for fine-tuning. |

Scripts that build these splits live in `scripts/data_prep/`. They read and write in the current
directory, so run them from inside the data folder, e.g.
`cd data/BVSE && python ../../scripts/data_prep/BVSE/split_df.py BVSE_notNA.csv`.
