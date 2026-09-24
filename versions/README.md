# Code versions

The top-level code (`pigen/`, `metrics/`, `scripts/`, `tests/`) is the **default** version, taken from
the former `pigen_isambardAI/` checkout. Each folder here is an *overlay*: it contains **only the files
that differ** from the default, at the same relative paths. Everything else is shared.

| Version | Origin | What differs |
|---|---|---|
| `aug26/` | `pigen_Aug26/` (git branch `aug26`) | Package code identical to default. Older metric scripts: `compute_compactness.py` has a `--charge` option (average cationic/anionic radii); no `legacy_valid` in `spp_error.py`; `compute_mled.py` without NaN-fingerprint guard. |
| `draft/` | `pigen_draft/` (no git history) | Early prototype: CSP scripts, charge-balance eval (`pigen/eval/charges.py`, `check_balance.py`, `bond_table_ionic.py`), `sample.py`, older settings/tests. Its data is the same `Alex_MP_20_M_LED` split (compactness stored as `target_energy`). **Its model (`diffusion_pi.py`) is not here:** it differed from the default only in the compactness loss, which is now `--cmpt_mode draft`. |

Removed (2026-09-24):
- `occ/` — its `diffusion_pi.py` did not run (a mis-indented `def` ended the `CSPDiffusion` class after
  `__init__`) and computed compactness from the noise predictions; the soft E[r³] idea is in the default
  model. Its `simple_dataset.py` (occupancy/disorder dataset) was never wired into a model and removed
  `SimpleCrystDataset`. Full code is kept in git branch `occ`.
- The separate model files `diffusion_pigate.py`, `diffusion_pi_cmptdiff.py` and the `*_detach.py`
  entry points: all folded into `pigen/assets/diffusion_pi.py` + `--cmpt_mode` (see the top-level README).

`LOG_DIR` in the draft overlay's `settings.py` was updated to `outputs/logs` to match the new layout.

## Using a version

Overlay it onto a scratch copy (don't overwrite the default in place):

```bash
git worktree add ../pigen-draft HEAD        # or: cp -Rc . ../pigen-draft
rsync -a versions/draft/ ../pigen-draft/
```

For `aug26` and `occ` the exact original commit history is kept as git branches
(`git switch occ`). `draft` exists only as this overlay.

## Compactness-loss fixes (2026-09-24)

These change results for `--cmpt_mode none|types|full` compared with runs made before this date
(`draft` is kept verbatim):

1. **Radii index.** Atom-type channel `k` is atomic number `k+1`, but `a_radii` is indexed by atomic
   number, so every element got its predecessor's radius (Li, Na, K… got 0). Now `a_radii[1:]`.
2. **Type sharpening.** `softmax` of a near one-hot vector over 100 classes gives the true element only
   p ≈ 0.03, so the loss saw an "average" atom everywhere (clean LiCoO₂: 1.85 instead of 0.77). Now
   `softmax(x0 / cmpt_tau)`, `cmpt_tau = 0.1` → 0.778.
3. **Target column.** `--cmpt_target` reads the target from any CSV column (e.g. `compactness_av` in
   `BVSE_PSI`, which has no `target_energy`; that caused the `KeyError: 'target_energy'` crash).

Old checkpoints still load; they default to `cmpt_mode='types'`.
