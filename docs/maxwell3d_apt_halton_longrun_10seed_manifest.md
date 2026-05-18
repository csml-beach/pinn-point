# Maxwell 3D APT vs Halton Long-Run Manifest (10 Seeds)

Date: 2026-05-18  
Status: ready-to-submit

## Goal

Run a longer `maxwell_coil_core_3d` head-to-head where early/mid-iteration
separation between `adaptive_power_tempered` (APT) and `halton` is expected to
be larger than `e300_i4`, while keeping the study tractable.

## Rationale for this spec

- `e300_i4` gave small deltas and statistically inconclusive results.
- In prior budget sweep, larger separation appeared at higher iteration counts
  (e.g., `e150_i8` and `e100_i12`), but `e100_i12` is very expensive.
- This manifest uses a **longer-than-e300_i4** run with more refinement rounds:
  `iterations=8`, `epochs=200` (total optimizer epochs = 1600 per method).

## Fixed protocol

- Problem: `maxwell_coil_core_3d`
- Methods: `adaptive_power_tempered,halton`
- Device: CPU (`m3-cpu-xl`)
- Mesh size: `0.35`
- Reference mesh factor: `0.15`
- Validation policy: use script defaults (same as prior fair runs)
- Setup mode: `--skip-setup` (assuming remote repo already prepared)

## Seeds (10)

`42,123,456,789,1011,2022,3033,4044,5055,6066`

## Submit command

```bash
bash scripts/submit_m3_large_cpu_elasticity_3d_screen_confirm.sh \
  --problem maxwell_coil_core_3d \
  --methods adaptive_power_tempered,halton \
  --seeds 42,123,456,789,1011,2022,3033,4044,5055,6066 \
  --epochs 200 \
  --iterations 8 \
  --mesh-size 0.35 \
  --reference-mesh-factor 0.15 \
  --parallel 5 \
  --config ../remote-ops/pinn-point/config.m3-cpu-xl.env \
  --sync-root /Users/arash/Documents/GitHub/pinn-point/outputs/m3-cpu-xl-maxwell3d-apt-vs-halton-longrun10-e200i8-2026-05-18 \
  --session-prefix cpu-maxwell3d-apt-halton-longrun10-e200i8 \
  --skip-setup
```

## Expected outputs

Root:

- `outputs/m3-cpu-xl-maxwell3d-apt-vs-halton-longrun10-e200i8-2026-05-18`

Per run:

- `reports/all_methods_histories.csv`
- `reports/performance_summary.txt`
- `reports/run_config.json`

## Analysis protocol

Head-to-head metrics (`APT` vs `Halton`):

1. `total_error` (primary)
2. `fixed_rms_residual`
3. `cumulative_runtime_sec`

Tests:

- Exact paired sign-flip permutation (two-sided)
- Bootstrap 95% CI on mean paired difference

Compare:

- Final selected iteration metrics
- Convergence curves (mean ± std) across iterations

