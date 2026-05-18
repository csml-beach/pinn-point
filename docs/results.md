# Current Results Summary

This file is intentionally concise and only tracks the paper-facing, post-fix state.
Detailed historical/tuning narrative is omitted from the main summary.

## Statistical Test Protocol (Paired Permutation + CI)

Use this for all head-to-head claims from seed runs:

- Unit: paired seed (`seed` shared across compared methods)
- Difference: `d_i = metric(method_B, seed_i) - metric(method_A, seed_i)`  
  (for lower-is-better metrics, `d_i > 0` favors `method_A`)
- Test: exact paired sign-flip permutation test (two-sided p-value)
- Effect size: mean paired difference
- Interval: bootstrap 95% CI on mean paired difference

Minimum report block:

- run directory root
- metric column name
- compared methods
- paired seed count `n`
- mean(method_A), mean(method_B)
- mean paired difference (`method_B - method_A`)
- permutation p-value
- bootstrap 95% CI

## Fair-Halton Context (Status)

- Pre-fix Halton/Sobol runs (before 2026-04-30 fix) are historical only.
- Paper-facing claims should use post-fix seed-matched reruns.

## Confirmed Packaging Scope

The following benchmarks are confirmed and safe for tabulation:

- Allen-Cahn (AC), 20 seeds: `outputs/m3-large-cpu-allen-cahn-obstacles-screen-confirm-400e`
- Navier-Stokes (NS), 20 seeds: `outputs/m3-cpu-xl-navier-stokes-halton-rerun-tend1p0-ref0035-dt0001-200e-20seed`
- Advection-Diffusion (AD), 20 seeds: `outputs/m3-cpu-xl-advection-halton-rerun-300e-20seed`
- Poisson hard-narrow, 20 seeds: `outputs/m3-cpu-xl-poisson-hard-doe-400e-hard-narrow-20seed/hard_narrow/ref_0p05`

3D elasticity is intentionally kept outside this file and remains tracked in its dedicated run notes/artifacts.

## Current Headline Outcomes

- AD: strongest positive case for adaptive (error and residual advantage in the approved comparison suite).
- NS: clear error/residual/runtime tradeoff (Halton stronger on error/runtime; adaptive stronger on residual).
- AC: top error methods are close; adaptive methods provide stronger residual control.
- Poisson (`hard_narrow`, `ref=0.05`, 20 seeds): adaptive_power_tempered has best mean error, but is statistically tied with Halton head-to-head on error and residual.
- Maxwell 3D (`maxwell_coil_core_3d`, 5 seeds per budget): adaptive_power_tempered shows a consistent error/residual advantage over Halton across tested `(epochs, iterations)` budgets, with a runtime penalty.
- Maxwell 3D long-run (`e300_i8`, `mesh=0.30`, `ref=0.10`, 20 seeds): adaptive methods remain better on mean error/residual than Halton, with a clear runtime penalty; head-to-head p-values for error/residual remain above 0.05.

## Maxwell 3D Long-Run (All 4 Methods, 20 Seeds)

Run root:

- `outputs/m3-cpu-xl-maxwell3d-all4-longrun10-e300i8-ms030-ref010-2026-05-18`

Dedup policy: latest `run_id` per seed.

Backed CSVs:

- `outputs/m3-cpu-xl-maxwell3d-all4-longrun10-e300i8-ms030-ref010-2026-05-18/merged_all4_10seed_dedup_latest.csv`
- `outputs/m3-cpu-xl-maxwell3d-all4-longrun10-e300i8-ms030-ref010-2026-05-18/merged_apt_halton_10seed_dedup_latest.csv`
- `artifacts/metrics/maxwell3d_e300i8_ms030_ref010_10seed/merged_all4_10seed_dedup_latest.csv`
- `artifacts/metrics/maxwell3d_e300i8_ms030_ref010_10seed/merged_apt_halton_10seed_dedup_latest.csv`
- `outputs/m3-cpu-xl-maxwell3d-all4-longrun10-e300i8-ms030-ref010-2026-05-18/merged_all4_20seed_dedup_latest.csv`
- `outputs/m3-cpu-xl-maxwell3d-all4-longrun10-e300i8-ms030-ref010-2026-05-18/merged_apt_halton_20seed_dedup_latest.csv`
- `artifacts/metrics/maxwell3d_e300i8_ms030_ref010_20seed/merged_all4_20seed_dedup_latest.csv`
- `artifacts/metrics/maxwell3d_e300i8_ms030_ref010_20seed/merged_apt_halton_20seed_dedup_latest.csv`

Selected-checkpoint means (20 seeds):

| Method | Total Error | Fixed RMS Residual | Runtime (s) |
| --- | ---: | ---: | ---: |
| `adaptive_power_tempered` | `0.2442 ± 0.1357` | `26.06 ± 24.44` | `135.3 ± 17.2` |
| `random` | `0.2511 ± 0.1364` | `20.62 ± 3.93` | `63.7 ± 12.8` |
| `halton` | `0.2612 ± 0.1466` | `31.35 ± 19.27` | `71.2 ± 24.9` |
| `rad` | `0.2608 ± 0.1413` | `46.00 ± 105.52` | `55.0 ± 10.6` |

Paired test (`adaptive_power_tempered` vs `halton`, `n = 20`, deduped):

- Error: mean paired diff (`halton - APT`) = `+0.01707`, `p = 0.1008`, 95% CI `[-0.00140, +0.03650]`
- Residual: mean paired diff = `+5.289`, `p = 0.5170`, 95% CI `[-10.04, +16.93]`
- Runtime: mean paired diff = `-64.08 s` (Halton faster), `p = 1.91e-06`, 95% CI `[-77.93, -50.04]`

## Maxwell 3D Budget Sweep (5 Seeds Per Budget)

Run root:

- `outputs/m3-cpu-xl-maxwell3d-budget-sweep-2026-05-05`

Methods: `adaptive_power_tempered`, `halton`, `random`, `rad`  
Seeds: `42, 123, 456, 789, 1011`  
Problem: `maxwell_coil_core_3d`

APT vs Halton (mean deltas; lower is better for error/residual):

| Budget | Error delta (APT - Halton) | Residual delta (APT - Halton) | Runtime delta (APT - Halton) |
| --- | ---: | ---: | ---: |
| `e100_i12` | `-0.00963` | `-12.02` | `+20836 s` |
| `e150_i8` | `-0.01095` | `-13.15` | `+18382 s` |
| `e200_i6` | `-0.00319` | `-23.43` | `+11541 s` |
| `e300_i4` | `-0.00151` | `-1.89` | `+4838 s` |

Bookkeeping note: runtime is cumulative iteration runtime and includes point selection/refinement + training + fixed-reference evaluation each iteration.

## Poisson Hard-Narrow (20 Seeds, `ref=0.05`)

Run root:

- `outputs/m3-cpu-xl-poisson-hard-doe-400e-hard-narrow-20seed/hard_narrow/ref_0p05`

Selected-checkpoint means:

| Method | Relative L2 Error | Relative Fixed L2 Residual | Runtime (s) |
| --- | ---: | ---: | ---: |
| `adaptive_power_tempered` | `0.4076 ± 0.1097` | `0.7195 ± 0.1136` | `177.7 ± 28.7` |
| `halton` | `0.4469 ± 0.1262` | `0.7151 ± 0.1133` | `162.9 ± 53.4` |
| `random` | `0.5341 ± 0.1889` | `0.8122 ± 0.1447` | `144.7 ± 57.6` |
| `adaptive_halton_base` | `0.6591 ± 0.1679` | `0.7980 ± 0.1767` | `161.7 ± 40.2` |
| `rad` | `0.6755 ± 0.2356` | `0.8793 ± 0.1953` | `148.3 ± 64.5` |

Paired test (`adaptive_power_tempered` vs `halton`, `n = 20`):

- Error: mean paired diff (`halton - adaptive_power_tempered`) = `+0.03935`, `p = 0.2939`, 95% CI `[-0.02819, +0.10943]`
- Residual: mean paired diff = `-0.00445`, `p = 0.9108`, 95% CI `[-0.07773, +0.07344]`
- Runtime: mean paired diff = `-14.77 s`, `p = 0.2859`, 95% CI `[-42.51, +9.32]`
