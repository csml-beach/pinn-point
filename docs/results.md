# Current Results Summary

## Clean 5-Method Paper Runs (2026-04-28) ← USE THESE FOR PAPER

Definitive paper-facing runs with all 5 methods in a **single run directory per seed** (no gap-fill artifacts). Both AC and NS ran at 100 epochs, 4 iterations on `m3-cpu-xl` at commit `5e8c148`.

**Methods**: `halton`, `random`, `rad`, `adaptive_power_tempered`, `adaptive_halton_base`  
**Seeds**: 42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066 (10 seeds)

**Output roots**:
- `outputs/m3-large-cpu-allen-cahn-obstacles-5method-100e-10seed/` — run_id `2026-04-28_08-14-06_screen-cpu-allen-cahn-screen-seed{N}`
- `outputs/m3-large-cpu-navier-stokes-5method-100e-10seed/` — run_id `2026-04-28_08-16-52_screen-seed{N}`

**10-seed means (final relative L2 error ± std)**:

| Problem | Method | Mean L2 ↓ | Std | Mean Resid ↓ | Std |
| --- | --- | ---: | ---: | ---: | ---: |
| Allen-Cahn | `adaptive_power_tempered` | **0.4112** | 0.0131 | **0.03977** | 0.00695 |
| Allen-Cahn | `adaptive_halton_base` | 0.4123 | 0.0234 | 0.04296 | 0.00605 |
| Allen-Cahn | `rad` | 0.4158 | 0.0226 | 0.04403 | 0.00596 |
| Allen-Cahn | `random` | 0.4193 | 0.0205 | 0.04753 | 0.00884 |
| Allen-Cahn | `halton` | 0.4213 | 0.0229 | 0.04810 | 0.00947 |
| Navier-Stokes | `halton` | **0.6287** | 0.0318 | 0.06228 | 0.01099 |
| Navier-Stokes | `adaptive_halton_base` | 0.6304 | 0.0233 | 0.05735 | 0.00919 |
| Navier-Stokes | `adaptive_power_tempered` | 0.6330 | 0.0250 | **0.05659** | 0.01008 |
| Navier-Stokes | `random` | 0.6328 | 0.0304 | 0.06127 | 0.01110 |
| Navier-Stokes | `rad` | 0.6331 | 0.0268 | **0.05651** | 0.00858 |

**Note**: NS methods are effectively tied on L2 (spread < 0.5%); adaptive methods have lower residual on both problems. AC shows clearer separation.

---

## Approved Suite with `adaptive_halton_base`, 20 Seeds

Current strongest approved-suite comparison:

- methods: `adaptive_halton_base`, `adaptive_persistent`, `adaptive`, `random`, `halton`, `rad`
- seeds: `20` per benchmark
- metrics: selected-checkpoint metrics from `reports/all_methods_histories.csv`
- manifest policy: per-seed manifests are the source of truth; failed first-attempt Navier-Stokes folders are ignored

Primary output roots:

- `outputs/m3-large-cpu-allen-cahn-obstacles-screen-haltonbase-800e-20seed`
- `outputs/m3-large-cpu-advection-diffusion-screen-haltonbase-300e-20seed`
- `outputs/m3-large-cpu-navier-stokes-screen-haltonbase-tend1p0-ref0035-dt0001-20seed`

## Method Status

Current paper-facing default suite:

- `adaptive_power_tempered`: current clean mesh-native candidate; keep as the
  main power-tempered method with `beta_max = 4.0` and no coverage floor.
- `adaptive_halton_base`: strongest residual-control reference method.
- `adaptive_persistent`, `adaptive`, `random`, `halton`, `rad`: retained
  baselines.

Retained negative/tuning variants, not default:

- `adaptive_entropy_balanced`: tested as an entropy-mixture adaptive variant;
  not competitive enough to keep in the default suite.
- `adaptive_power_tempered_beta25` and `adaptive_power_tempered_beta30`: lower
  beta caps did not improve the overall tradeoff.
- `adaptive_power_tempered_floor15` and `adaptive_power_tempered_floor25`: fixed
  true-area coverage floors did not close the residual gap and degraded
  advection performance.

These negative variants remain registered and selectable via `--methods` so the
same dead ends are not rediscovered later.

## `adaptive_power_tempered` Head-to-Head, 10 Seeds

`adaptive_power_tempered` is a mesh-native variant that replaces the explicit
Halton backbone with one power-tempered element distribution:

`p_i proportional to area_i^a * exp(beta * z_i)`.

Here `z_i` is a rank-persistent residual score in `[0, 1]`, and `beta` increases
when the residual field is spatially concentrated. The goal is to preserve broad
coverage without piggybacking on a low-discrepancy sequence.

Primary output roots:

- `outputs/m3-large-cpu-allen-cahn-obstacles-power-vs-haltonbase-800e-10seed`
- `outputs/m3-large-cpu-advection-power-vs-haltonbase-300e-10seed`
- `outputs/m3-large-cpu-navier-stokes-power-vs-haltonbase-tend1p0-ref0035-dt0001-10seed`

### 10-seed selected-checkpoint means

| Problem | Method | Relative L2 Error | Relative Fixed L2 Residual |
| --- | --- | ---: | ---: |
| Allen-Cahn obstacles | `adaptive_power_tempered` | `0.18155 ± 0.00627` | `0.01797 ± 0.00487` |
| Allen-Cahn obstacles | `adaptive_halton_base` | `0.18357 ± 0.00616` | `0.01584 ± 0.00308` |
| Advection-diffusion | `adaptive_power_tempered` | `0.57611 ± 0.06164` | `0.59527 ± 0.04870` |
| Advection-diffusion | `adaptive_halton_base` | `0.59712 ± 0.07019` | `0.62473 ± 0.06560` |
| Navier-Stokes channel-obstacle | `adaptive_power_tempered` | `0.49786 ± 0.03128` | `0.14821 ± 0.01426` |
| Navier-Stokes channel-obstacle | `adaptive_halton_base` | `0.49860 ± 0.03005` | `0.14395 ± 0.01279` |

### Paired seed-by-seed check

Paired differences below use `adaptive_power_tempered - adaptive_halton_base`;
negative means `adaptive_power_tempered` is better.

| Problem | Metric | Mean Paired Difference | Sign Count Favoring `adaptive_power_tempered` |
| --- | --- | ---: | ---: |
| Allen-Cahn obstacles | Relative L2 Error | `-0.00202 ± 0.00243` | `9 / 10` |
| Allen-Cahn obstacles | Relative Fixed L2 Residual | `+0.00213 ± 0.00316` | `2 / 10` |
| Advection-diffusion | Relative L2 Error | `-0.02101 ± 0.02331` | `8 / 10` |
| Advection-diffusion | Relative Fixed L2 Residual | `-0.02947 ± 0.04744` | `7 / 10` |
| Navier-Stokes channel-obstacle | Relative L2 Error | `-0.00075 ± 0.00721` | `4 / 10` |
| Navier-Stokes channel-obstacle | Relative Fixed L2 Residual | `+0.00426 ± 0.00917` | `3 / 10` |

Interpretation:

- On Allen-Cahn, `adaptive_power_tempered` improves solution error slightly but
  gives up residual control relative to `adaptive_halton_base`.
- On advection-diffusion, `adaptive_power_tempered` improves both solution error
  and residual in this 10-seed head-to-head.
- On Navier-Stokes, `adaptive_power_tempered` is effectively tied on solution
  error but gives up a small amount of residual control.
- The result is promising as a cleaner method family, but the current
  `beta_max = 4.0` setting appears slightly too concentrated on two of the
  three benchmarks. The next tuning target is a lower `beta_max`.

### Beta cap tuning

Follow-up tuning compared `beta_max = 2.5`, `3.0`, and `4.0` against
`adaptive_halton_base` on the same 10 seeds. The variants are named
`adaptive_power_tempered_beta25`, `adaptive_power_tempered_beta30`, and
`adaptive_power_tempered`, respectively.

Primary output roots:

- `outputs/m3-large-cpu-allen-cahn-obstacles-power-beta-tune-800e-10seed`
- `outputs/m3-large-cpu-advection-power-beta-tune-300e-10seed`
- `outputs/m3-large-cpu-navier-stokes-power-beta-tune-tend1p0-ref0035-dt0001-10seed`

| Problem | Method | Relative L2 Error | Relative Fixed L2 Residual |
| --- | --- | ---: | ---: |
| Allen-Cahn obstacles | `adaptive_power_tempered_beta25` | `0.18181 ± 0.00666` | `0.01884 ± 0.00491` |
| Allen-Cahn obstacles | `adaptive_power_tempered_beta30` | `0.18199 ± 0.00671` | `0.01801 ± 0.00429` |
| Allen-Cahn obstacles | `adaptive_power_tempered` | `0.18155 ± 0.00627` | `0.01797 ± 0.00487` |
| Allen-Cahn obstacles | `adaptive_halton_base` | `0.18357 ± 0.00616` | `0.01584 ± 0.00308` |
| Advection-diffusion | `adaptive_power_tempered_beta25` | `0.58319 ± 0.06518` | `0.59331 ± 0.02997` |
| Advection-diffusion | `adaptive_power_tempered_beta30` | `0.58615 ± 0.05756` | `0.59578 ± 0.02204` |
| Advection-diffusion | `adaptive_power_tempered` | `0.57611 ± 0.06164` | `0.59527 ± 0.04870` |
| Advection-diffusion | `adaptive_halton_base` | `0.59712 ± 0.07019` | `0.62473 ± 0.06560` |
| Navier-Stokes channel-obstacle | `adaptive_power_tempered_beta25` | `0.50012 ± 0.02755` | `0.14582 ± 0.01736` |
| Navier-Stokes channel-obstacle | `adaptive_power_tempered_beta30` | `0.50146 ± 0.03305` | `0.14495 ± 0.01554` |
| Navier-Stokes channel-obstacle | `adaptive_power_tempered` | `0.50059 ± 0.03105` | `0.14404 ± 0.01623` |
| Navier-Stokes channel-obstacle | `adaptive_halton_base` | `0.50054 ± 0.03076` | `0.14305 ± 0.01284` |

Interpretation:

- Lowering `beta_max` did not fix the Allen-Cahn or Navier-Stokes residual gap.
- On advection-diffusion, `beta_max = 2.5` gives the best residual, but the
  original `beta_max = 4.0` gives the best error and remains essentially tied
  on residual.
- The safest fixed power-tempered setting is still the original
  `beta_max = 4.0`. The problem is not simply over-concentration from too large
  a beta cap.

### Coverage-floor tuning

Follow-up tuning tested whether the residual gap was caused by insufficient
background coverage. The variants mix the persistent power-tempered element
distribution with true area-proportional mesh coverage:

`p_i = (1 - rho) * power_tempered_i + rho * area_i_distribution`.

The tested variants were `rho = 0.15` and `rho = 0.25`, named
`adaptive_power_tempered_floor15` and `adaptive_power_tempered_floor25`.

Primary output roots:

- `outputs/m3-large-cpu-allen-cahn-obstacles-power-floor-tune-800e-10seed`
- `outputs/m3-large-cpu-advection-power-floor-tune-300e-10seed`
- `outputs/m3-large-cpu-navier-stokes-power-floor-tune-tend1p0-ref0035-dt0001-10seed`

| Problem | Method | Relative L2 Error | Relative Fixed L2 Residual |
| --- | --- | ---: | ---: |
| Allen-Cahn obstacles | `adaptive_power_tempered_floor15` | `0.18167 ± 0.00692` | `0.01879 ± 0.00447` |
| Allen-Cahn obstacles | `adaptive_power_tempered_floor25` | `0.18111 ± 0.00604` | `0.02251 ± 0.00758` |
| Allen-Cahn obstacles | `adaptive_power_tempered` | `0.18155 ± 0.00627` | `0.01797 ± 0.00487` |
| Allen-Cahn obstacles | `adaptive_halton_base` | `0.18357 ± 0.00616` | `0.01584 ± 0.00308` |
| Advection-diffusion | `adaptive_power_tempered_floor15` | `0.58669 ± 0.06896` | `0.60834 ± 0.03984` |
| Advection-diffusion | `adaptive_power_tempered_floor25` | `0.59217 ± 0.06650` | `0.61857 ± 0.04898` |
| Advection-diffusion | `adaptive_power_tempered` | `0.57611 ± 0.06164` | `0.59527 ± 0.04870` |
| Advection-diffusion | `adaptive_halton_base` | `0.59712 ± 0.07019` | `0.62473 ± 0.06560` |
| Navier-Stokes channel-obstacle | `adaptive_power_tempered_floor15` | `0.49733 ± 0.03218` | `0.15116 ± 0.01368` |
| Navier-Stokes channel-obstacle | `adaptive_power_tempered_floor25` | `0.50063 ± 0.03052` | `0.14961 ± 0.01075` |
| Navier-Stokes channel-obstacle | `adaptive_power_tempered` | `0.50404 ± 0.02640` | `0.14309 ± 0.01262` |
| Navier-Stokes channel-obstacle | `adaptive_halton_base` | `0.50235 ± 0.02837` | `0.14495 ± 0.01338` |

Interpretation:

- A fixed true-area floor did not solve the residual-control weakness.
- On Allen-Cahn, both floors preserve or slightly improve error, but residual
  gets worse; `floor25` is especially poor.
- On advection-diffusion, the floors reduce the original `adaptive_power_tempered`
  advantage in both error and residual.
- On Navier-Stokes, `floor15` improves mean error, but it gives up residual.
- The best current power-tempered mainline remains the unfloored
  `adaptive_power_tempered` with `beta_max = 4.0`.

### 20-seed selected-checkpoint means

| Problem | Best Mean Error | `adaptive_halton_base` Error | Best Mean Residual | `adaptive_halton_base` Residual |
| --- | ---: | ---: | ---: | ---: |
| Allen-Cahn obstacles | `rad`: `0.1864 ± 0.0134` | `0.1913 ± 0.0124` | `adaptive_halton_base`: `0.01392 ± 0.00343` | `0.01392 ± 0.00343` |
| Advection-diffusion | `adaptive_halton_base`: `0.6423 ± 0.0822` | `0.6423 ± 0.0822` | `adaptive_halton_base`: `0.6785 ± 0.0959` | `0.6785 ± 0.0959` |
| Navier-Stokes channel-obstacle | `halton`: `0.4927 ± 0.0280` | `0.4968 ± 0.0267` | `adaptive_halton_base`: `0.1384 ± 0.0148` | `0.1384 ± 0.0148` |

### Full method means

Allen-Cahn obstacles (`800` epochs, `6` iterations):

| Method | Relative L2 Error | Relative Fixed L2 Residual |
| --- | ---: | ---: |
| `rad` | `0.1864 ± 0.0134` | `0.02364 ± 0.00598` |
| `halton` | `0.1873 ± 0.0132` | `0.02448 ± 0.00711` |
| `random` | `0.1878 ± 0.0127` | `0.02664 ± 0.00839` |
| `adaptive_persistent` | `0.1898 ± 0.0128` | `0.01667 ± 0.00488` |
| `adaptive` | `0.1909 ± 0.0114` | `0.01400 ± 0.00409` |
| `adaptive_halton_base` | `0.1913 ± 0.0124` | `0.01392 ± 0.00343` |

Advection-diffusion (`300` epochs, `8` iterations):

| Method | Relative L2 Error | Relative Fixed L2 Residual |
| --- | ---: | ---: |
| `adaptive_halton_base` | `0.6423 ± 0.0822` | `0.6785 ± 0.0959` |
| `random` | `0.6660 ± 0.0965` | `0.8253 ± 0.1030` |
| `halton` | `0.6690 ± 0.0888` | `0.8144 ± 0.0862` |
| `rad` | `0.6704 ± 0.0742` | `0.7800 ± 0.0591` |
| `adaptive_persistent` | `0.6833 ± 0.0905` | `1.0046 ± 0.1911` |
| `adaptive` | `0.6968 ± 0.0705` | `0.9071 ± 0.0980` |

Navier-Stokes channel-obstacle (`t_end = 1.0`, `dt = 0.001`, `200` epochs, `6` iterations):

| Method | Relative L2 Error | Relative Fixed L2 Residual |
| --- | ---: | ---: |
| `halton` | `0.4927 ± 0.0280` | `0.1694 ± 0.0200` |
| `rad` | `0.4937 ± 0.0283` | `0.1552 ± 0.0181` |
| `adaptive_halton_base` | `0.4968 ± 0.0267` | `0.1384 ± 0.0148` |
| `random` | `0.4970 ± 0.0225` | `0.1725 ± 0.0272` |
| `adaptive_persistent` | `0.4987 ± 0.0253` | `0.1386 ± 0.0105` |
| `adaptive` | `0.5008 ± 0.0269` | `0.1443 ± 0.0156` |

### Paired seed-by-seed conclusions

Paired differences below use `adaptive_halton_base - baseline`; negative means `adaptive_halton_base` is better.

Allen-Cahn:

- Error: `adaptive_halton_base` is not the winner; it is significantly worse than `rad`, `halton`, and `random` on mean paired error.
- Residual: `adaptive_halton_base` is the best residual method, beating `random`, `halton`, and `rad` on `20 / 20` seeds and beating `adaptive_persistent` on `17 / 20` seeds.
- Versus plain `adaptive`, residual is essentially tied: mean paired residual difference `-0.000077`, `p = 0.884`.

Advection-diffusion:

- Error: `adaptive_halton_base` beats every baseline; paired wins are `17 / 20` vs `adaptive`, `14 / 20` vs `adaptive_persistent`, `14 / 20` vs `random`, `15 / 20` vs `halton`, and `17 / 20` vs `rad`.
- Residual: `adaptive_halton_base` beats every baseline with paired p-values below `0.001` versus all methods except no exception; wins range from `16 / 20` to `20 / 20`.
- This is the cleanest positive benchmark for the new scheme.

Navier-Stokes:

- Error: differences are small; `halton` and `rad` have slightly better mean error, but `adaptive_halton_base` is statistically comparable to most methods.
- Residual: `adaptive_halton_base` has the best mean residual and significantly beats `random`, `halton`, and `rad`; it is essentially tied with `adaptive_persistent`.
- Versus plain `adaptive`, `adaptive_halton_base` improves both error and residual modestly.

### Current interpretation

The defensible paper claim is not that `adaptive_halton_base` always minimizes solution error. The stronger and better-supported claim is:

> A Halton-backed persistent adaptive sampler provides the most consistent PDE-residual control across the approved suite while remaining competitive in solution error, and it is the clear winner on advection-diffusion in both error and residual.

This gives a cleaner story than the original `adaptive_persistent` mainline:

- The Halton backbone fixes the coverage weakness exposed by competitive `random` and `halton` baselines.
- Rank-normalized persistence makes residual targeting less sensitive to raw PDE residual scale.
- On Allen-Cahn and Navier-Stokes, the method sits on the residual-favorable side of the error/residual Pareto tradeoff.
- On advection-diffusion, it improves both objectives.

Recommended figure/table package:

1. A three-problem table with mean ± standard deviation for error and residual.
2. A paired-difference plot for `adaptive_halton_base` versus `random`, `halton`, and `rad`.
3. A Pareto plot of mean error versus mean residual, one panel per problem.
4. A short negative-control note: Allen-Cahn still rewards broad-coverage methods on solution error, but not on residual.

## Allen-Cahn Obstacles, 400 Epochs

Current strongest Allen-Cahn comparison:

- problem: `allen_cahn_obstacles_2d`
- methods: `adaptive_persistent`, `adaptive`, `random`, `halton`, `rad`
- seeds: `20`
- profile: `screen` (`6` iterations, `400` epochs)
- metrics: selected-checkpoint metrics, aggregated over space and time

Primary output root:

- `outputs/m3-large-cpu-allen-cahn-obstacles-screen-confirm-400e`

### 20-seed selected-checkpoint means

| Method | Relative L2 Error | Relative Fixed L2 Residual | Selected Iteration | Selected Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| `random` | `0.19869 ± 0.01429` | `0.01995 ± 0.00468` | `5.00 ± 0.00` | `34.56 ± 0.28` |
| `rad` | `0.19952 ± 0.01361` | `0.01787 ± 0.00419` | `5.00 ± 0.00` | `33.90 ± 0.63` |
| `adaptive` | `0.19951 ± 0.01296` | `0.01484 ± 0.00306` | `4.95 ± 0.22` | `34.45 ± 1.30` |
| `halton` | `0.19963 ± 0.01302` | `0.01885 ± 0.00507` | `5.00 ± 0.00` | `34.27 ± 0.59` |
| `adaptive_persistent` | `0.20145 ± 0.01648` | `0.01830 ± 0.00395` | `4.95 ± 0.22` | `36.20 ± 1.54` |

Headline interpretation:

- At `400` epochs, the error story is effectively flat: `random`, `rad`, `adaptive`, and `halton` are all in the same error band.
- The residual story is not flat: `adaptive` is clearly the best residual method.
- `adaptive_persistent` is competitive, but it is not the winner on this benchmark.

### Paired seed-by-seed check

Against `rad` on residual:

- mean paired residual difference (`adaptive - rad`): `-0.00303`
- sign count: `20 / 20` seeds favor `adaptive`

Against `adaptive_persistent` on residual:

- mean paired residual difference (`adaptive - adaptive_persistent`): `-0.00346`
- sign count: `19 / 20` seeds favor `adaptive`, `1 / 20` favors `adaptive_persistent`

Against `halton` on residual:

- mean paired residual difference (`adaptive - halton`): `-0.00402`
- sign count: `17 / 20` seeds favor `adaptive`, `3 / 20` favor `halton`

On error, the extra 10 seeds showed that the winner is not stable enough to claim strongly:

- mean paired error difference (`rad - random`): `+0.00083`
- sign count: `11 / 20` seeds favor `random`, `9 / 20` favor `rad`

Interpretation:

- The main Allen-Cahn conclusion is now stable: `adaptive` is the best residual method.
- The Allen-Cahn error winner is intentionally weakly stated, because the top methods are statistically too close to separate cleanly.

## Navier-Stokes Channel-Obstacle, Long Horizon

Current strongest Navier-Stokes comparison:

- problem: `navier_stokes_channel_obstacle`
- horizon: `t_end = 1.0`
- reference settings: `reference_mesh_factor = 0.035`, `dt = 0.001`
- methods: `adaptive_persistent`, `adaptive`, `random`, `halton`, `rad`
- seeds: `10`
- profile: `screen` (`6` iterations, `200` epochs)
- metrics: selected-checkpoint metrics, aggregated over space and time

Primary output root:

- `outputs/m3-large-cpu-navier-stokes-screen-confirm-tend1p0-ref0035-dt0001`

### 10-seed selected-checkpoint means

| Method | Relative L2 Error | Relative Fixed L2 Residual | Selected Iteration | Selected Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| `rad` | `0.4966 ± 0.0379` | `0.1576 ± 0.0265` | `4.8 ± 0.4` | `89.0 ± 7.0` |
| `halton` | `0.4976 ± 0.0349` | `0.1632 ± 0.0222` | `4.8 ± 0.4` | `89.0 ± 6.7` |
| `random` | `0.4991 ± 0.0221` | `0.1650 ± 0.0139` | `4.8 ± 0.6` | `89.8 ± 10.8` |
| `adaptive` | `0.5016 ± 0.0332` | `0.1441 ± 0.0169` | `5.0 ± 0.0` | `94.2 ± 3.3` |
| `adaptive_persistent` | `0.5023 ± 0.0341` | `0.1414 ± 0.0176` | `5.0 ± 0.0` | `95.0 ± 3.1` |

Headline interpretation:

- Best mean error is still non-adaptive (`rad`, with `halton` very close).
- Best mean residual is `adaptive_persistent`.
- The adaptive variants appear to trade a small amount of error for better PDE residual.

### Paired seed-by-seed check

To test whether "better residual at comparable error" is defensible, compare `adaptive_persistent` seed-by-seed against the strongest non-adaptive baselines.

Against `halton`:

- mean paired error difference (`adaptive_persistent - halton`): `+0.0047 ± 0.0158`
- mean paired residual difference: `-0.0218 ± 0.0149`
- residual sign count: `10 / 10` seeds favor `adaptive_persistent`
- error sign count: `6 / 10` seeds favor `halton`, `4 / 10` favor `adaptive_persistent`

Against `rad`:

- mean paired error difference (`adaptive_persistent - rad`): `+0.0058 ± 0.0107`
- mean paired residual difference: `-0.0162 ± 0.0196`
- residual sign count: `7 / 10` seeds favor `adaptive_persistent`
- error sign count: `7 / 10` seeds favor `rad`, `3 / 10` favor `adaptive_persistent`

Interpretation:

- Versus `halton`, the residual advantage of `adaptive_persistent` is very clean and consistent, while the error penalty is small.
- Versus `rad`, the same tradeoff exists, but it is weaker and less uniform.
- The defensible claim is therefore:
  `adaptive_persistent` improves residual substantially relative to `halton` while staying in the same general error band.

## Advection-Diffusion Status

The current hardened advection-diffusion benchmark already has a valid `adaptive_persistent` 10-seed five-way comparison.

Output roots:

- `outputs/jetstream-medium-advection-persistent-fiveway-final`
- `outputs/m3-large-cpu-advection-persistent-fiveway-final`

Combined 10-seed selected-checkpoint means:

| Method | Relative L2 Error | Relative Fixed L2 Residual |
| --- | ---: | ---: |
| `adaptive_persistent` | `0.6110 ± 0.0538` | `0.8023 ± 0.0733` |
| `halton` | `0.6166 ± 0.0693` | `0.7784 ± 0.0690` |
| `random` | `0.6174 ± 0.0886` | `0.7823 ± 0.0800` |
| `rad` | `0.6282 ± 0.0714` | `0.7774 ± 0.0902` |
| `adaptive` | `0.6518 ± 0.0744` | `0.8631 ± 0.0835` |

Interpretation:

- This benchmark already supports `adaptive_persistent` as the best method on mean error.
- It does **not** support `adaptive_persistent` as the best residual method.
- This experiment is already aligned with the current adaptive mainline and does not need to be rerun immediately.

## Poisson Status

Poisson is no longer a blocking gap, but it is not a good showcase for the current adaptive mainline.

What we found:

- `poisson_ring` at higher budget (`400` epochs) becomes competitive, but it does not favor `adaptive_persistent`.
- `poisson_ring_hard` also does not help the adaptive methods, even after fixing the point-budget confound.
- The harder Poisson geometry mostly rewards broad-coverage baselines rather than interior-focused adaptive refinement.

Interpretation:

- Poisson is useful negative evidence.
- It is not the strongest paper-facing benchmark for the current method family.

## Recommended Next Experiment

If the goal is paper packaging rather than more raw runs, the next move should be:

1. keep **advection-diffusion** as the benchmark where `adaptive_persistent` wins on error
2. keep **Navier-Stokes** as the benchmark where `adaptive_persistent` improves residual at near-comparable error
3. keep **Allen-Cahn** as the benchmark where `adaptive` is the clear residual winner and the error tier is effectively tied
4. de-emphasize **Poisson** as a control/negative case rather than a main showcase

That gives a much cleaner story:

- one PDE where persistence helps most on error (`advection-diffusion`)
- one PDE where persistence helps most on residual while staying near the top error band (`Navier-Stokes`)
- one PDE where plain adaptive residual focusing is strongest on residual (`Allen-Cahn`)
- one family of PDEs where broad coverage remains hard to beat (`Poisson`)

So the immediate answer is no longer "rerun Poisson." The immediate answer is:

- **No, Poisson does not need more tuning before writing.**
- **Yes, the current three-PDE story is strong enough to start packaging into paper figures and tables.**
