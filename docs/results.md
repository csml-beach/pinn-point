# Current Results Summary

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
