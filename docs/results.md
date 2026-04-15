# Current Results Summary

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

The stored Poisson comparison is not yet aligned with the current adaptive mainline.

- It predates `adaptive_persistent`.
- It predates the current validation/checkpoint policy.

That means Poisson is the first PDE that should be rerun if we want a cleaner cross-PDE story.

## Recommended Next Experiment

If the goal is more publishable experiments, the next move should be:

1. rerun **Poisson** with `adaptive_persistent, adaptive, random, halton, rad`
2. keep the current checkpoint policy consistent with the recent runs
3. use `10` seeds on `m3-large-cpu`

That gives:

- one PDE where `adaptive_persistent` currently wins on error (`advection-diffusion`)
- one PDE where it wins on residual at near-comparable error (`Navier-Stokes`)
- one baseline PDE rerun under the current method stack (`Poisson`)

So the immediate answer is:

- **No, advection-diffusion does not need to be rerun first.**
- **Yes, Poisson should be rerun next if we want the next clean experiment.**
