# fair-halton-paper-reruns-ac-ns-ad-2026-05-03

Curated paper-experiment snapshot for the 5-seed reruns performed after changing Halton/Sobol sampling from cached per-iteration reuse to fair per-iteration resampling.

This snapshot intentionally includes only:

- Allen-Cahn obstacles: `outputs/m3-cpu-xl-allen-cahn-obstacles-halton-rerun-400e-5seed`
- Navier-Stokes channel obstacle: `outputs/m3-cpu-xl-navier-stokes-halton-rerun-tend1p0-ref0035-dt0001-200e-5seed`
- Advection-diffusion: `outputs/m3-cpu-xl-advection-halton-rerun-300e-5seed`

Poisson reruns are intentionally excluded. They are useful negative controls, but they are not part of this paper-facing fair-Halton snapshot.

The DVC-tracked payload lives in:

`artifacts/snapshots/fair-halton-paper-reruns-ac-ns-ad-2026-05-03/`

Snapshot description:

> Paper-facing AC + NS + AD 5-seed reruns after making Halton fair by resampling quasi-random points each iteration instead of reusing a cached point set. Poisson excluded.

Source-state note: the repository HEAD at snapshot time was `f43c0b71f19a2175d9bf1867e1824b2a0625e776` with a dirty working tree containing the fair-Halton/sampler overlay. The relevant overlay diff is stored in the DVC payload at `source_state/fair_halton_and_sampler_overlay.patch`.

