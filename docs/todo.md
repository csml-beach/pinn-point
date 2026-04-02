# Repo TODO

This file tracks follow-ups that should be revisited after the current refactor and smoke-test work.

## Backend / Legacy API

- [ ] Fix the residual quadrature path in `train/mesh_refinement.py`: `compute_model_residual_on_reference_quadrature()` still calls `reference_mesh.Elements2D()` and falls back to Monte Carlo on the current netgen/ngsolve backend. Replace it with a stable mesh traversal or integration path so evaluation is backend-safe and deterministic.
- [ ] Audit other `ngsolve`/`netgen` API calls that may have changed across versions, especially in mesh traversal, element access, and `GridFunction` projection code.

## Core Experiment Structure

- [ ] Finish moving all PDE-specific logic out of `train/pinn_model.py` and `train/fem_solver.py` so new PDEs only require work inside `train/problems/`.
- [ ] Replace the remaining string-based method dispatch in `train/experiments.py` with a cleaner registry-driven flow end to end.
- [ ] Decouple geometry selection from the current hardcoded L-shaped domain in `train/geometry.py` so problems can own their own domain construction.
- [ ] Implement the hybrid anchor-based adaptive method described in `docs/hybrid_adaptive_plan.md`.

## Completed Recently

- [x] Add a first-class smoke-test path (`scripts/smoke_test.sh` and `train/main.py smoke`) so major changes can be checked with a tiny end-to-end run.
- [x] Expand the canonical multi-method metrics/reporting stack described in `docs/metrics_expansion_plan.md` before implementing the hybrid method.
- [x] Revisit the adaptive-vs-baseline budget policy so all methods use the same configured optimizer-step budget per iteration, adaptive no longer gets extra fine-tuning, and runtime metrics include method-specific sampling/refinement overhead.
