# Histories CSV: Columns and Definitions

This document explains each column written by `reports/histories.csv` so you can interpret run outputs and aggregate results consistently.

Each row corresponds to an iteration index (starting at 0). Some methods may append values at slightly different cadence; when a value is missing for an iteration, it is recorded as `NaN`.

## Columns

- iteration
  - The iteration index for the experiment loop.

- adaptive_total_error
  - Integrated error for the adaptive method: ∫Ω (u_adapt − u_ref)^2 dΩ.
  - Computed on the fixed high-fidelity reference mesh by projecting the PINN prediction to the reference FE space and integrating.

- random_total_error
  - Integrated error for the random method: ∫Ω (u_rand − u_ref)^2 dΩ.
  - Same evaluation procedure as the adaptive method for fairness.

- adaptive_total_error_rms
  - RMS error for the adaptive method: sqrt(mean((u_adapt − u_ref)^2)) evaluated at the reference-mesh vertices.
  - DOF-agnostic alternative to the integrated error; no FE projection required.

- random_total_error_rms
  - RMS error for the random method: sqrt(mean((u_rand − u_ref)^2)) evaluated at the reference-mesh vertices.

- adaptive_total_residual
  - Integrated PDE residual during training on the current mesh (∫Ω r^2) for the adaptive method.
  - This is computed on the evolving training mesh and primarily used for refinement logic/tracking.

- random_total_residual
  - Integrated PDE residual during training (∫Ω r^2) for the random method on its sampling mesh/points.

- adaptive_fixed_total_residual
  - Fixed-grid integral of residual^2: ∫Ω r^2 dΩ for the adaptive method evaluated on the high-fidelity reference mesh.
  - This provides a fair, mesh-independent residual metric for comparison.

- random_fixed_total_residual
  - Fixed-grid integral of residual^2 for the random method on the same reference mesh.

- adaptive_fixed_boundary_residual
  - Integrated boundary residual (∫∂Ω r_b^2) for the adaptive method on the reference mesh when available.
  - May be `NaN` if boundary-only evaluation is not produced in a given iteration.

- random_fixed_boundary_residual
  - Integrated boundary residual for the random method on the reference mesh when available.

- adaptive_fixed_rms_residual
  - RMS residual for the adaptive method: sqrt(mean(r^2)) at the reference-mesh vertices.
  - DOF-agnostic and comparable across methods.

- random_fixed_rms_residual
  - RMS residual for the random method at the reference-mesh vertices.

## Notes

- RMS vs Integrated:
  - Integrated metrics depend on FE projection/integration and reflect domain-weighted totals.
  - RMS metrics are point-sampled at reference vertices, avoiding DOF/projection issues; use them for quick, fair comparisons.

- Missing values (NaN):
  - If an evaluation step fails or a size mismatch occurs (e.g., FE DOF count vs vertex count), the pipeline records `NaN` and logs a warning.

- Reproducibility:
  - All metrics are computed against the same reference mesh to ensure a fair comparison between adaptive and random methods.

- Suggested usage:
  - For publication plots, prefer the fixed-grid metrics (RMS residual and integrated residual) and present both the integrated and RMS error.
  - Use training residuals to illustrate the behavior during refinement, but rely on fixed-grid metrics for cross-method fairness.
