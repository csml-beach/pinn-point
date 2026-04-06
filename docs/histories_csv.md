# Histories CSV: Columns and Definitions

This document explains the canonical run-history CSV output so you can interpret experiment results and aggregate them consistently.

Each row corresponds to one method at one iteration. Missing values are recorded as `NaN`.

## `reports/all_methods_histories.csv`

This is the single source of truth for postprocessing and method comparisons.

Each row corresponds to one method at one iteration.

### Columns

- `method`
  - Method name written by the experiment runner.

- `iteration`
  - Iteration index for that method.

- `total_error`
  - Integrated squared solution error on the fixed reference mesh.
  - This is the raw quantity `∫Ω (u-u_ref)^2 dΩ`.

- `relative_l2_error`
  - Relative L2 solution error on the fixed reference mesh.
  - Computed as `sqrt( total_error / ∫Ω u_ref^2 dΩ )`.
  - This is the preferred headline solution-error metric for interpretation.

- `total_error_rms`
  - RMS solution error at reference-mesh vertices.

- `relative_error_rms`
  - RMS solution error normalized by the reference-solution RMS at the same reference-mesh vertices.
  - This is the easiest scale-free counterpart to `total_error_rms`.

- `boundary_error`
  - Integrated boundary error on the reference mesh.

- `fixed_total_residual`
  - Fixed-grid integrated residual on the common reference mesh.

- `relative_fixed_l2_residual`
  - Relative fixed-grid L2 residual on the common reference mesh.
  - Computed as `sqrt( fixed_total_residual / ∫Ω f^2 dΩ )` when the PDE source term `f` is available.
  - This is the preferred headline residual metric for interpretation.

- `fixed_boundary_residual`
  - Fixed-grid boundary residual on the common reference mesh when available.

- `fixed_rms_residual`
  - RMS residual on the common reference mesh.

- `relative_fixed_rms_residual`
  - RMS residual on the common reference mesh normalized by the RMS of the source term on the same reference mesh vertices.
  - This is the scale-free counterpart to `fixed_rms_residual`.

- `point_count`
  - Number of collocation points used during that iteration.
  - This is intended for error-vs-point-budget comparisons.

- `iteration_runtime_sec`
  - Wall-clock runtime for the full iteration.
  - Includes method-specific point selection or mesh refinement, training, and evaluation.

- `cumulative_runtime_sec`
  - Cumulative wall-clock runtime through that iteration using the same full-iteration scope.

### Suggested Usage

- For main comparison plots:
  - prefer `relative_l2_error` or `relative_error_rms` against iteration, point count, and cumulative runtime;
  - keep `total_error` and `total_error_rms` for continuity and debugging.

- For residual plots:
  - prefer `relative_fixed_l2_residual` or `relative_fixed_rms_residual`;
  - keep `fixed_total_residual` and `fixed_rms_residual` as supporting raw measures.

### Not Included On Purpose

- Training residuals measured on each method's own evolving collocation set are intentionally excluded from the canonical history CSV.
- They are method-local diagnostics, not fair cross-method comparison metrics.
