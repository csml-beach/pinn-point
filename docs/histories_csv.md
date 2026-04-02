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
  - Integrated solution error on the fixed reference mesh.

- `total_error_rms`
  - RMS solution error at reference-mesh vertices.

- `boundary_error`
  - Integrated boundary error on the reference mesh.

- `total_residual`
  - Training residual measured on the method's own current collocation mesh or points.
  - Treat this primarily as a method-behavior diagnostic, not as the main fair comparison metric.

- `fixed_total_residual`
  - Fixed-grid integrated residual on the common reference mesh.

- `fixed_boundary_residual`
  - Fixed-grid boundary residual on the common reference mesh when available.

- `fixed_rms_residual`
  - RMS residual on the common reference mesh.
  - This is the preferred residual-based comparison metric across methods.

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
  - use `total_error` or `total_error_rms` against iteration, point count, and cumulative runtime.

- For residual plots:
  - prefer `fixed_rms_residual`;
  - use `fixed_total_residual` as a supporting integrated measure.

- For method diagnostics:
  - inspect `total_residual` to understand how each method behaves on its own training points or mesh.
