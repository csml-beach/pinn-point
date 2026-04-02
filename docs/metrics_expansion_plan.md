# Metrics Expansion Plan

Status: implemented on April 2, 2026.

This document now serves as the record of the completed metrics/reporting expansion that landed before hybrid-method work.

## Goal

Add the minimum metric and reporting upgrades needed to make upcoming multi-method comparisons, especially the hybrid adaptive method, scientifically cleaner and easier to analyze.

This plan is intentionally narrow. It focuses on comparison metrics and exports, not on broader performance or refactor work.

## What Is Already Good

These metrics should remain part of the main evaluation stack:

- fixed reference-mesh integrated error
  - current `total_error`
  - use as the main domain-weighted solution-error metric

- fixed reference-mesh RMS error
  - current `total_error_rms`
  - use as the DOF-agnostic error metric

- fixed reference-mesh RMS residual
  - current `fixed_rms_residual`
  - use as the main residual-based comparison metric

These are already conceptually sound because they evaluate all methods on the same fine reference mesh.

## What Is Not Enough Yet

### 1. The generalized multi-method CSV was too thin

Before the implementation, `all_methods_histories.csv` stored only:
- `method`
- `iteration`
- `total_error`
- `boundary_error`
- `point_count`

That was not enough for `adaptive_hybrid_anchor`, because we also needed the fair fixed-grid metrics and runtime.

### 2. Training residual is not a headline comparison metric

Current per-method training residuals are measured on each method's own evolving collocation set or mesh. That is useful for method behavior, but it is not a fair primary comparison across methods.

It should remain available, but it should not be the metric that drives the main paper figures.

### 3. There was no time metric yet

Once the hybrid method adds anchor scoring and local aggregation, comparing only by iteration or point count will be incomplete. We need runtime to judge whether a gain is efficient or just more expensive.

## Metrics To Add

### Required Before Hybrid

These fields were added to the generalized per-method export:

- `total_error`
- `total_error_rms`
- `fixed_total_residual`
- `fixed_rms_residual`
- `point_count`
- `iteration_runtime_sec`
- `cumulative_runtime_sec`

These are the minimum fields I want available for every method before hybrid implementation.

### Useful But Secondary

- `boundary_error`
- `fixed_boundary_residual`
- `total_residual`

These are still useful diagnostics, but they should remain secondary in the comparison story.

### Optional Debug Metric

If we want one additional safeguard for the hybrid method later:

- held-out anchor RMSE

This should only be added if we create a small labeled validation anchor set disjoint from the anchor set used in hybrid scoring. It is a useful debug metric, but not required before implementing hybrid.

## Headline Comparisons

After the metrics upgrade, the default method comparisons should be organized around:

1. error vs iteration
   - use `total_error_rms` or `total_error`

2. error vs point count
   - use `point_count` on the x-axis

3. error vs cumulative runtime
   - use `cumulative_runtime_sec` on the x-axis

For residual comparisons:

- prefer `fixed_rms_residual`
- use `fixed_total_residual` as a supporting integrated metric

## Implemented Changes

### Step 1. Runtime tracking

Implemented:
- start a timer before each train/evaluate iteration
- record iteration time after metrics are computed
- accumulate a cumulative runtime series

This should be method-agnostic and live in the shared experiment runner path.

### Step 2. Model histories

Implemented histories:
- `iteration_runtime_history`
- `cumulative_runtime_history`

Keep the naming parallel to the existing error and residual histories.

### Step 3. Extend `all_methods_histories.csv`

Implemented columns:
- `method`
- `iteration`
- `total_error`
- `total_error_rms`
- `fixed_total_residual`
- `fixed_rms_residual`
- `boundary_error`
- `fixed_boundary_residual`
- `total_residual`
- `point_count`
- `iteration_runtime_sec`
- `cumulative_runtime_sec`

This is the main deliverable of the metrics pass.

### Step 4. Update docs and plotting expectations

Completed:
- `docs/histories_csv.md`
- aggregation scripts and plotting helpers that previously assumed only adaptive vs random

The repo should make it obvious which metrics are:
- fair comparison metrics
- training diagnostics
- optional debug metrics

## Success Criteria

Completed:

- every method writes the same core fair-comparison metrics
- runtime is logged per iteration and cumulatively
- the generalized multi-method CSV is rich enough for hybrid comparisons
- the existing smoke-test path still passes

## Relationship To Hybrid Work

This was completed before the hybrid adaptive method implementation.

Reason:
- once hybrid exists, we will want to compare it immediately against `adaptive`, `random`, and at least one non-adaptive low-discrepancy baseline
- without the richer generalized metrics export, we will create another round of cleanup work right after implementation
