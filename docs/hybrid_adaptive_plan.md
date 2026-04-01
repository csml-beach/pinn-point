# Hybrid Adaptive Method Plan

## Goal

Add a new adaptive method that refines the mesh using a blended local score:

`score(K) = alpha * R_norm(K) + beta * E_norm(K)`

where:
- `R_norm(K)` is a normalized local residual indicator on element `K`
- `E_norm(K)` is a normalized local supervised error indicator on element `K`

The method should use a fixed anchor set of labeled FEM pairs for scoring and must not use the dense reference mesh that is reserved for evaluation.

## Why This Variant

The current adaptive method is residual-only. That is a clean primary method, but it can miss regions where the PINN prediction error is still high even when the PDE residual is locally modest.

A hybrid method is worth testing because:
- residual indicates PDE inconsistency
- anchor-point supervised error indicates mismatch to trusted labeled data
- the combination may refine regions that are important for solution accuracy but not obvious from residual alone

## Constraints

- Do not use the dense reference FEM mesh or reference error field for refinement.
- Keep the current residual-only `adaptive` method unchanged as the baseline adaptive method.
- Use a separate fixed anchor set of FEM-labeled points created once per run.
- Keep evaluation against the existing fine reference mesh unchanged.

## Proposed Method

### 1. Anchor Set

Create a fixed anchor set of labeled FEM pairs:
- sample points once at the start of the run
- solve/evaluate FEM at those points
- keep this anchor set fixed across all iterations

Initial implementation:
- start with anchor points sampled from the initial mesh or domain interior
- use a moderate anchor count so scoring is informative but not too expensive

### 2. Local Data-Error Indicator

For each anchor point `(x_i, y_i, u_i)`:
- evaluate the PINN prediction `u_hat_i`
- compute pointwise squared error `e_i = (u_hat_i - u_i)^2`

Map pointwise errors to mesh elements:
- start with nearest-element or containing-element averaging
- aggregate point errors per element to get `E(K)`

First implementation choice:
- use containing-element assignment if backend access is stable
- otherwise use nearest-vertex / nearest-element fallback with a clear TODO

### 3. Local Residual Indicator

Reuse the current residual-based element indicator machinery to produce `R(K)`.

Important:
- for the hybrid method, residuals should be normalized explicitly
- the current `threshold * maxerr` rule is fine for residual-only refinement, but not sufficient when blending channels

### 4. Normalization

Normalize both channels robustly per iteration.

Initial normalization rule:
- compute `q95(R)` and `q95(E)`
- clip each field at its 95th percentile
- divide by the clipped scale plus epsilon

Concretely:
- `R_norm(K) = min(R(K), q95(R)) / (q95(R) + eps)`
- `E_norm(K) = min(E(K), q95(E)) / (q95(E) + eps)`

Rationale:
- avoids domination by one outlier element
- puts residual and data-error fields onto comparable scales
- makes `alpha` and `beta` interpretable

### 5. Blended Refinement Score

Use an additive score:

`score(K) = alpha * R_norm(K) + beta * E_norm(K)`

Refine elements with:

`score(K) > tau * max(score)`

Initial parameters:
- `alpha = 1.0`
- `beta = 1.0`
- `tau = current refinement threshold`

These should later become method-specific configuration knobs.

## Implementation Steps

1. Add a plan-level config for the hybrid method:
   - anchor count
   - `alpha`
   - `beta`
   - quantile for normalization

2. Add a helper to build the fixed anchor set once per run.

3. Add a helper to compute pointwise anchor errors from the current PINN.

4. Add a helper to aggregate anchor errors into elementwise `E(K)`.

5. Add a helper to normalize elementwise indicators robustly.

6. Implement a new method, likely `adaptive_hybrid_anchor`, without changing the existing `adaptive` method.

7. Log the residual and data-error channels separately so the behavior is inspectable.

8. Compare against:
   - `adaptive` residual-only
   - `random`
   - one low-discrepancy baseline such as `halton` or `sobol`

## Validation Plan

Before larger runs:
- ensure the new method passes the smoke-test path
- verify that anchor sets are fixed across iterations
- verify that the dense reference mesh is used only for evaluation
- inspect per-iteration statistics for `R(K)`, `E(K)`, and blended score

Primary analysis:
- error vs iteration
- error vs collocation-point count
- wall-clock time

Useful ablations:
- residual-only
- data-error-only
- hybrid with equal weights
- hybrid with `alpha > beta`

## Risks

- backend fragility in element lookup / assignment for anchor points
- data-error field may be too sparse or noisy if anchor coverage is weak
- blended score may over-prioritize labeled regions and under-refine PDE-difficult regions
- current adaptive fine-tuning asymmetry should still be cleaned up for fair comparisons

## Immediate Next Step

Take a repository snapshot with this plan committed, then implement the hybrid method on top of that checkpoint.
