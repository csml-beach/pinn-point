# Hybrid Adaptive Method Plan

Status: implemented on April 2, 2026 as `adaptive_hybrid_anchor`.

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

Implemented:
- sample fixed anchor points uniformly from the domain interior using rejection sampling on the initial mesh
- label them once using the initial FEM solution
- keep the anchor set fixed across all iterations

### 2. Local Data-Error Indicator

For each anchor point `(x_i, y_i, u_i)`:
- evaluate the PINN prediction `u_hat_i`
- compute pointwise squared error `e_i = (u_hat_i - u_i)^2`

Map pointwise errors to mesh elements:
- start with nearest-element or containing-element averaging
- aggregate point errors per element to get `E(K)`

Implemented choice:
- use containing-element assignment through `mesh(x, y).nr`
- average pointwise anchor errors within each current mesh element to form `E(K)`

### 3. Local Residual Indicator

Reuse the current residual-based element indicator machinery to produce `R(K)`.

Important:
- for the hybrid method, residuals should be normalized explicitly
- the current `threshold * maxerr` rule is fine for residual-only refinement, but not sufficient when blending channels

### 4. Normalization

Normalize both channels robustly per iteration.

Implemented normalization rule:
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

Current default parameters:
- `alpha = 1.0`
- `beta = 1.0`
- `tau = current refinement threshold`

These should later become method-specific configuration knobs.

## Implementation Record

Completed:
1. Added `HYBRID_ADAPTIVE_CONFIG` for anchor count, blend weights, and normalization quantile.
2. Implemented `train/methods/hybrid_anchor.py` and registered `adaptive_hybrid_anchor`.
3. Added fixed-anchor initialization from the initial FEM solution.
4. Added containing-element aggregation of anchor point errors to elementwise `E(K)`.
5. Added robust per-iteration normalization for both residual and anchor-error channels.
6. Routed the new method through the mesh-refinement experiment path without touching the dense reference mesh.
7. Added per-iteration hybrid refinement stats on the trained model.

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

## Current Risks

- backend fragility in element lookup / assignment for anchor points
- data-error field may be too sparse or noisy if anchor coverage is weak
- blended score may over-prioritize labeled regions and under-refine PDE-difficult regions
- the current implementation stores summary stats for the hybrid channels, but not full per-element diagnostic dumps

## Next Analysis Step

Run multi-seed comparisons against `adaptive`, `random`, and at least one low-discrepancy baseline to see whether the hybrid scoring improves error versus point count and runtime.
