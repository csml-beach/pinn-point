# Lean Baseline Protocol

Status: recommended fast-iteration protocol as of April 5, 2026.

## Goal

Define a lighter experiment profile that still runs full end-to-end training sessions, writes the canonical reports, and preserves the main scientific comparison logic.

This protocol is for:
- method iteration
- parameter tuning
- quick multi-seed screening
- reproducing bugs without paying the cost of the heaviest production runs

This protocol is not the final paper setting. It is the default "inner loop" benchmark.

## Design Principles

- Keep the PDE, loss structure, and fair-comparison logic unchanged.
- Reduce the main cost drivers first: optimizer cost, training budget, reference-mesh cost, and geometry complexity.
- Use a small but meaningful method set during iteration.
- Treat lean runs as ranking and debugging tools, then promote only promising settings to heavier confirmation runs.

## Recommended Lean Settings

These settings are the recommended default for fast iteration.

### Training budget

```python
TRAINING_CONFIG = {
    "optimizer": "Adam",
    "epochs": 100,
    "iterations": 4,
}
```

Rationale:
- `Adam` is much cheaper than `LBFGS` for method iteration.
- `100 x 4` is large enough to remain a real training run, but small enough to run often.

### Mesh and evaluation

```python
MESH_CONFIG = {
    "maxh": 0.7,
    "reference_mesh_factor": 0.05,
}
```

Rationale:
- the initial mesh is coarser than the current default
- the reference mesh remains fixed and fair across methods, but is much cheaper than the production setting

### Model-side batch sizes and supervision weight

```python
MODEL_CONFIG = {
    "hidden_size": 64,
    "num_data": 128,
    "num_bd": 1000,
    "w_data": 0.5,
}
```

Rationale:
- `64` hidden units improved screening accuracy substantially without materially changing local iteration cost
- the coarse labeled data still anchors training, but it no longer dominates the residual term as strongly
- the reduced label batch keeps the fixed-data paradigm intact while giving residual point placement more influence

### Geometry simplification

Keep the same square-domain scale and the same Poisson problem, but simplify the perforated geometry.

```python
GEOMETRY_CONFIG = {
    "domain_size": 5,
    "grid_n": 3,
    "cell_fill": 0.45,
    "circle_radius": 0.7,
}
```

Rationale:
- preserves a nontrivial geometry
- removes much of the fine internal structure that makes meshing, FEM solves, and point rejection sampling expensive

## Recommended Method Sets

Do not use the full method suite during inner-loop development.

### Default fast comparison

Use:
- `adaptive`
- `random`
- `halton`

Why:
- `random` is the minimal baseline
- `halton` is a stronger non-adaptive baseline
- `adaptive` is the main residual-guided method of interest under the fixed collocation budget

### Hybrid-method iteration

Use:
- `adaptive_hybrid_anchor`
- `adaptive`
- `random`

Additional lean setting:

```python
HYBRID_ADAPTIVE_CONFIG = {
    "anchor_count": 128,
    "beta": 0.5,
    "refinement_threshold": 0.9,
}
```

Use `256` if the method becomes too noisy at `128`.
Use the lower `beta` and higher hybrid-specific threshold to keep point growth under control during iteration.

### Screening default

For the stronger `screen` profile, use:
- `adaptive`
- `random`
- `halton`
- `rad`

Reason:
- `adaptive` is now a fixed-budget residual-guided sampler
- `rad` is the closest residual-only continuous-space competitor
- `adaptive_hybrid_anchor` remains available, but it is no longer part of the default mainline comparison because it introduces extra interior supervised anchor labels

## Seed Policy

Use a staged seed policy instead of running a large seed suite every time.

- development: `1` seed
- screening: `3` seeds
- final confirmation: larger frozen seed set

Recommended progression:
1. Run one seed while changing code or method parameters.
2. Once a setting looks promising, rerun with three seeds.
3. Only promote clear winners to the heavy paper-facing runs.

## What Counts As A Lean Baseline Run

A run should still:
- train each included method end to end
- use the shared reference solution and common evaluation mesh
- write the canonical CSV and summaries
- preserve the same fair-comparison budget policy across methods

A run should not:
- export heavy visualization bundles by default
- create GIFs
- include every method in the repository
- use the finest reference mesh reserved for final reporting

## Suggested Run Shapes

### Development

- methods: `adaptive`, `random`, `halton`
- seeds: `1`
- settings: lean defaults above
- exports: off

Use this to test code changes, parameter changes, and method behavior.

CLI:
```bash
python3 train/main.py dev
```

### Screening

- methods: `adaptive`, `random`, `halton`, `rad`
- seeds: `3`
- hidden size: `64`
- optimizer: `Adam`
- iterations: `6`
- epochs: `200`
- reference mesh factor: `0.05`
- exports: off

Use this to decide whether a change is worth promoting.

CLI:
```bash
python3 train/main.py screen
```

Use `--reference-mesh-factor 0.025` only for slower confirmatory reruns after the screen result looks promising.

### Final confirmation

- restore heavier training and evaluation settings
- expand the method set as needed
- use the larger frozen seed set

Use this only after the development and screening phases converge on a small number of candidate settings.

## Interpretation Rules

- Lean runs are for ranking ideas and identifying obvious failures.
- Do not make strong paper claims from lean runs alone.
- If a change only helps under the lean protocol and disappears under the heavier protocol, treat it as unconfirmed.
- If a change does not help under the lean protocol, it usually does not deserve a heavy run unless there is a strong scientific reason.

## Immediate Recommendation For This Repo

Use the following as the default fast-iteration benchmark:

- optimizer: `Adam`
- epochs: `100`
- iterations: `4`
- mesh size: `0.7`
- reference mesh factor: `0.05`
- hidden size: `64`
- data points: `128`
- boundary points: `1000`
- data-loss weight: `0.5`
- geometry: `domain_size=5`, `grid_n=3`, `cell_fill=0.45`, `circle_radius=0.7`
- methods: `adaptive`, `random`, `halton`
- seeds: `1` for development, `3` for screening

This is the protocol that should be used to iterate on the main method and reach the primary result faster before launching expensive full-suite runs.
