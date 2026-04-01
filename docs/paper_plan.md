# Paper Plan

## Repo Summary

- The repository studies how collocation-point selection changes PINN performance on a 2D Poisson problem posed on a complex geometry.
- The main experiment pipeline couples an NGSolve FEM reference solve with a PINN training loop and compares adaptive mesh refinement against random and residual-based sampling baselines.
- The current codebase already supports `adaptive`, `random`, `halton`, `sobol`, `random_r`, and `rad` as training methods.

## Evidence Base

- Project overview and CLI usage: `README.md`
- System structure and fairness constraints: `docs/ARCHITECTURE.md`
- Hyperparameter and ablation workflow: `docs/parameter_study.md`, `docs/ablation_study_plan.md`
- Literature notes for residual-based sampling: `docs/sampling_methods_paper.md`
- Entry point and orchestration: `train/main.py`, `train/experiments.py`
- Method implementations: `train/methods/*.py`
- PDE and FEM setup: `train/problems/poisson.py`, `train/fem_solver.py`, `train/geometry.py`

## Working Claim

Provisional claim for the paper:

Residual-aware collocation strategies improve PINN error convergence relative to fixed random sampling, and mesh-aware adaptive refinement is especially compelling on complex 2D geometries where geometric fidelity matters. This claim still needs curated seed-aggregated evidence from this repository before it should appear as a finalized paper statement.

## Section Plan

| Section | Goal | Key source files | Open questions | Status |
| --- | --- | --- | --- | --- |
| Abstract | State the problem, methods compared, main finding, and scope in 4-6 sentences. | `README.md`, `docs/ARCHITECTURE.md`, curated results in `artifacts/metrics/` | Which methods make the final comparison set? What is the single headline metric? | planned |
| Introduction | Motivate collocation-point selection for PINNs and explain why complex geometry changes the tradeoff. | `README.md`, `docs/ARCHITECTURE.md`, `docs/sampling_methods_paper.md` | Which related-work thread should lead: adaptive sampling, mesh refinement, or complex-geometry PINNs? | planned |
| Method | Describe the Poisson problem, FEM supervision, PINN architecture, and each sampling strategy. | `train/problems/poisson.py`, `train/fem_solver.py`, `train/pinn_model.py`, `train/training.py`, `train/methods/*.py` | What exact loss terms and hyperparameters are stable enough to report? | planned |
| Experiments | Lock the benchmark protocol, seeds, metrics, figures, and ablations. | `train/main.py`, `train/experiments.py`, `docs/parameter_study.md`, `docs/ablation_study_plan.md` | Which seed set, point-budget policy, and compute budget are final? | planned |
| Conclusions | Summarize supported findings, limitations, and future extensions. | Final curated figures/tables plus experiment logs | Which limitations should be explicit: single PDE, single geometry family, or compute cost? | planned |
| References | Track canonical PINN, adaptive sampling, and FEM comparison citations. | `docs/sampling_methods_paper.md`, `literature/Residual-based-sampling.pdf`, `paper/references.bib` | Which prior adaptive/PINN papers are mandatory beyond Wu et al.? | planned |

## Experiment Queue

- [ ] Freeze the primary method comparison set for the paper.
- [ ] Choose the reporting metric(s): final error, convergence rate, and/or error at matched point budget.
- [ ] Curate paper-facing figures from `outputs/<run-id>/` into `artifacts/figures/`.
- [ ] Export aggregated tables and CSV summaries into `artifacts/metrics/`.
- [ ] Save any presentation-quality GIFs or trajectory animations into `artifacts/animations/`.
- [ ] Record one reproducible command per figure/table in `experiments/`.
- [ ] Add final citation metadata to `paper/references.bib`.

## Iteration Loop

1. Run or rerun experiments with `python3 train/main.py ...` or a frozen launcher in `experiments/`.
2. Promote the small set of paper-ready outputs from `outputs/` into `artifacts/`.
3. Use `scripts/snapshot_research.sh --dry-run` to preview a Git plus DVC checkpoint.
4. Update `paper/labels.md` and the relevant LaTeX section once a figure/table becomes stable.

## Initial Drafting Guardrails

- Keep `outputs/` as raw per-run experiment storage.
- Keep `artifacts/` for curated figures, metrics, and animations that are worth versioning for the manuscript.
- Avoid claiming that adaptive methods outperform all baselines until the repo-generated aggregates are written into `artifacts/metrics/`.
