# Artifacts

This directory stores the curated, paper-facing outputs that should survive beyond individual experiment runs.

- `figures/`: publication figures promoted from `outputs/<run-id>/images/`
- `metrics/`: CSV and JSON summaries used to build tables and reported numbers
- `animations/`: GIF or video assets worth keeping for talks, demos, or supplements

Raw runs remain under `outputs/`. Move only the stable outputs you want to cite in the paper into `artifacts/`, then snapshot them with DVC.
