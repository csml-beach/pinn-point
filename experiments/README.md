# Experiments

This repository already exposes the main experiment entry points through `train/main.py`.

Use this directory for frozen, paper-specific launchers once the manuscript protocol stabilizes. Typical contents are:

- one script for the main method comparison
- one script for ablations or sensitivity studies
- one script for exporting aggregate figures or metrics into `artifacts/`

Keeping paper-facing launchers here prevents the manuscript workflow from depending on ad hoc shell history.
