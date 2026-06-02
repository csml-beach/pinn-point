# Remote AI Handoff: Paper Results and DVC Setup

This note is for continuing the paper-writing workflow after cloning the Git
repository on another machine.

## Current DVC Storage

The local DVC remote used on the source machine is:

```text
/Users/arash/Documents/GitHub/pinn-point-dvc-storage
```

It is about 2.3 GB and should be backed up or copied to the remote machine.
The repository DVC config currently defines this remote as:

```text
localstore = ../../pinn-point-dvc-storage
```

This relative path is resolved from `.dvc/config`.  The easiest layout on the
remote machine is therefore:

```text
<workspace>/
  pinn-point/
  pinn-point-dvc-storage/
```

With this layout, the existing DVC remote should work without modification.

## After Git Clone

From the remote machine:

```bash
git clone https://github.com/csml-beach/pinn-point.git
cd pinn-point
git checkout main
```

Install or activate an environment with DVC.  Then check the configured remote:

```bash
dvc remote list
dvc status
```

If the DVC storage is adjacent to the repo as shown above, pull the data:

```bash
dvc pull
dvc checkout
```

If the DVC storage is somewhere else, update the remote path first:

```bash
dvc remote modify localstore url /absolute/path/to/pinn-point-dvc-storage
dvc pull
dvc checkout
```

## Data Sources for the Paper

Use these sources when writing the results section:

- `outputs.dvc`: tracks the full `outputs/` directory containing the synchronized
  experiment runs.
- `artifacts/snapshots/fair-halton-paper-reruns-ac-ns-ad-2026-05-03.dvc`:
  tracks the paper snapshot for the fair-Halton reruns of AC, NS, and AD.
- `artifacts/metrics/maxwell3d_e300i8_ms030_ref010_10seed/`: Git-tracked
  Maxwell 3D 10-seed summary CSVs.
- `artifacts/metrics/maxwell3d_e300i8_ms030_ref010_20seed/`: Git-tracked
  Maxwell 3D 20-seed summary CSVs.
- `paper/figures/`: Git-tracked paper figures, including the residual-scaffold
  method diagrams.
- `artifacts/figures/maxwell3d/`: Git-tracked Maxwell 3D figure drafts.

## Paper Files to Continue Editing

The manuscript is under `paper/`.  The current active section is:

```text
paper/method.tex
```

Label bookkeeping is in:

```text
paper/labels.md
```

Do not run `latexmk` unless the user explicitly asks to compile.
