"""
Path manager for per-run output directories.

Creates outputs/<run-id>/ with standard subfolders and exposes helpers
to access images, reports, and checkpoints directories at runtime.

Also provides a utility to write run metadata (configs, system, git) to JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Optional

from config import (
    PROJECT_ROOT,
    DIRECTORY as DEFAULT_IMAGES,
    REPORTS_DIRECTORY as DEFAULT_REPORTS,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    MESH_CONFIG,
    RANDOM_CONFIG,
    VIZ_CONFIG,
)


OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")
ACTIVE_RUN: Optional[Dict[str, str]] = None


def generate_run_id(tag: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{tag}" if tag else ts


def get_paths(run_id: str) -> Dict[str, str]:
    root = os.path.join(OUTPUTS_ROOT, run_id)
    paths = {
        "root": root,
        "images": os.path.join(root, "images"),
        "reports": os.path.join(root, "reports"),
        "checkpoints": os.path.join(root, "checkpoints"),
        "artifacts": os.path.join(root, "artifacts"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def set_active_run(run_id: str) -> Dict[str, str]:
    global ACTIVE_RUN
    ACTIVE_RUN = get_paths(run_id)
    return ACTIVE_RUN


def images_dir() -> str:
    return ACTIVE_RUN["images"] if ACTIVE_RUN else DEFAULT_IMAGES


def reports_dir() -> str:
    return ACTIVE_RUN["reports"] if ACTIVE_RUN else DEFAULT_REPORTS


def checkpoints_dir() -> str:
    if ACTIVE_RUN:
        return ACTIVE_RUN["checkpoints"]
    # Fallback default when no active run; co-locate with reports' parent
    parent = os.path.dirname(DEFAULT_REPORTS)
    path = os.path.join(parent, "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path


def _get_git_info() -> Dict[str, str]:
    info = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": "unknown",
    }
    try:
        # Get commit
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            info["commit"] = res.stdout.strip()
        # Get branch
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            info["branch"] = res.stdout.strip()
        # Dirty state
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            info["dirty"] = "yes" if res.stdout.strip() else "no"
    except Exception:
        pass
    return info


def write_run_metadata(
    extra: Optional[Dict] = None, filename: str = "run_config.json"
) -> str:
    """Write merged config + system + git metadata to reports directory.

    Returns the path to the written JSON file.
    """
    meta = {
        "timestamp": datetime.now().isoformat(),
        "run_id": ACTIVE_RUN["root"] if ACTIVE_RUN else "default",
        "configs": {
            "model": MODEL_CONFIG,
            "training": TRAINING_CONFIG,
            "mesh": MESH_CONFIG,
            "random": RANDOM_CONFIG,
            "viz": VIZ_CONFIG,
        },
        "git": _get_git_info(),
    }

    # System info (import lazily to avoid circulars)
    try:
        from utils import get_system_info

        meta["system"] = get_system_info()
    except Exception:
        meta["system"] = {}

    if extra:
        meta["extra"] = extra

    out_path = os.path.join(reports_dir(), filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    return out_path
