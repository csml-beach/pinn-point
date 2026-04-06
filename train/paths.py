"""
Path manager for per-run output directories.

Creates outputs/<run-id>/ with standard subfolders and exposes helpers
to access images, reports, and checkpoints directories at runtime.

Also provides utilities to organize comparison outputs, per-method outputs,
and run metadata/manifests.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional

from config import (
    PROJECT_ROOT,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    MESH_CONFIG,
    HYBRID_ADAPTIVE_CONFIG,
    RANDOM_CONFIG,
    RUNTIME_CONFIG,
    VIZ_CONFIG,
)


OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")
ACTIVE_RUN: Optional[Dict[str, str]] = None
DEFAULT_RUN_ID = "_default"


def generate_run_id(tag: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{tag}" if tag else ts


def get_paths(run_id: str) -> Dict[str, str]:
    root = os.path.join(OUTPUTS_ROOT, run_id)
    paths = {
        "root": root,
        "images": os.path.join(root, "images"),
        "images_comparison": os.path.join(root, "images", "comparison"),
        "images_methods": os.path.join(root, "images", "methods"),
        "reports": os.path.join(root, "reports"),
        "reports_methods": os.path.join(root, "reports", "methods"),
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


@lru_cache(maxsize=1)
def _default_paths() -> Dict[str, str]:
    return get_paths(DEFAULT_RUN_ID)


def images_dir() -> str:
    return ACTIVE_RUN["images"] if ACTIVE_RUN else _default_paths()["images"]


def comparison_images_dir() -> str:
    if ACTIVE_RUN:
        return ACTIVE_RUN["images_comparison"]
    return _default_paths()["images_comparison"]


def method_images_root_dir() -> str:
    if ACTIVE_RUN:
        return ACTIVE_RUN["images_methods"]
    return _default_paths()["images_methods"]


def method_images_dir(method_name: str) -> str:
    path = os.path.join(method_images_root_dir(), method_name)
    os.makedirs(path, exist_ok=True)
    return path


def reports_dir() -> str:
    return ACTIVE_RUN["reports"] if ACTIVE_RUN else _default_paths()["reports"]


def method_reports_root_dir() -> str:
    if ACTIVE_RUN:
        return ACTIVE_RUN["reports_methods"]
    return _default_paths()["reports_methods"]


def method_reports_dir(method_name: str) -> str:
    path = os.path.join(method_reports_root_dir(), method_name)
    os.makedirs(path, exist_ok=True)
    return path


def checkpoints_dir() -> str:
    if ACTIVE_RUN:
        return ACTIVE_RUN["checkpoints"]
    return _default_paths()["checkpoints"]


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
            "hybrid_adaptive": HYBRID_ADAPTIVE_CONFIG,
            "random": RANDOM_CONFIG,
            "runtime": RUNTIME_CONFIG,
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


def write_run_manifest(
    *,
    methods: Optional[list[str]] = None,
    extra: Optional[Dict] = None,
    filename: str = "run_manifest.json",
) -> str:
    """Write a run manifest describing the output layout for this run."""
    paths = ACTIVE_RUN if ACTIVE_RUN else _default_paths()
    methods = list(methods or [])
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "run_root": paths["root"],
        "directories": {
            "images_root": paths["images"],
            "comparison_images": paths["images_comparison"],
            "method_images_root": paths["images_methods"],
            "reports_root": paths["reports"],
            "method_reports_root": paths["reports_methods"],
            "checkpoints_root": paths["checkpoints"],
            "artifacts_root": paths["artifacts"],
        },
        "canonical_reports": {
            "run_config": os.path.join(paths["reports"], "run_config.json"),
            "run_manifest": os.path.join(paths["reports"], filename),
            "all_methods_histories": os.path.join(
                paths["reports"], "all_methods_histories.csv"
            ),
            "performance_summary": os.path.join(
                paths["reports"], "performance_summary.txt"
            ),
            "point_usage_table": os.path.join(
                paths["reports"], "point_usage_table.txt"
            ),
        },
        "methods": {
            method_name: {
                "images_dir": os.path.join(paths["images_methods"], method_name),
                "reports_dir": os.path.join(paths["reports_methods"], method_name),
                "history_csv": os.path.join(
                    paths["reports_methods"], method_name, "history.csv"
                ),
                "iteration_diagnostics_csv": os.path.join(
                    paths["reports_methods"], method_name, "iteration_diagnostics.csv"
                ),
                "diagnostics_json": os.path.join(
                    paths["reports_methods"], method_name, "diagnostics.json"
                ),
            }
            for method_name in methods
        },
        "git": _get_git_info(),
    }
    if extra:
        manifest["extra"] = extra

    out_path = os.path.join(reports_dir(), filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path
