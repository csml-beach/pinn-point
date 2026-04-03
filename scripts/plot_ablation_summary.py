#!/usr/bin/env python3
"""Aggregate ablation runs and generate summary plots.

Reads outputs/*/reports/all_methods_histories.csv and writes plots to
outputs/ablation_summary/.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


def _parse_args():
    parser = argparse.ArgumentParser(description="Aggregate ablation run CSVs")
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Root outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--run-id-contains",
        default="ablation-",
        help=(
            "Only include runs whose output directory name contains this token "
            "(default: ablation-). Set empty string to include all runs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outputs = Path(args.outputs)
    csvs = list(outputs.glob("*/reports/all_methods_histories.csv"))
    if args.run_id_contains:
        csvs = [p for p in csvs if args.run_id_contains in p.parents[1].name]
    if not csvs:
        print(
            "No matching all_methods_histories.csv found in "
            f"{outputs}/*/reports (filter={args.run_id_contains!r})"
        )
        return 1
    print(f"Including {len(csvs)} run(s) for aggregation")

    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        df["run_id"] = p.parents[1].name
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=["total_error"])
    if data.empty:
        print("No valid total_error entries found after dropping NaNs")
        return 1
    out_dir = outputs / "ablation_summary"
    out_dir.mkdir(exist_ok=True)

    stats = (
        data.groupby(["method", "iteration"]).agg(
            mean_err=("total_error", "mean"),
            std_err=("total_error", "std"),
            mean_pts=("point_count", "mean"),
        )
    ).reset_index()

    # 1) Error convergence (mean ± std)
    plt.figure(figsize=(10, 6))
    for method in sorted(stats["method"].unique()):
        m = stats[stats["method"] == method]
        plt.plot(m["iteration"], m["mean_err"], label=method)
        plt.fill_between(
            m["iteration"],
            m["mean_err"] - m["std_err"],
            m["mean_err"] + m["std_err"],
            alpha=0.2,
        )
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Total Error")
    plt.title("Error Convergence (mean ± std across seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_error_convergence.png", dpi=150)
    plt.close()

    # 2) Final error bar chart
    last_iters = (
        data.sort_values("iteration").groupby(["run_id", "method"]).tail(1)
    )
    final_stats = last_iters.groupby("method").agg(
        mean_final=("total_error", "mean"),
        std_final=("total_error", "std"),
    ).reset_index()

    plt.figure(figsize=(9, 5))
    plt.bar(
        final_stats["method"],
        final_stats["mean_final"],
        yerr=final_stats["std_final"],
    )
    plt.yscale("log")
    plt.ylabel("Final Total Error")
    plt.title("Final Error by Method (mean ± std)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_final_error.png", dpi=150)
    plt.close()

    # 3) Error vs point count (mean trajectory)
    plt.figure(figsize=(10, 6))
    for method in sorted(stats["method"].unique()):
        m = stats[stats["method"] == method]
        plt.plot(m["mean_pts"], m["mean_err"], label=method)
    plt.yscale("log")
    plt.xlabel("Point Count (mean)")
    plt.ylabel("Total Error (mean)")
    plt.title("Error vs Point Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_error_vs_points.png", dpi=150)
    plt.close()

    print(f"Saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
