#!/usr/bin/env python3
"""Paired significance test for seed-matched method comparisons.

Computes paired sign-flip permutation p-values and bootstrap CI for the mean
paired difference on a chosen metric column.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Path to per-seed analysis CSV")
    p.add_argument("--metric", required=True, help="Metric column name")
    p.add_argument("--method-a", required=True, help="First method name")
    p.add_argument("--method-b", required=True, help="Second method name")
    p.add_argument(
        "--seed-col", default="seed", help="Seed column name (default: seed)"
    )
    p.add_argument(
        "--method-col", default="method", help="Method column name (default: method)"
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=20000,
        help="Bootstrap samples for CI (default: 20000)",
    )
    p.add_argument(
        "--lower-better",
        action="store_true",
        help="Interpret lower metric values as better (default: true behavior in report text only)",
    )
    return p.parse_args()


def load_differences(
    csv_path: Path, seed_col: str, method_col: str, metric: str, method_a: str, method_b: str
) -> tuple[list[float], dict[int, tuple[float, float]]]:
    rows = list(csv.DictReader(csv_path.open()))
    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        method = row[method_col]
        if method not in (method_a, method_b):
            continue
        seed = int(float(row[seed_col]))
        by_seed.setdefault(seed, {})[method] = float(row[metric])

    paired: dict[int, tuple[float, float]] = {}
    diffs: list[float] = []
    for seed in sorted(by_seed):
        vals = by_seed[seed]
        if method_a in vals and method_b in vals:
            a = vals[method_a]
            b = vals[method_b]
            paired[seed] = (a, b)
            diffs.append(b - a)
    return diffs, paired


def exact_sign_flip_pvalues(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    observed = sum(diffs) / n
    count = 1 << n
    extreme = 0
    one_sided = 0
    for mask in range(count):
        s = 0.0
        for i, x in enumerate(diffs):
            s += x if ((mask >> i) & 1) == 0 else -x
        m = s / n
        if abs(m) >= abs(observed) - 1e-15:
            extreme += 1
        if m >= observed - 1e-15:
            one_sided += 1
    return extreme / count, one_sided / count


def bootstrap_ci(diffs: list[float], n_bootstrap: int) -> tuple[float, float]:
    n = len(diffs)
    rng = random.Random(0)
    means = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_bootstrap)]
    hi = means[int(0.975 * n_bootstrap)]
    return lo, hi


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    diffs, paired = load_differences(
        csv_path,
        args.seed_col,
        args.method_col,
        args.metric,
        args.method_a,
        args.method_b,
    )
    if not diffs:
        raise SystemExit("No paired seeds found for requested methods.")
    if len(diffs) > 22:
        raise SystemExit(
            "Exact sign-flip enumeration is capped here for practicality (n<=22)."
        )

    n = len(diffs)
    mean_a = sum(a for a, _ in paired.values()) / n
    mean_b = sum(b for _, b in paired.values()) / n
    mean_diff = sum(diffs) / n
    p_two, p_one = exact_sign_flip_pvalues(diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs, args.bootstrap)

    print(f"csv={csv_path}")
    print(f"metric={args.metric}")
    print(f"method_a={args.method_a}")
    print(f"method_b={args.method_b}")
    print(f"n={n}")
    print(f"mean_{args.method_a}={mean_a}")
    print(f"mean_{args.method_b}={mean_b}")
    print(f"mean_diff_{args.method_b}_minus_{args.method_a}={mean_diff}")
    print(f"perm_p_two_sided={p_two}")
    print(f"perm_p_one_sided={p_one}")
    print(f"bootstrap95_mean_diff=[{ci_lo}, {ci_hi}]")
    print(f"seeds={','.join(str(s) for s in sorted(paired))}")


if __name__ == "__main__":
    main()
