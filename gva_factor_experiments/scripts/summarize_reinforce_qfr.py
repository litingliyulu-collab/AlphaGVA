#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


METRIC_COLS = [
    "timestep",
    "episode",
    "formula_eval_cnt",
    "sample_formula_eval_cnt",
    "baseline_formula_eval_cnt",
    "pool/eval_cnt",
    "pool/best_ic_ret",
    "valid/ic_mean",
    "valid/rank_ic_mean",
    "test/ic_mean",
    "test/rank_ic_mean",
]


def method_seed_from_path(path: Path) -> tuple[str, int]:
    method_seed = path.parts[-4]
    method, seed_raw = method_seed.rsplit("_s", 1)
    return method, int(seed_raw)


def read_latest(root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(root.glob("*/results/*/metrics.csv")):
        method, seed = method_seed_from_path(metrics_path)
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        row = df.iloc[-1].to_dict()
        row.update({"method": method, "seed": seed, "run": metrics_path.parent.name})
        rows.append(row)
    return pd.DataFrame(rows)


def read_eval_aligned(root: Path) -> pd.DataFrame:
    frames = []
    for metrics_path in sorted(root.glob("*/results/*/metrics.csv")):
        method, seed = method_seed_from_path(metrics_path)
        df = pd.read_csv(metrics_path)
        if df.empty or "formula_eval_cnt" not in df.columns:
            continue
        df = df.copy()
        df["method"] = method
        df["seed"] = seed
        df["run"] = metrics_path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    target = int(all_df.groupby("method")["formula_eval_cnt"].max().min())
    rows = []
    for (method, seed), group in all_df.groupby(["method", "seed"]):
        eligible = group[group["formula_eval_cnt"] <= target]
        if eligible.empty:
            eligible = group.iloc[[0]]
        row = eligible.iloc[-1].to_dict()
        row["aligned_formula_eval_cnt"] = target
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"__{title}__")
    if df.empty:
        print("EMPTY")
        return
    cols = ["method", "seed", "run"] + [c for c in METRIC_COLS if c in df.columns]
    if "aligned_formula_eval_cnt" in df.columns:
        cols.insert(3, "aligned_formula_eval_cnt")
    print(df[cols].sort_values(["method", "seed"]).to_string(index=False))
    print(f"__{title}_MEAN_STD__")
    agg_cols = [c for c in ["test/ic_mean", "test/rank_ic_mean", "pool/best_ic_ret", "formula_eval_cnt"] if c in df.columns]
    print(df.groupby("method")[agg_cols].agg(["mean", "std"]).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    print_table("FINAL_BY_STEP", read_latest(root))
    print_table("ALIGNED_BY_FORMULA_EVAL", read_eval_aligned(root))


if __name__ == "__main__":
    main()

