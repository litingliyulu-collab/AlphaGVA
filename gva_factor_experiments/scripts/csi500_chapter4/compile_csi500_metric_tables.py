#!/usr/bin/env python3
"""Compile CSI500 chapter-4 IC/RankIC tables from remote run roots (test set only)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/root/autodl-tmp/gva_factor_experiments/compiled/csi500_chapter4")
OUT.mkdir(parents=True, exist_ok=True)

MAIN = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_main_compare_20260630_220231")
ACTOR = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_actor_gva_20260630_220231")
FILTER = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_filter_xgb_20260701_003217")
PPO_FILTER = Path(
    "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_ppo_history_filter_20260701_004456"
)
GVA_FILTER = Path(
    "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_gva_filter_rescue_20260701_003217"
)
ML = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_ml_baselines_20260701_003217")
A2C_ROOT = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_rl_advanced_20260630_220231")
RQ_EARLY = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_reinforce_qfr_20260630_220932")
RQ_RESUME = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_reinforce_qfr_resume_20260701_003217")
AFF_BT = Path(
    "/root/autodl-tmp/aff_official_workspace_csi500_20260701_003217/out/affofficial_csi500_csi500_2023_0_mainwindow/aff_official_metrics.csv"
)


def last_row(path: Path) -> pd.Series:
    return pd.read_csv(path).iloc[-1]


def metrics_path(base: Path, subdir: str) -> Path:
    return next((base / subdir / "results").glob("*/metrics.csv"))


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for (group, method), g in df.groupby(["group", "method"], sort=False):
        out.append(
            {
                "group": group,
                "method": method,
                "n": len(g),
                "IC_mean": g["ic"].mean(),
                "IC_std": g["ic"].std(ddof=1) if len(g) > 1 else np.nan,
                "RankIC_mean": g["rank_ic"].mean(),
                "RankIC_std": g["rank_ic"].std(ddof=1) if len(g) > 1 else np.nan,
                "seeds": ",".join(map(str, sorted(g["seed"].tolist()))),
            }
        )
    return pd.DataFrame(out)


rows: list[dict] = []


def add_rl(method: str, group: str, base: Path, pattern: str, seeds=(0, 1, 2)):
    for seed in seeds:
        path = metrics_path(base, pattern.format(seed=seed))
        r = last_row(path)
        rows.append(
            {
                "group": group,
                "method": method,
                "seed": seed,
                "ic": float(r["test/ic_2"]),
                "rank_ic": float(r["test/rank_ic_2"]),
                "path": str(path),
            }
        )


def add_generic(
    method: str,
    group: str,
    paths: list[Path],
    seeds=(0, 1, 2),
    ic_col="test/ic_mean",
    rank_col="test/rank_ic_mean",
):
    for seed, path in zip(seeds, paths):
        if not path.exists():
            print("MISS", method, seed, path)
            continue
        r = last_row(path)
        rows.append(
            {
                "group": group,
                "method": method,
                "seed": seed,
                "ic": float(r[ic_col]),
                "rank_ic": float(r[rank_col]),
                "path": str(path),
            }
        )


add_rl("AlphaGen", "Main RL baselines", MAIN, "ppo_s{seed}")
add_rl("GVA", "Proposed method", MAIN, "full_gva25_s{seed}")
add_rl("Critic-GVA", "Ablation", MAIN, "critic_gva25_s{seed}")
add_rl("Actor-GVA", "Ablation", ACTOR, "actor_gva25_s{seed}")
add_rl("A2C", "Advanced RL baselines", A2C_ROOT, "a2c_baseline_s{seed}")

add_generic(
    "PPO-filter",
    "Filter baselines",
    [metrics_path(PPO_FILTER, f"ppo_filter_s{s}") for s in range(3)],
)
add_generic(
    "GP",
    "Symbolic/ML baselines",
    [metrics_path(FILTER, f"gp_filter_strong_s{s}") for s in range(3)],
)
add_generic(
    "XGBoost",
    "Symbolic/ML baselines",
    [metrics_path(FILTER, f"xgboost_s{s}") for s in range(3)],
)
add_generic(
    "LightGBM",
    "Symbolic/ML baselines",
    [metrics_path(ML, f"lightgbm_s{s}") for s in range(3)],
)
add_generic(
    "MLP",
    "Symbolic/ML baselines",
    [metrics_path(ML, f"mlp_s{s}") for s in range(3)],
)
add_generic(
    "GVA-filter",
    "Ablation",
    [metrics_path(GVA_FILTER, f"gva_filter_s{s}") for s in range(3)],
)

reinforce_paths = [
    metrics_path(RQ_EARLY, "reinforce_s0"),
    metrics_path(RQ_EARLY, "reinforce_s1"),
    metrics_path(RQ_RESUME, "reinforce_s2"),
]
qfr_paths = [
    metrics_path(RQ_EARLY, "qfr_s0"),
    metrics_path(RQ_RESUME, "qfr_s1"),
    metrics_path(RQ_RESUME, "qfr_s2"),
]
add_generic("REINFORCE", "Advanced RL baselines", reinforce_paths)
add_generic("QFR", "Advanced RL baselines", qfr_paths)

if AFF_BT.exists():
    r = last_row(AFF_BT)
    rows.append(
        {
            "group": "Two-stage baselines",
            "method": "AFF",
            "seed": 0,
            "ic": float(r["test/ic_mean"]),
            "rank_ic": float(r["test/rank_ic_mean"]),
            "path": str(AFF_BT),
        }
    )
else:
    print("MISS AFF backtest metrics", AFF_BT)

raw = pd.DataFrame(rows)
raw.to_csv(OUT / "csi500_metric_rows.csv", index=False)
summary = summarize(rows)

comparison_order = [
    "GVA",
    "AlphaGen",
    "PPO-filter",
    "GP",
    "XGBoost",
    "LightGBM",
    "MLP",
    "A2C",
    "REINFORCE",
    "QFR",
    "AFF",
]
ablation_order = ["GVA", "Critic-GVA", "Actor-GVA", "GVA-filter", "PPO-filter"]

comparison = summary[summary["method"].isin(comparison_order)].copy()
comparison["order"] = comparison["method"].map({m: i for i, m in enumerate(comparison_order)})
comparison = comparison.sort_values("order").drop(columns="order")

ablation = summary[summary["method"].isin(ablation_order)].copy()
ablation["order"] = ablation["method"].map({m: i for i, m in enumerate(ablation_order)})
ablation = ablation.sort_values("order").drop(columns="order")

comparison.to_csv(OUT / "table_csi500_comparison_metrics.csv", index=False)
ablation.to_csv(OUT / "table_csi500_ablation_metrics.csv", index=False)

print("COMPARISON")
print(comparison.to_string(index=False))
print("ABLATION")
print(ablation.to_string(index=False))
print("OUT", OUT)
