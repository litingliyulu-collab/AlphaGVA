#!/usr/bin/env python3
"""Top-K backtest for Full-GVA-original vs Full-GVA-multi, each with IC and multi weights."""
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib
from alphagen_qlib.utils import load_alpha_pool_by_path
from tools.reward_pool_weights import optimize_multi_weights


WEIGHT_LABELS = {
    "ic": "IC weights (pool.json)",
    "multi": "Multi/R4 weights (train-fit)",
}


def parse_step(path: Path) -> int:
    match = re.search(r"(\d+)_steps_pool\.json$", path.name)
    return int(match.group(1)) if match else -1


def find_final_pool(run_root: Path, method: str, seed: int) -> Path:
    method_dir = run_root / f"{method}_s{seed}"
    paths = list(method_dir.glob("results/*/*_steps_pool.json"))
    if not paths:
        raise FileNotFoundError(f"No pool json under {method_dir}")
    return sorted(paths, key=parse_step)[-1]


def make_scores_from_exprs(
    exprs,
    weights: List[float],
    data: StockData,
    device: torch.device,
) -> pd.DataFrame:
    with torch.no_grad():
        if not exprs:
            score = torch.zeros((data.n_days, data.n_stocks), dtype=torch.float32, device=device)
        else:
            factors = [
                normalize_by_day(expr.evaluate(data)) * float(weights[i])
                for i, expr in enumerate(exprs)
            ]
            score = torch.sum(torch.stack(factors, dim=0), dim=0)
    df = data.make_dataframe(score, columns=["score"]).reset_index()
    df.columns = ["datetime", "instrument", "score"]
    return df.pivot(index="datetime", columns="instrument", values="score").sort_index()


def make_forward_returns(data: StockData) -> pd.DataFrame:
    close_idx = list(data._features).index(FeatureType.CLOSE)
    bt = data.max_backtrack_days
    n = data.n_days
    close = data.data[:, close_idx, :]
    ret = close[bt + 1: bt + n + 1] / close[bt: bt + n] - 1.0
    df = data.make_dataframe(ret, columns=["ret"]).reset_index()
    df.columns = ["datetime", "instrument", "ret"]
    return df.pivot(index="datetime", columns="instrument", values="ret").sort_index()


def calc_backtest(
    scores: pd.DataFrame, returns: pd.DataFrame, topk: int, cost_rate: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_idx = scores.index.intersection(returns.index)
    common_cols = scores.columns.intersection(returns.columns)
    scores = scores.loc[common_idx, common_cols]
    returns = returns.loc[common_idx, common_cols]

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        valid = row.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) == 0:
            continue
        picks = valid.nlargest(min(topk, len(valid))).index
        weights.loc[dt, picks] = 1.0 / len(picks)

    gross = (weights * returns).sum(axis=1, min_count=1).fillna(0.0)
    bench = returns.mean(axis=1, skipna=True).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * cost_rate
    net = gross - cost
    excess = net - bench

    report = pd.DataFrame(
        {
            "return_gross": gross,
            "cost": cost,
            "return": net,
            "bench": bench,
            "excess_return": excess,
            "turnover": turnover,
        }
    )
    positions = weights.stack().rename("weight").reset_index()
    positions.columns = ["datetime", "instrument", "weight"]
    positions = positions[positions["weight"] != 0]
    return report, positions


def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())


def summarize(report: pd.DataFrame) -> Dict[str, float]:
    r = report["return"].fillna(0.0)
    b = report["bench"].fillna(0.0)
    e = report["excess_return"].fillna(0.0)
    n = max(len(r), 1)
    ann = 252
    cum = (1 + r).cumprod()
    bcum = (1 + b).cumprod()
    ecum = (1 + e).cumprod()
    vol = float(r.std(ddof=1) * math.sqrt(ann)) if n > 1 else 0.0
    return {
        "days": n,
        "cum_return": float(cum.iloc[-1] - 1),
        "bench_cum_return": float(bcum.iloc[-1] - 1),
        "excess_cum_return": float(ecum.iloc[-1] - 1),
        "annual_return": float(cum.iloc[-1] ** (ann / n) - 1),
        "bench_annual_return": float(bcum.iloc[-1] ** (ann / n) - 1),
        "excess_annual_return": float(ecum.iloc[-1] ** (ann / n) - 1),
        "annual_vol": vol,
        "sharpe": float((r.mean() / r.std(ddof=1)) * math.sqrt(ann)) if r.std(ddof=1) > 0 else np.nan,
        "information_ratio": float((e.mean() / e.std(ddof=1)) * math.sqrt(ann)) if e.std(ddof=1) > 0 else np.nan,
        "max_drawdown": max_drawdown(cum),
        "excess_max_drawdown": max_drawdown(ecum),
        "avg_turnover": float(report["turnover"].mean()),
        "avg_cost": float(report["cost"].mean()),
        "win_rate": float((r > 0).mean()),
    }


def plot_curves(rows: pd.DataFrame, method_reports: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    for _, row in rows.drop_duplicates(["method", "weight_scheme"]).iterrows():
        method = row["method"]
        scheme = row["weight_scheme"]
        label = f"{row['method_label']} | {row['weight_label']}"
        reps = [r for k, r in method_reports.items() if k.startswith(f"{method}_") and k.endswith(f"_{scheme}")]
        if not reps:
            continue
        returns = pd.concat([r["return"].rename(i) for i, r in enumerate(reps)], axis=1).fillna(0.0).mean(axis=1)
        cum = (1 + returns).cumprod() - 1
        plt.plot(cum.index, cum.values, label=label, linewidth=2)
    first = next(iter(method_reports.values()))
    bench = (1 + first["bench"].fillna(0.0)).cumprod() - 1
    plt.plot(bench.index, bench.values, label="CSI300 universe EW", color="black", linestyle=":", linewidth=2)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("Cumulative Return on Test Set (IC vs Multi weights)")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_return_by_method.png", dpi=180)
    plt.close()


def resolve_weights(
    exprs,
    ic_weights: List[float],
    train_data: StockData,
    device: torch.device,
    scheme: str,
    multi_opt_max_steps: int,
    multi_opt_tolerance: int,
) -> Tuple[List[float], str]:
    if scheme == "ic":
        return [float(w) for w in ic_weights], "loaded from pool.json"
    multi_weights = optimize_multi_weights(
        exprs,
        train_data,
        ic_weights,
        max_steps=multi_opt_max_steps,
        tolerance=multi_opt_tolerance,
        device=device,
    )
    return [float(w) for w in multi_weights], "fitted on train with R4 quality objective"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--orig_run_root",
        default="/root/alpha_1203/gva_factor_experiments/runs_newdata/main_compare_20260628_211640",
    )
    ap.add_argument(
        "--multi_run_root",
        default="/root/alpha_1203/gva_factor_experiments/runs_newdata/reward_multi_formal_20260629_235442",
    )
    ap.add_argument(
        "--output_dir",
        default="/root/alpha_1203/gva_factor_experiments/backtests/reward_compare_dual_weights_top50",
    )
    ap.add_argument("--qlib_data_path", default="/root/autodl-tmp/cn_data_akshare_2010_2026")
    ap.add_argument("--instruments", default="csi300")
    ap.add_argument("--train_start", default="2010-01-04")
    ap.add_argument("--train_end", default="2021-12-31")
    ap.add_argument("--test_start", default="2024-01-02")
    ap.add_argument("--test_end", default="2026-05-28")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--cost_rate", type=float, default=0.0015)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--multi_opt_max_steps", type=int, default=2000)
    ap.add_argument("--multi_opt_tolerance", type=int, default=200)
    ap.add_argument(
        "--weight_schemes",
        default="ic,multi",
        help="Comma-separated: ic (pool.json), multi (train-fit R4 weights)",
    )
    args = ap.parse_args()

    weight_schemes = [s.strip() for s in args.weight_schemes.split(",") if s.strip()]
    specs = [
        ("full_gva25", Path(args.orig_run_root), "Full-GVA-original"),
        ("full_gva_multi", Path(args.multi_run_root), "Full-GVA-multi"),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    initialize_qlib(args.qlib_data_path, kernels=1)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"DEVICE={device}", flush=True)

    train_data = StockData(args.instruments, args.train_start, args.train_end, device=device)
    test_data = StockData(args.instruments, args.test_start, args.test_end, device=device)
    returns = make_forward_returns(test_data)

    rows: List[dict] = []
    method_reports: Dict[str, pd.DataFrame] = {}

    for method, run_root, label in specs:
        for seed in [0, 1, 2]:
            pool_path = find_final_pool(run_root, method, seed)
            exprs, ic_weights = load_alpha_pool_by_path(str(pool_path))
            print(
                f"POOL {method}_s{seed} n={len(exprs)} pool={pool_path}",
                flush=True,
            )

            for scheme in weight_schemes:
                weights, weight_note = resolve_weights(
                    exprs,
                    ic_weights,
                    train_data,
                    device,
                    scheme,
                    args.multi_opt_max_steps,
                    args.multi_opt_tolerance,
                )
                key = f"{method}_s{seed}_{scheme}"
                print(f"BACKTEST {key} weights={weight_note}", flush=True)

                scores = make_scores_from_exprs(exprs, weights, test_data, device)
                report, positions = calc_backtest(scores, returns, args.topk, args.cost_rate)
                method_reports[key] = report

                run_dir = out_dir / key
                run_dir.mkdir(exist_ok=True)
                report.to_csv(run_dir / "daily_report.csv")
                positions.to_csv(run_dir / "positions.csv", index=False)
                scores.stack().rename("score").reset_index().to_csv(run_dir / "scores.csv", index=False)
                with open(run_dir / "factor_weights.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "weight_scheme": scheme,
                            "weight_note": weight_note,
                            "exprs": [str(e) for e in exprs],
                            "ic_weights": [float(w) for w in ic_weights],
                            "used_weights": weights,
                            "pool_path": str(pool_path),
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                summary = summarize(report)
                summary.update(
                    {
                        "method": method,
                        "method_label": label,
                        "seed": seed,
                        "weight_scheme": scheme,
                        "weight_label": WEIGHT_LABELS.get(scheme, scheme),
                        "n_factors": len(exprs),
                        "pool_path": str(pool_path),
                    }
                )
                rows.append(summary)

    metrics = pd.DataFrame(rows)
    cols = [
        "method",
        "method_label",
        "seed",
        "weight_scheme",
        "weight_label",
        "n_factors",
        "days",
        "cum_return",
        "bench_cum_return",
        "excess_cum_return",
        "annual_return",
        "bench_annual_return",
        "excess_annual_return",
        "annual_vol",
        "sharpe",
        "information_ratio",
        "max_drawdown",
        "excess_max_drawdown",
        "avg_turnover",
        "avg_cost",
        "win_rate",
        "pool_path",
    ]
    metrics = metrics[cols]
    metrics.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    numeric = [
        c
        for c in metrics.columns
        if c not in ["method", "method_label", "seed", "weight_scheme", "weight_label", "pool_path"]
    ]
    summary = metrics.groupby(["method", "method_label", "weight_scheme", "weight_label"])[numeric].agg(["mean", "std"])
    summary.to_csv(out_dir / "metrics_summary.csv")
    plot_curves(metrics, method_reports, out_dir)

    print("OUTPUT_DIR", out_dir)
    print(
        metrics[
            [
                "method_label",
                "weight_scheme",
                "seed",
                "n_factors",
                "cum_return",
                "annual_return",
                "excess_cum_return",
                "information_ratio",
                "max_drawdown",
                "avg_turnover",
            ]
        ].to_string(index=False)
    )
    print("SUMMARY")
    print(
        summary[
            [
                ("cum_return", "mean"),
                ("annual_return", "mean"),
                ("excess_cum_return", "mean"),
                ("information_ratio", "mean"),
                ("max_drawdown", "mean"),
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()
