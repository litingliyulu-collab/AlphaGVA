#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib
from alphagen_qlib.utils import load_alpha_pool_by_path


METHOD_LABELS = {
    "alphagen": "AlphaGen",
    "critic_gva": "Critic-GVA",
    "gva": "GVA",
    "ppo_filter": "PPO-filter",
    "gva_filter": "GVA-filter",
    "reinforce": "REINFORCE",
}
METHOD_ORDER = ["ppo_filter", "gva_filter", "reinforce", "alphagen", "critic_gva", "gva"]


def parse_step(path: Path) -> int:
    match = re.search(r"(\d+)_steps_pool\.json$", path.name)
    return int(match.group(1)) if match else -1


def latest(pattern: str) -> str:
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(pattern)
    return max(paths, key=os.path.getmtime)


def final_step_pool(pattern: str) -> str:
    paths = [Path(p) for p in glob.glob(pattern)]
    if not paths:
        raise FileNotFoundError(pattern)
    return str(sorted(paths, key=parse_step)[-1])


def make_returns(data: StockData) -> pd.DataFrame:
    close_idx = list(data._features).index(FeatureType.CLOSE)
    bt = data.max_backtrack_days
    close = data.data[:, close_idx, :]
    ret = close[bt + 1: bt + data.n_days + 1] / close[bt: bt + data.n_days] - 1.0
    frame = data.make_dataframe(ret, columns=["ret"]).reset_index()
    frame.columns = ["datetime", "instrument", "ret"]
    return frame.pivot(index="datetime", columns="instrument", values="ret").sort_index()


def pool_scores(data: StockData, pool_path: str, device: torch.device) -> pd.DataFrame:
    exprs, weights = load_alpha_pool_by_path(pool_path)
    with torch.no_grad():
        if not exprs:
            score = torch.zeros((data.n_days, data.n_stocks), dtype=torch.float32, device=device)
        else:
            factors = [expr.evaluate(data) * float(weights[i]) for i, expr in enumerate(exprs)]
            score = torch.sum(torch.stack(factors, dim=0), dim=0)
    frame = data.make_dataframe(score, columns=["score"]).reset_index()
    frame.columns = ["datetime", "instrument", "score"]
    return frame.pivot(index="datetime", columns="instrument", values="score").sort_index()


def calc_backtest(scores: pd.DataFrame, returns: pd.DataFrame, topk: int, cost_rate: float) -> pd.DataFrame:
    idx = scores.index.intersection(returns.index)
    cols = scores.columns.intersection(returns.columns)
    scores = scores.loc[idx, cols]
    returns = returns.loc[idx, cols]

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        valid = row.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        picks = valid.nlargest(min(topk, len(valid))).index
        weights.loc[dt, picks] = 1.0 / len(picks)

    gross = (weights * returns).sum(axis=1, min_count=1).fillna(0.0)
    bench = returns.mean(axis=1, skipna=True).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * cost_rate
    net = gross - cost
    return pd.DataFrame(
        {
            "return_gross": gross,
            "cost": cost,
            "return": net,
            "bench": bench,
            "excess_return": net - bench,
            "turnover": turnover,
        }
    )


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())


def summarize(report: pd.DataFrame) -> Dict[str, float]:
    ret = report["return"].fillna(0.0)
    excess = report["excess_return"].fillna(0.0)
    bench = report["bench"].fillna(0.0)
    days = max(len(ret), 1)
    ann = 252
    nav = (1 + ret).cumprod()
    ex_nav = (1 + excess).cumprod()
    bench_nav = (1 + bench).cumprod()
    return {
        "days": days,
        "cum_return": float(nav.iloc[-1] - 1),
        "bench_cum_return": float(bench_nav.iloc[-1] - 1),
        "excess_cum_return": float(ex_nav.iloc[-1] - 1),
        "annual_return": float(nav.iloc[-1] ** (ann / days) - 1),
        "excess_annual_return": float(ex_nav.iloc[-1] ** (ann / days) - 1),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * math.sqrt(ann)) if ret.std(ddof=1) > 0 else np.nan,
        "information_ratio": float(excess.mean() / excess.std(ddof=1) * math.sqrt(ann)) if excess.std(ddof=1) > 0 else np.nan,
        "max_drawdown": max_drawdown(nav),
        "excess_max_drawdown": max_drawdown(ex_nav),
        "avg_turnover": float(report["turnover"].mean()),
    }


def discover(args: argparse.Namespace) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    main = Path(args.main_root)
    for seed in range(3):
        specs.append({"method": "alphagen", "seed": str(seed), "path": final_step_pool(str(main / f"ppo_s{seed}" / "results" / "*" / "*_steps_pool.json"))})
        specs.append({"method": "critic_gva", "seed": str(seed), "path": final_step_pool(str(main / f"critic_gva25_s{seed}" / "results" / "*" / "*_steps_pool.json"))})
        specs.append({"method": "gva", "seed": str(seed), "path": final_step_pool(str(main / f"full_gva25_s{seed}" / "results" / "*" / "*_steps_pool.json"))})

    if args.ppo_filter_root:
        root = Path(args.ppo_filter_root)
        for seed in range(3):
            specs.append({"method": "ppo_filter", "seed": str(seed), "path": latest(str(root / f"ppo_filter_s{seed}" / "results" / "*" / "final_pool.json"))})

    if args.gva_filter_root:
        root = Path(args.gva_filter_root)
        for seed in range(3):
            specs.append({"method": "gva_filter", "seed": str(seed), "path": latest(str(root / f"gva_filter_s{seed}" / "results" / "*" / "final_pool.json"))})

    if args.reinforce_root:
        root = Path(args.reinforce_root)
        for seed in range(3):
            specs.append({"method": "reinforce", "seed": str(seed), "path": latest(str(root / f"reinforce_s{seed}" / "results" / "*" / "final_pool.json"))})
    return specs


def plot_curves(reports: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    for column, name, title, ylabel in [
        ("return", "cumulative_return_by_method.png", "Cumulative Return on Test Set", "Cumulative Return"),
        ("excess_return", "cumulative_excess_return_by_method.png", "Cumulative Excess Return on Test Set", "Cumulative Excess Return"),
    ]:
        plt.figure(figsize=(12, 6))
        for method in METHOD_ORDER:
            reps = [report for key, report in reports.items() if key.startswith(method + "_s")]
            if not reps:
                continue
            avg = pd.concat([rep[column].rename(str(i)) for i, rep in enumerate(reps)], axis=1).fillna(0.0).mean(axis=1)
            curve = (1 + avg).cumprod() - 1
            plt.plot(curve.index, curve.values, label=METHOD_LABELS[method], linewidth=2)
        if column == "return" and reports:
            first = next(iter(reports.values()))
            bench = (1 + first["bench"].fillna(0.0)).cumprod() - 1
            plt.plot(bench.index, bench.values, label="CSI1000 universe EW", color="black", linestyle=":", linewidth=2)
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--main_root", required=True)
    parser.add_argument("--ppo_filter_root", default="")
    parser.add_argument("--gva_filter_root", default="")
    parser.add_argument("--reinforce_root", default="")
    parser.add_argument("--qlib_data_path", default="/root/autodl-tmp/cn_data_akshare_2010_2026")
    parser.add_argument("--instruments", default="csi1000")
    parser.add_argument("--test_start", default="2024-01-02")
    parser.add_argument("--test_end", default="2026-05-28")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--cost_rate", type=float, default=0.0015)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    initialize_qlib(args.qlib_data_path, kernels=1)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data = StockData(args.instruments, args.test_start, args.test_end, device=device)
    returns = make_returns(data)
    specs = discover(args)
    (out_dir / "input_specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    reports: Dict[str, pd.DataFrame] = {}
    summary_rows = []
    for spec in specs:
        key = f"{spec['method']}_s{spec['seed']}"
        print(f"[BACKTEST] {key} {spec['path']}", flush=True)
        report = calc_backtest(pool_scores(data, spec["path"], device), returns, args.topk, args.cost_rate)
        reports[key] = report
        report.to_csv(out_dir / f"{key}_daily_report.csv")
        summary = summarize(report)
        summary.update({"method": spec["method"], "seed": int(spec["seed"]), "path": spec["path"]})
        summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_by_seed.csv", index=False)
    grouped = summary.groupby("method").agg(["mean", "std"])
    grouped.to_csv(out_dir / "summary_by_method.csv")
    plot_curves(reports, out_dir)
    print(grouped)
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
