#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "alphagen": "AlphaGen",
    "critic_gva": "Critic-GVA",
    "gva": "GVA",
    "ppo_filter": "PPO-filter",
    "gva_filter": "GVA-filter",
    "reinforce": "REINFORCE",
}
ORDER = ["ppo_filter", "gva_filter", "reinforce", "alphagen", "critic_gva", "gva"]
NUMERIC_COLS = [
    "days",
    "cum_return",
    "bench_cum_return",
    "excess_cum_return",
    "annual_return",
    "excess_annual_return",
    "sharpe",
    "information_ratio",
    "max_drawdown",
    "excess_max_drawdown",
    "avg_turnover",
]


def load_reports(out_dir: Path):
    reports = {}
    for path in sorted(out_dir.glob("*_daily_report.csv")):
        key = path.name.removesuffix("_daily_report.csv")
        reports[key] = pd.read_csv(path, index_col=0, parse_dates=True)
    return reports


def plot_curves(reports, out_dir: Path) -> None:
    for column, filename, title, ylabel in [
        ("return", "cumulative_return_by_method.png", "Cumulative Return on Test Set", "Cumulative Return"),
        ("excess_return", "cumulative_excess_return_by_method.png", "Cumulative Excess Return on Test Set", "Cumulative Excess Return"),
    ]:
        plt.figure(figsize=(12, 6))
        for method in ORDER:
            reps = [report for key, report in reports.items() if key.startswith(method + "_s")]
            if not reps:
                continue
            avg = pd.concat([rep[column].rename(str(i)) for i, rep in enumerate(reps)], axis=1).fillna(0.0).mean(axis=1)
            curve = (1 + avg).cumprod() - 1
            plt.plot(curve.index, curve.values, label=LABELS.get(method, method), linewidth=2)
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
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    summary = pd.read_csv(out_dir / "summary_by_seed.csv")
    grouped = summary.groupby("method")[NUMERIC_COLS].agg(["mean", "std"])
    grouped.to_csv(out_dir / "summary_by_method.csv")
    reports = load_reports(out_dir)
    plot_curves(reports, out_dir)
    print(grouped[["cum_return", "excess_cum_return", "sharpe", "information_ratio"]])
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
