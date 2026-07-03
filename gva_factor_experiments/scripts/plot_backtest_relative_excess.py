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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    rows = []
    method_curves = {}
    bench_curve = None
    for method in ORDER:
        ret_curves = []
        rel_excess_curves = []
        diff_excess_curves = []
        for path in sorted(out_dir.glob(f"{method}_s*_daily_report.csv")):
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            nav = (1 + frame["return"].fillna(0)).cumprod() - 1
            bench_nav = (1 + frame["bench"].fillna(0)).cumprod() - 1
            rel_excess = (1 + nav) / (1 + bench_nav) - 1
            diff_excess = nav - bench_nav
            ret_curves.append(nav.rename(path.stem))
            rel_excess_curves.append(rel_excess.rename(path.stem))
            diff_excess_curves.append(diff_excess.rename(path.stem))
            if bench_curve is None:
                bench_curve = bench_nav

        if not ret_curves:
            continue

        avg_ret = pd.concat(ret_curves, axis=1).mean(axis=1)
        avg_rel = pd.concat(rel_excess_curves, axis=1).mean(axis=1)
        avg_diff = pd.concat(diff_excess_curves, axis=1).mean(axis=1)
        method_curves[method] = {"return": avg_ret, "relative_excess": avg_rel, "diff_excess": avg_diff}
        rows.append(
            {
                "method": method,
                "cum_return_curve": float(avg_ret.iloc[-1]),
                "relative_excess_curve": float(avg_rel.iloc[-1]),
                "diff_excess_curve": float(avg_diff.iloc[-1]),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "summary_curve_endpoints_corrected_excess.csv", index=False)

    for key, filename, title, ylabel in [
        ("relative_excess", "cumulative_relative_excess_return_by_method.png", "Cumulative Relative Excess Return on Test Set", "Relative Excess Return"),
        ("diff_excess", "cumulative_return_minus_benchmark_by_method.png", "Cumulative Return Minus Benchmark on Test Set", "Return - Benchmark"),
    ]:
        plt.figure(figsize=(12, 6))
        for method in ORDER:
            if method not in method_curves:
                continue
            curve = method_curves[method][key]
            plt.plot(curve.index, curve.values, label=LABELS[method], linewidth=2)
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
