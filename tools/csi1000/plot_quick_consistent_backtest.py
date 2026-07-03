from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("paper/backtests/csi1000_chapter4_filter_20260701")
EXTRA = Path("paper/backtests/csi1000_extra_rl_top50_20260701")
OUT = Path("paper/figures/chapter4/fig_csi1000_backtest_quick_consistent_cum_return.png")


def read_curve(base: Path, prefix: str) -> tuple[pd.Series, pd.DataFrame]:
    curves = []
    first = None
    for seed in range(3):
        report = pd.read_csv(base / f"{prefix}_s{seed}" / "daily_report.csv", index_col=0, parse_dates=True)
        if first is None:
            first = report
        curves.append(((1 + report["return"].fillna(0)).cumprod() - 1).rename(seed))
    return pd.concat(curves, axis=1).mean(axis=1), first


def main() -> None:
    specs = [
        ("PPO-filter", ROOT, "ppo_filter_weak", "#1f77b4"),
        ("GVA-filter", ROOT, "gva_filter", "#ff7f0e"),
        ("REINFORCE", EXTRA, "reinforce", "#2ca02c"),
        ("AlphaGen", ROOT, "alphagen_ppo", "#d62728"),
        ("Critic-GVA", ROOT, "critic_gva", "#9467bd"),
        ("GVA", ROOT, "full_gva", "#8c564b"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    first_report = None
    for label, base, prefix, color in specs:
        curve, report = read_curve(base, prefix)
        if first_report is None:
            first_report = report
        ax.plot(curve.index, curve.values, label=label, linewidth=2, color=color)
    bench = (1 + first_report["bench"].fillna(0)).cumprod() - 1
    ax.plot(bench.index, bench.values, label="CSI1000 universe EW", color="black", linestyle=":", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("CSI1000 Quick-Consistent Backtest: Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
