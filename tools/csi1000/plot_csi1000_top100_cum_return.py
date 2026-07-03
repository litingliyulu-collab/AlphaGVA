from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = Path("paper/figures/chapter4")
BT_FILTER = Path("paper/backtests/csi1000_chapter4_filter_top100_20260701")
BT_EXTRA = Path("paper/backtests/csi1000_extra_rl_top100_20260701")


def read_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        for candidate in ["date", "Date", "Unnamed: 0", "index"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "datetime"})
                break
    if "datetime" not in df.columns:
        df = pd.read_csv(path, index_col=0).reset_index().rename(columns={"index": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()


def reports_for(base: Path, prefix: str) -> list[pd.DataFrame]:
    reports = []
    for seed in range(3):
        p = base / f"{prefix}_s{seed}" / "daily_report.csv"
        if p.exists():
            reports.append(read_daily(p))
    return reports


def mean_cum_return(reports: list[pd.DataFrame]) -> pd.Series:
    curves = [((1 + r["return"].fillna(0.0)).cumprod() - 1).rename(i) for i, r in enumerate(reports)]
    return pd.concat(curves, axis=1).mean(axis=1)


def bench_curve(report: pd.DataFrame) -> pd.Series:
    return (1 + report["bench"].fillna(0.0)).cumprod() - 1


def plot(specs, out_name: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    first_report = None
    for label, base, prefix, color, lw in specs:
        reports = reports_for(base, prefix)
        if not reports:
            print("missing reports", label, base, prefix)
            continue
        if first_report is None:
            first_report = reports[0]
        curve = mean_cum_return(reports)
        ax.plot(curve.index, curve.values, label=label, color=color, linewidth=lw)
    if first_report is not None:
        bench = bench_curve(first_report)
        ax.plot(bench.index, bench.values, label="CSI1000 universe EW", color="black", linestyle=":", linewidth=2.0)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    ax.legend(fontsize=9, ncol=2, frameon=True)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    comparison_specs = [
        ("GVA", BT_FILTER, "full_gva", "#d62728", 2.8),
        ("AlphaGen", BT_FILTER, "alphagen_ppo", "#1f77b4", 2.1),
        ("PPO-filter", BT_FILTER, "ppo_filter_weak", "#ff7f0e", 1.9),
        ("GP", BT_FILTER, "gp_filter_strong", "#2ca02c", 1.9),
        ("XGBoost", BT_FILTER, "xgboost", "#9467bd", 1.7),
        ("LightGBM", BT_FILTER, "lightgbm", "#8c564b", 1.7),
        ("MLP", BT_FILTER, "mlp", "#e377c2", 1.7),
        ("A2C", BT_EXTRA, "a2c", "#17becf", 1.8),
        ("REINFORCE", BT_EXTRA, "reinforce", "#7f7f7f", 1.8),
        ("QFR", BT_EXTRA, "qfr", "#bcbd22", 1.9),
    ]
    ablation_specs = [
        ("Full-GVA", BT_FILTER, "full_gva", "#d62728", 2.7),
        ("Critic-GVA", BT_FILTER, "critic_gva", "#ff7f0e", 2.2),
        ("Actor-GVA", BT_EXTRA, "actor_gva", "#2ca02c", 2.2),
        ("GVA-filter", BT_FILTER, "gva_filter", "#9467bd", 1.9),
        ("PPO-filter", BT_FILTER, "ppo_filter_weak", "#1f77b4", 1.8),
    ]
    print(
        plot(
            comparison_specs,
            "fig_csi1000_top100_backtest_comparison_cum_return.png",
            "CSI1000 Comparison Backtest: Cumulative Return (Top100)",
        )
    )
    print(
        plot(
            ablation_specs,
            "fig_csi1000_top100_backtest_ablation_cum_return.png",
            "CSI1000 Ablation Backtest: Cumulative Return (Top100)",
        )
    )


if __name__ == "__main__":
    main()
