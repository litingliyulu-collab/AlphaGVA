#!/usr/bin/env python3
"""Copy AFF main-window metrics into CSI500 chapter-4 backtest bundle."""
from pathlib import Path
import shutil

import pandas as pd

AFF_DIR = Path(
    "/root/autodl-tmp/aff_official_workspace_csi500_20260701_003217/out/"
    "affofficial_csi500_csi500_2023_0_mainwindow"
)
BT_DIR = Path(
    "/root/autodl-tmp/gva_factor_experiments/backtests/"
    "csi500_chapter4_filter_20260701/aff_official_adapted_s0"
)


def main() -> None:
    BT_DIR.mkdir(parents=True, exist_ok=True)
    src_report = AFF_DIR / "aff_official_daily_report.csv"
    dst_report = BT_DIR / "daily_report.csv"
    df = pd.read_csv(src_report)
    df = df.rename(columns={"date": "datetime", "benchmark_return": "bench"})
    cols = ["datetime", "return", "bench", "excess_return"]
    if "turnover" in df.columns:
        cols.append("turnover")
    df[cols].to_csv(dst_report, index=False)
    shutil.copy(AFF_DIR / "aff_official_metrics.csv", BT_DIR / "aff_official_metrics.csv")
    print("wrote", dst_report)
    print("wrote", BT_DIR / "aff_official_metrics.csv")


if __name__ == "__main__":
    main()
