#!/usr/bin/env python3
"""Run CSI1000 chapter-4 top100 backtests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path("/root/alpha_1203/gva_factor_experiments/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

import backtest_csi1000_chapter4 as base  # noqa: E402


def main() -> None:
    args = SimpleNamespace(
        qlib_data_path="/root/autodl-tmp/cn_data_akshare_2010_2026",
        instruments="csi1000",
        test_start="2024-01-02",
        test_end="2026-05-28",
        topk=100,
        cost_rate=0.0015,
        device="cuda:0",
    )
    base.run_suite(
        Path("/root/autodl-tmp/gva_factor_experiments/backtests/csi1000_chapter4_filter_top100_20260701"),
        base.build_filter_specs(),
        args,
    )
    base.run_suite(
        Path("/root/autodl-tmp/gva_factor_experiments/backtests/csi1000_extra_rl_top100_20260701"),
        base.build_extra_specs(),
        args,
    )


if __name__ == "__main__":
    main()
