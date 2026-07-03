#!/usr/bin/env python3
"""Run CSI1000 chapter-4 backtests for comparison and ablation figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

SCRIPT_DIR = Path("/root/alpha_1203/gva_factor_experiments/scripts")
sys.path.insert(0, str(SCRIPT_DIR))
import backtest_filter_suite as b  # noqa: E402

OUT_FILTER = Path("/root/autodl-tmp/gva_factor_experiments/backtests/csi1000_chapter4_filter_20260701")
OUT_EXTRA = Path("/root/autodl-tmp/gva_factor_experiments/backtests/csi1000_extra_rl_top50_20260701")

MAIN = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_main_compare_20260701_022717")
PPO_FILTER = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_ppo_history_filter_20260701_110038")
GVA_FILTER = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_gva_filter_20260701_110038")
GP_XGB = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_gp_xgb_20260701_154332")
ML = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_ml_baselines_20260701_154332")
ACTOR = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_actor_gva_20260701_154332")
A2C = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_rl_advanced_20260701_154332")
REINFORCE0 = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_reinforce_only_20260701_110038")
REINFORCE12 = Path(
    "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_reinforce_parallel_extra_20260701_111555"
)
QFR0 = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_qfr_only_20260701_154332")
QFR12 = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_qfr_parallel_extra_20260701_163125")


def latest_pool(result_dir: Path) -> str:
    xs = list(result_dir.glob("*_steps_pool.json"))
    if xs:
        return str(sorted(xs, key=lambda p: int(p.name.split("_")[0]))[-1])
    final = result_dir / "final_pool.json"
    if final.exists():
        return str(final)
    raise FileNotFoundError(result_dir)


def latest_npz(result_dir: Path) -> str:
    path = result_dir / "pred_test.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def result_dir(base: Path, sub: str) -> Path:
    return next((base / sub / "results").glob("*"))


def build_filter_specs() -> list[dict]:
    specs: list[dict] = []
    for seed in range(3):
        specs.extend(
            [
                {
                    "method": "alphagen_ppo",
                    "seed": str(seed),
                    "kind": "pool",
                    "path": latest_pool(result_dir(MAIN, f"ppo_s{seed}")),
                },
                {
                    "method": "critic_gva",
                    "seed": str(seed),
                    "kind": "pool",
                    "path": latest_pool(result_dir(MAIN, f"critic_gva25_s{seed}")),
                },
                {
                    "method": "full_gva",
                    "seed": str(seed),
                    "kind": "pool",
                    "path": latest_pool(result_dir(MAIN, f"full_gva25_s{seed}")),
                },
                {
                    "method": "ppo_filter_weak",
                    "seed": str(seed),
                    "kind": "pool",
                    "path": latest_pool(result_dir(PPO_FILTER, f"ppo_filter_s{seed}")),
                },
                {
                    "method": "gp_filter_strong",
                    "seed": str(seed),
                    "kind": "pool",
                    "path": latest_pool(result_dir(GP_XGB, f"gp_filter_strong_s{seed}")),
                },
                {
                    "method": "gva_filter",
                    "seed": str(seed),
                    "kind": "pool",
                    "path": latest_pool(result_dir(GVA_FILTER, f"gva_filter_s{seed}")),
                },
                {
                    "method": "xgboost",
                    "seed": str(seed),
                    "kind": "npz",
                    "path": latest_npz(result_dir(GP_XGB, f"xgboost_s{seed}")),
                },
                {
                    "method": "lightgbm",
                    "seed": str(seed),
                    "kind": "npz",
                    "path": latest_npz(result_dir(ML, f"lightgbm_s{seed}")),
                },
                {
                    "method": "mlp",
                    "seed": str(seed),
                    "kind": "npz",
                    "path": latest_npz(result_dir(ML, f"mlp_s{seed}")),
                },
            ]
        )
    return specs


def build_extra_specs() -> list[dict]:
    specs: list[dict] = []
    for seed in range(3):
        specs.extend(
            [
                {
                    "method": "actor_gva",
                    "seed": str(seed),
                    "kind": "pool",
                    "path": latest_pool(result_dir(ACTOR, f"actor_gva25_s{seed}")),
                },
                {
                    "method": "a2c",
                    "seed": str(seed),
                    "kind": "pool",
                    "path": latest_pool(result_dir(A2C, f"a2c_baseline_s{seed}")),
                },
            ]
        )
    for seed, root, sub in [
        (0, REINFORCE0, "reinforce_s0"),
        (1, REINFORCE12, "reinforce_s1"),
        (2, REINFORCE12, "reinforce_s2"),
        (0, QFR0, "qfr_s0"),
        (1, QFR12, "qfr_s1"),
        (2, QFR12, "qfr_s2"),
    ]:
        specs.append(
            {
                "method": "reinforce" if sub.startswith("reinforce") else "qfr",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(result_dir(root, sub)),
            }
        )
    return specs


def run_suite(out_dir: Path, specs: list[dict], args: SimpleNamespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input_specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    b.initialize_qlib(args.qlib_data_path, kernels=1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = b.StockData(args.instruments, args.test_start, args.test_end, device=device)
    returns = b.make_forward_returns(data)
    rows, failures = [], []
    for spec in specs:
        method, seed = spec["method"], int(spec["seed"])
        key = f"{method}_s{seed}"
        print("BACKTEST", key, spec["kind"], spec["path"], flush=True)
        try:
            if spec["kind"] == "pool":
                scores = b.pool_scores(data, spec["path"], device)
            else:
                scores = b.npz_scores(data, spec["path"], device)
            report, positions = b.calc_backtest(scores, returns, args.topk, args.cost_rate)
            run_dir = out_dir / key
            run_dir.mkdir(exist_ok=True)
            report.to_csv(run_dir / "daily_report.csv")
            positions.to_csv(run_dir / "positions.csv", index=False)
            rows.append({"method": method, "seed": seed, **b.summarize(report), "path": spec["path"]})
        except Exception as exc:
            print("FAILED", key, type(exc).__name__, exc, flush=True)
            failures.append({**spec, "key": key, "error_type": type(exc).__name__, "error": str(exc)})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_seed.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failed_specs.csv", index=False)
    print("OUT", out_dir, "rows", len(rows), "failures", len(failures))


def main() -> None:
    args = SimpleNamespace(
        qlib_data_path="/root/autodl-tmp/cn_data_akshare_2010_2026",
        instruments="csi1000",
        test_start="2024-01-02",
        test_end="2026-05-28",
        topk=50,
        cost_rate=0.0015,
        device="cuda:0",
    )
    run_suite(OUT_FILTER, build_filter_specs(), args)
    run_suite(OUT_EXTRA, build_extra_specs(), args)


if __name__ == "__main__":
    main()
