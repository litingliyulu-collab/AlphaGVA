#!/usr/bin/env python3
"""Run CSI500 chapter-4 backtests (filter suite + extra RL + AFF)."""
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

OUT_FILTER = Path("/root/autodl-tmp/gva_factor_experiments/backtests/csi500_chapter4_filter_20260701")
OUT_EXTRA = Path("/root/autodl-tmp/gva_factor_experiments/backtests/csi500_extra_rl_top50_20260701")

MAIN = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_main_compare_20260630_220231")
FILTER = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_filter_xgb_20260701_003217")
PPO_FILTER = Path(
    "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_ppo_history_filter_20260701_004456"
)
ML = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_ml_baselines_20260701_003217")
AFF_PRED = Path(
    "/root/autodl-tmp/aff_official_workspace_csi500_20260701_003217/out/affofficial_csi500_csi500_2023_0/pred_2023_10_inf_0.pt"
)


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
        specs.append(
            {
                "method": "alphagen_ppo",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(result_dir(MAIN, f"ppo_s{seed}")),
            }
        )
        specs.append(
            {
                "method": "critic_gva",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(result_dir(MAIN, f"critic_gva25_s{seed}")),
            }
        )
        specs.append(
            {
                "method": "full_gva",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(result_dir(MAIN, f"full_gva25_s{seed}")),
            }
        )
        specs.append(
            {
                "method": "ppo_filter_weak",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(result_dir(PPO_FILTER, f"ppo_filter_s{seed}")),
            }
        )
        specs.append(
            {
                "method": "gp_filter_strong",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(result_dir(FILTER, f"gp_filter_strong_s{seed}")),
            }
        )
        specs.append(
            {
                "method": "gva_filter",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(
                    result_dir(
                        Path(
                            "/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_gva_filter_rescue_20260701_003217"
                        ),
                        f"gva_filter_s{seed}",
                    )
                ),
            }
        )
        specs.append(
            {
                "method": "xgboost",
                "seed": str(seed),
                "kind": "npz",
                "path": latest_npz(result_dir(FILTER, f"xgboost_s{seed}")),
            }
        )
        specs.append(
            {
                "method": "lightgbm",
                "seed": str(seed),
                "kind": "npz",
                "path": latest_npz(result_dir(ML, f"lightgbm_s{seed}")),
            }
        )
        specs.append(
            {
                "method": "mlp",
                "seed": str(seed),
                "kind": "npz",
                "path": latest_npz(result_dir(ML, f"mlp_s{seed}")),
            }
        )
    if AFF_PRED.exists():
        specs.append(
            {
                "method": "aff_official_adapted",
                "seed": "0",
                "kind": "aff_pt",
                "path": str(AFF_PRED),
            }
        )
    return specs


def build_extra_specs() -> list[dict]:
    actor = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_actor_gva_20260630_220231")
    a2c = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_rl_advanced_20260630_220231")
    rq1 = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_reinforce_qfr_20260630_220932")
    rq2 = Path("/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_reinforce_qfr_resume_20260701_003217")
    specs: list[dict] = []
    for seed in range(3):
        specs.append(
            {
                "method": "actor_gva",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(result_dir(actor, f"actor_gva25_s{seed}")),
            }
        )
        specs.append(
            {
                "method": "a2c",
                "seed": str(seed),
                "kind": "pool",
                "path": latest_pool(result_dir(a2c, f"a2c_baseline_s{seed}")),
            }
        )
    for seed, root, sub in [
        (0, rq1, "reinforce_s0"),
        (1, rq1, "reinforce_s1"),
        (2, rq2, "reinforce_s2"),
        (0, rq1, "qfr_s0"),
        (1, rq2, "qfr_s1"),
        (2, rq2, "qfr_s2"),
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


def aff_scores(data, pred_path: str, device: torch.device):
    import torch as th

    obj = th.load(pred_path, map_location=device)
    if isinstance(obj, dict) and "pred" in obj:
        pred = obj["pred"]
    else:
        pred = obj
    tensor = th.tensor(pred, dtype=th.float32, device=device)
    if tensor.shape != (data.n_days, data.n_stocks):
        raise ValueError(f"{pred_path} pred shape {tuple(tensor.shape)} != {(data.n_days, data.n_stocks)}")
    df = data.make_dataframe(tensor, columns=["score"]).reset_index()
    df.columns = ["datetime", "instrument", "score"]
    return df.pivot(index="datetime", columns="instrument", values="score").sort_index()


def run_suite(out_dir: Path, specs: list[dict], args: SimpleNamespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input_specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    b.initialize_qlib(args.qlib_data_path, kernels=1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = b.StockData(args.instruments, args.test_start, args.test_end, device=device)
    returns = b.make_forward_returns(data)
    rows, failures, reports = [], [], {}
    for spec in specs:
        method, seed = spec["method"], int(spec["seed"])
        key = f"{method}_s{seed}"
        print("BACKTEST", key, spec["kind"], spec["path"], flush=True)
        try:
            if spec["kind"] == "pool":
                scores = b.pool_scores(data, spec["path"], device)
            elif spec["kind"] == "aff_pt":
                scores = aff_scores(data, spec["path"], device)
            else:
                scores = b.npz_scores(data, spec["path"], device)
            report, positions = b.calc_backtest(scores, returns, args.topk, args.cost_rate)
            reports[key] = report
            run_dir = out_dir / key
            run_dir.mkdir(exist_ok=True)
            report.to_csv(run_dir / "daily_report.csv")
            positions.to_csv(run_dir / "positions.csv", index=False)
            row = {
                "method": method,
                "seed": seed,
                **b.summarize(report),
                "path": spec["path"],
            }
            rows.append(row)
            if spec["kind"] == "aff_pt":
                pd.DataFrame([row]).to_csv(run_dir / "aff_official_metrics.csv", index=False)
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
        instruments="csi500",
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
