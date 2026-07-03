import argparse
import csv
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

from alphagen.data.expression import (
    Abs,
    Add,
    CSRank,
    Constant,
    Corr,
    Cov,
    Delta,
    Div,
    EMA,
    Expression,
    Feature,
    Greater,
    Less,
    Mad,
    Max,
    Mean,
    Med,
    Min,
    Mul,
    OutOfDataRangeError,
    Rank,
    Ref,
    Sign,
    Std,
    Sub,
    Sum,
    Var,
    WMA,
)
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib


UNARY_OPS = [Abs, Sign, CSRank]
BINARY_OPS = [Add, Sub, Mul, Div, Greater, Less]
ROLLING_OPS = [Ref, Mean, Sum, Std, Var, Max, Min, Med, Mad, Rank, Delta, WMA, EMA]
PAIR_ROLLING_OPS = [Cov, Corr]
WINDOWS = [5, 10, 20, 30, 40, 50]
CONSTANTS = [-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
FEATURES = [
    Feature(FeatureType.OPEN),
    Feature(FeatureType.CLOSE),
    Feature(FeatureType.HIGH),
    Feature(FeatureType.LOW),
    Feature(FeatureType.VOLUME),
    Feature(FeatureType.VWAP),
]


def build_target() -> Expression:
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


def make_run_dir(output_root: str, method_name: str, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(output_root, f"{method_name}_s{seed}", "results", f"{timestamp}_random")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def random_terminal(rng: random.Random) -> Expression:
    if rng.random() < 0.78:
        return rng.choice(FEATURES)
    return Constant(rng.choice(CONSTANTS))


def random_expr(rng: random.Random, max_depth: int, p_terminal: float = 0.28) -> Expression:
    if max_depth <= 0 or rng.random() < p_terminal:
        return random_terminal(rng)
    kind = rng.choices(
        ["unary", "binary", "rolling", "pair_rolling"],
        weights=[0.18, 0.38, 0.34, 0.10],
        k=1,
    )[0]
    if kind == "unary":
        return rng.choice(UNARY_OPS)(random_expr(rng, max_depth - 1, p_terminal))
    if kind == "binary":
        return rng.choice(BINARY_OPS)(
            random_expr(rng, max_depth - 1, p_terminal),
            random_expr(rng, max_depth - 1, p_terminal),
        )
    if kind == "rolling":
        return rng.choice(ROLLING_OPS)(random_expr(rng, max_depth - 1, p_terminal), rng.choice(WINDOWS))
    return rng.choice(PAIR_ROLLING_OPS)(
        random_expr(rng, max_depth - 1, p_terminal),
        random_expr(rng, max_depth - 1, p_terminal),
        rng.choice(WINDOWS),
    )


def evaluate_pool(
    pool: MseAlphaPool,
    valid_calculator: QLibStockDataCalculator,
    test_calculator: QLibStockDataCalculator,
) -> Dict[str, float]:
    if pool.size == 0:
        return {
            "valid/ic_mean": 0.0,
            "valid/rank_ic_mean": 0.0,
            "test/ic_mean": 0.0,
            "test/rank_ic_mean": 0.0,
        }
    valid_ic, valid_ric = pool.test_ensemble(valid_calculator)
    test_ic, test_ric = pool.test_ensemble(test_calculator)
    return {
        "valid/ic_mean": valid_ic,
        "valid/rank_ic_mean": valid_ric,
        "test/ic_mean": test_ic,
        "test/rank_ic_mean": test_ric,
    }


def write_metrics_header(path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attempt",
                "valid_exprs",
                "unique",
                "pool/size",
                "pool/eval_cnt",
                "pool/best_ic_ret",
                "valid/ic_mean",
                "valid/rank_ic_mean",
                "test/ic_mean",
                "test/rank_ic_mean",
            ],
        )
        writer.writeheader()


def append_metrics(path: str, row: Dict[str, float]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def build_data(args: argparse.Namespace, device: torch.device) -> Tuple[QLibStockDataCalculator, QLibStockDataCalculator, QLibStockDataCalculator]:
    initialize_qlib(args.qlib_data_path, kernels=args.qlib_kernels)
    target = build_target()
    train = StockData(args.instruments, args.train_start, args.train_end, device=device)
    valid = StockData(args.instruments, args.valid_start, args.valid_end, device=device)
    test = StockData(args.instruments, args.test_start, args.test_end, device=device)
    return (
        QLibStockDataCalculator(train, target),
        QLibStockDataCalculator(valid, target),
        QLibStockDataCalculator(test, target),
    )


def build_pool_from_scores(
    scores: Dict[str, Tuple[float, Expression]],
    train_calc: QLibStockDataCalculator,
    capacity: int,
    candidate_multiplier: int,
    l1_alpha: float,
    device: torch.device,
) -> MseAlphaPool:
    pool = MseAlphaPool(
        capacity=capacity,
        calculator=train_calc,
        ic_lower_bound=None,
        l1_alpha=l1_alpha,
        device=device,
    )
    candidates = sorted(scores.values(), key=lambda item: abs(item[0]), reverse=True)
    exprs = [expr for score, expr in candidates[: capacity * candidate_multiplier] if np.isfinite(score)]
    pool.force_load_exprs(exprs)
    pool.eval_cnt = len(scores)
    return pool


def emit_metrics(
    metrics_path: str,
    attempt: int,
    scores: Dict[str, Tuple[float, Expression]],
    pool: MseAlphaPool,
    valid_calc: QLibStockDataCalculator,
    test_calc: QLibStockDataCalculator,
) -> None:
    eval_scores = evaluate_pool(pool, valid_calc, test_calc)
    row = {
        "attempt": attempt,
        "valid_exprs": len(scores),
        "unique": len(scores),
        "pool/size": pool.size,
        "pool/eval_cnt": pool.eval_cnt,
        "pool/best_ic_ret": pool.best_ic_ret,
        **eval_scores,
    }
    append_metrics(metrics_path, row)
    print(row, flush=True)


def run_one(args: argparse.Namespace, seed: int) -> None:
    reseed_everything(seed)
    rng = random.Random(seed)
    if args.device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device_str)
    train_calc, valid_calc, test_calc = build_data(args, device)
    run_dir = make_run_dir(args.output_root, args.method_name, seed)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    write_metrics_header(metrics_path)

    scores: Dict[str, Tuple[float, Expression]] = {}
    pool = build_pool_from_scores(scores, train_calc, args.pool_capacity, args.pool_candidate_multiplier, args.l1_alpha, device)
    for attempt in range(1, args.attempts + 1):
        expr = random_expr(rng, args.max_depth)
        key = str(expr)
        if key in scores:
            continue
        try:
            score = train_calc.calc_single_IC_ret(expr)
        except (OutOfDataRangeError, FloatingPointError, RuntimeError, ValueError, ZeroDivisionError):
            continue
        if not np.isfinite(score):
            continue
        scores[key] = (float(score), expr)

        if attempt % args.eval_every == 0:
            pool = build_pool_from_scores(scores, train_calc, args.pool_capacity, args.pool_candidate_multiplier, args.l1_alpha, device)
            emit_metrics(metrics_path, attempt, scores, pool, valid_calc, test_calc)

    pool = build_pool_from_scores(scores, train_calc, args.pool_capacity, args.pool_candidate_multiplier, args.l1_alpha, device)
    emit_metrics(metrics_path, args.attempts, scores, pool, valid_calc, test_calc)
    with open(os.path.join(run_dir, "final_pool.json"), "w") as f:
        json.dump(pool.to_json_dict(), f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seeds", default="0")
    parser.add_argument("--method_name", default="random_search")
    parser.add_argument("--output_root", default="/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_manual")
    parser.add_argument("--qlib_data_path", default="/root/autodl-tmp/cn_data_akshare_2010_2026")
    parser.add_argument("--qlib_kernels", type=int, default=1)
    parser.add_argument("--device_str", default="auto")
    parser.add_argument("--instruments", default="csi300")
    parser.add_argument("--pool_capacity", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=30000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--pool_candidate_multiplier", type=int, default=5)
    parser.add_argument("--l1_alpha", type=float, default=5e-3)
    parser.add_argument("--train_start", default="2010-01-04")
    parser.add_argument("--train_end", default="2021-12-31")
    parser.add_argument("--valid_start", default="2022-01-04")
    parser.add_argument("--valid_end", default="2023-12-29")
    parser.add_argument("--test_start", default="2024-01-02")
    parser.add_argument("--test_end", default="2026-05-28")
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]


if __name__ == "__main__":
    args = parse_args()
    for seed in parse_seeds(args.random_seeds):
        run_one(args, seed)
