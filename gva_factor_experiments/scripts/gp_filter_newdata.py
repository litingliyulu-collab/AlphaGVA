import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

from alphagen.data.parser import ExpressionParser, ExpressionParsingError
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
    Log,
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
from alphagen_generic.features import close, high, low, open_, volume, vwap
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


EVAL_ENV = {
    "open_": open_,
    "close": close,
    "high": high,
    "low": low,
    "volume": volume,
    "vwap": vwap,
    "Constant": Constant,
    "Abs": Abs,
    "Sign": Sign,
    "Log": Log,
    "CSRank": CSRank,
    "Add": Add,
    "Sub": Sub,
    "Mul": Mul,
    "Div": Div,
    "Greater": Greater,
    "Less": Less,
    "Ref": Ref,
    "Mean": Mean,
    "Sum": Sum,
    "Std": Std,
    "Var": Var,
    "Max": Max,
    "Min": Min,
    "Med": Med,
    "Mad": Mad,
    "Rank": Rank,
    "Delta": Delta,
    "WMA": WMA,
    "EMA": EMA,
    "Cov": Cov,
    "Corr": Corr,
}
def build_target() -> Expression:
    close_expr = Feature(FeatureType.CLOSE)
    return Ref(close_expr, -20) / close_expr - 1


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]


def make_run_dir(output_root: str, method_name: str, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(output_root, f"{method_name}_s{seed}", "results", f"{timestamp}_gp")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


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


PARSER = ExpressionParser([Abs, Sign, Log, CSRank, Add, Sub, Mul, Div, Greater, Less, Ref, Mean, Sum, Std, Var, Max, Min, Med, Mad, Rank, Delta, WMA, EMA, Cov, Corr])

def safe_eval_expr(key: str) -> Expression:
    expr = eval(key, {"__builtins__": {}}, EVAL_ENV)
    # Round-trip through AlphaGen parser to enforce the same type rules used by saved pools/backtests.
    return PARSER.parse(str(expr))

def rank_score(score: float, mode: str) -> float:
    return abs(score) if mode == "strong" else max(score, 0.0)


def write_metrics_header(path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generation",
                "cache_size",
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


def run_one(args: argparse.Namespace, seed: int) -> None:
    reseed_everything(seed)
    if args.device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device_str)
    train_calc, valid_calc, test_calc = build_data(args, device)
    pool = MseAlphaPool(
        capacity=args.pool_capacity,
        calculator=train_calc,
        ic_lower_bound=None,
        l1_alpha=args.l1_alpha,
        device=device,
    )
    run_dir = make_run_dir(args.output_root, args.method_name, seed)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    write_metrics_header(metrics_path)
    cache: Dict[str, float] = {}
    generation = 0

    def metric(_x, y, _w):
        key = y[0]
        if key in cache:
            return cache[key]
        token_len = key.count("(") + key.count(")")
        if token_len > args.max_tokens:
            cache[key] = -1.0
            return cache[key]
        try:
            expr = safe_eval_expr(key)
            raw_score = float(train_calc.calc_single_IC_ret(expr))
            score = rank_score(raw_score, args.filter_mode)
        except (OutOfDataRangeError, FloatingPointError, RuntimeError, ValueError, ZeroDivisionError, SyntaxError, NameError, TypeError, ExpressionParsingError):
            score = -1.0
        if not np.isfinite(score):
            score = -1.0
        cache[key] = float(score)
        return cache[key]

    def refresh_pool() -> None:
        candidates = sorted(cache.items(), key=lambda kv: kv[1], reverse=True)
        exprs = []
        for key, score in candidates:
            if len(exprs) >= args.pool_capacity * args.pool_candidate_multiplier:
                break
            if score <= -1.0:
                continue
            try:
                exprs.append(safe_eval_expr(key))
            except Exception:
                continue
        new_pool = MseAlphaPool(
            capacity=args.pool_capacity,
            calculator=train_calc,
            ic_lower_bound=None,
            l1_alpha=args.l1_alpha,
            device=device,
        )
        if exprs:
            if args.filter_mode == "weak":
                new_pool.force_load_exprs(exprs, weights=[1.0 / min(len(exprs), args.pool_capacity)] * min(len(exprs), args.pool_capacity))
            else:
                new_pool.force_load_exprs(exprs)
        pool.exprs = new_pool.exprs
        pool.single_ics = new_pool.single_ics
        pool._weights = new_pool._weights
        pool._mutual_ics = new_pool._mutual_ics
        pool._extra_info = new_pool._extra_info
        pool.size = new_pool.size
        pool.best_obj = new_pool.best_obj
        pool.best_ic_ret = new_pool.best_ic_ret
        pool.eval_cnt = len(cache)

    def callback() -> None:
        nonlocal generation
        generation += 1
        if generation % args.eval_every_generations != 0 and generation != args.generations:
            return
        refresh_pool()
        scores = evaluate_pool(pool, valid_calc, test_calc)
        row = {
            "generation": generation,
            "cache_size": len(cache),
            "pool/size": pool.size,
            "pool/eval_cnt": pool.eval_cnt,
            "pool/best_ic_ret": pool.best_ic_ret,
            **scores,
        }
        append_metrics(metrics_path, row)
        print(row, flush=True)
        with open(os.path.join(run_dir, f"{generation}_snapshot.json"), "w") as f:
            json.dump({"cache": cache, "pool": pool.to_json_dict()}, f, indent=2)

    terminals = [
        "open_",
        "close",
        "high",
        "low",
        "volume",
        "vwap",
        *[f"Constant({v})" for v in [-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]],
    ]
    x_train = np.array([terminals])
    y_train = np.array([[1]])
    metric_obj = make_fitness(function=metric, greater_is_better=True)
    function_set = [make_function(**func._asdict()) for func in generic_funcs]
    estimator = SymbolicRegressor(
        population_size=args.population_size,
        generations=args.generations,
        init_depth=(2, args.init_max_depth),
        tournament_size=args.tournament_size,
        stopping_criteria=1.0,
        p_crossover=0.3,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01,
        p_point_mutation=0.1,
        p_point_replace=0.6,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.0,
        random_state=seed,
        function_set=function_set,
        metric=metric_obj,
        const_range=None,
        n_jobs=args.n_jobs,
    )
    estimator.fit(x_train, y_train, callback=callback)
    refresh_pool()
    scores = evaluate_pool(pool, valid_calc, test_calc)
    final_row = {
        "generation": args.generations,
        "cache_size": len(cache),
        "pool/size": pool.size,
        "pool/eval_cnt": pool.eval_cnt,
        "pool/best_ic_ret": pool.best_ic_ret,
        **scores,
    }
    append_metrics(metrics_path, final_row)
    with open(os.path.join(run_dir, "final_pool.json"), "w") as f:
        json.dump(pool.to_json_dict(), f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seeds", default="0")
    parser.add_argument("--method_name", default="gp")
    parser.add_argument("--output_root", default="/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_manual")
    parser.add_argument("--qlib_data_path", default="/root/autodl-tmp/cn_data_akshare_2010_2026")
    parser.add_argument("--qlib_kernels", type=int, default=1)
    parser.add_argument("--device_str", default="auto")
    parser.add_argument("--instruments", default="csi300")
    parser.add_argument("--pool_capacity", type=int, default=10)
    parser.add_argument("--population_size", type=int, default=500)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--tournament_size", type=int, default=120)
    parser.add_argument("--init_max_depth", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--eval_every_generations", type=int, default=2)
    parser.add_argument("--pool_candidate_multiplier", type=int, default=5)
    parser.add_argument("--filter_mode", choices=["weak", "strong"], default="strong")
    parser.add_argument("--l1_alpha", type=float, default=5e-3)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--train_start", default="2010-01-04")
    parser.add_argument("--train_end", default="2021-12-31")
    parser.add_argument("--valid_start", default="2022-01-04")
    parser.add_argument("--valid_end", default="2023-12-29")
    parser.add_argument("--test_start", default="2024-01-02")
    parser.add_argument("--test_end", default="2026-05-28")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for seed in parse_seeds(args.random_seeds):
        run_one(args, seed)




