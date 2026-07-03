#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from alphagen.data.expression import (
    Abs, Add, CSRank, Constant, Corr, Cov, Delta, Div, EMA, Expression, Feature,
    Greater, Less, Log, Mad, Max, Mean, Med, Min, Mul, OutOfDataRangeError,
    Rank, Ref, Sign, Std, Sub, Sum, Var, WMA,
)
from alphagen.data.parser import ExpressionParser, ExpressionParsingError
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib

PARSER = ExpressionParser([Abs, Sign, Log, CSRank, Add, Sub, Mul, Div, Greater, Less, Ref, Mean, Sum, Std, Var, Max, Min, Med, Mad, Rank, Delta, WMA, EMA, Cov, Corr])


def build_target() -> Expression:
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.replace('[', '').replace(']', '').split(',') if x.strip()]


def parse_step(path: str) -> int:
    m = re.search(r'(\d+)_steps_pool\.json$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def build_data(args: argparse.Namespace, device: torch.device):
    initialize_qlib(args.qlib_data_path, kernels=args.qlib_kernels)
    target = build_target()
    train = StockData(args.instruments, args.train_start, args.train_end, device=device)
    valid = StockData(args.instruments, args.valid_start, args.valid_end, device=device)
    test = StockData(args.instruments, args.test_start, args.test_end, device=device)
    return QLibStockDataCalculator(train, target), QLibStockDataCalculator(valid, target), QLibStockDataCalculator(test, target)


def collect_exprs(args: argparse.Namespace, seed: int) -> List[Expression]:
    pattern = os.path.join(args.source_root, args.source_pattern.format(seed=seed), 'results', '*', '*_steps_pool.json')
    files = sorted(glob.glob(pattern), key=parse_step)
    if args.max_snapshots > 0 and len(files) > args.max_snapshots:
        idx = np.linspace(0, len(files) - 1, args.max_snapshots).round().astype(int)
        files = [files[i] for i in sorted(set(idx))]
    seen = set()
    exprs: List[Expression] = []
    for path in files:
        try:
            raw = json.load(open(path))
        except Exception:
            continue
        for s in raw.get('exprs', []):
            if s in seen:
                continue
            try:
                expr = PARSER.parse(s)
            except Exception:
                continue
            seen.add(s)
            exprs.append(expr)
    return exprs


def score_exprs(exprs: List[Expression], calc: QLibStockDataCalculator) -> Dict[str, Tuple[float, Expression]]:
    scores: Dict[str, Tuple[float, Expression]] = {}
    for expr in exprs:
        key = str(expr)
        if key in scores:
            continue
        try:
            score = float(calc.calc_single_IC_ret(expr))
        except (OutOfDataRangeError, FloatingPointError, RuntimeError, ValueError, ZeroDivisionError, TypeError, ExpressionParsingError):
            continue
        if np.isfinite(score):
            scores[key] = (score, expr)
    return scores


def rank_score(score: float, mode: str) -> float:
    return abs(score) if mode == 'strong' else max(score, 0.0)


def build_filtered_pool(scores: Dict[str, Tuple[float, Expression]], calc: QLibStockDataCalculator, args: argparse.Namespace, device: torch.device) -> MseAlphaPool:
    selected: List[Expression] = []
    candidates = sorted(scores.values(), key=lambda item: rank_score(item[0], args.filter_mode), reverse=True)
    for score, expr in candidates:
        if len(selected) >= args.pool_capacity:
            break
        if args.filter_mode == 'weak' and score <= 0:
            continue
        redundant = False
        for old in selected:
            try:
                mutual = float(calc.calc_mutual_IC(expr, old))
            except Exception:
                redundant = True
                break
            if np.isfinite(mutual) and abs(mutual) > args.mutual_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(expr)
    pool = MseAlphaPool(capacity=args.pool_capacity, calculator=calc, ic_lower_bound=None, l1_alpha=args.l1_alpha, device=device)
    if selected:
        if args.filter_mode == 'weak':
            pool.force_load_exprs(selected, weights=[1.0 / len(selected)] * len(selected))
        else:
            pool.force_load_exprs(selected)
    pool.eval_cnt = len(scores)
    return pool


def evaluate(pool: MseAlphaPool, valid_calc: QLibStockDataCalculator, test_calc: QLibStockDataCalculator):
    if pool.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    valid_ic, valid_ric = pool.test_ensemble(valid_calc)
    test_ic, test_ric = pool.test_ensemble(test_calc)
    return valid_ic, valid_ric, test_ic, test_ric


def run_one(args: argparse.Namespace, seed: int) -> None:
    reseed_everything(seed)
    device = torch.device('cuda:0' if args.device_str == 'auto' and torch.cuda.is_available() else args.device_str)
    train_calc, valid_calc, test_calc = build_data(args, device)
    exprs = collect_exprs(args, seed)
    scores = score_exprs(exprs, train_calc)
    pool = build_filtered_pool(scores, train_calc, args, device)
    valid_ic, valid_ric, test_ic, test_ric = evaluate(pool, valid_calc, test_calc)
    run_dir = Path(args.output_root) / f'{args.method_name}_s{seed}' / 'results' / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_gva_filter"
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {
        'method': args.method_name,
        'seed': seed,
        'filter_mode': args.filter_mode,
        'source_pattern': args.source_pattern,
        'snapshot_exprs': len(exprs),
        'valid_exprs': len(scores),
        'pool/size': pool.size,
        'pool/eval_cnt': pool.eval_cnt,
        'pool/best_ic_ret': pool.best_ic_ret,
        'valid/ic_mean': valid_ic,
        'valid/rank_ic_mean': valid_ric,
        'test/ic_mean': test_ic,
        'test/rank_ic_mean': test_ric,
    }
    with open(run_dir / 'metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader(); writer.writerow(row)
    with open(run_dir / 'final_pool.json', 'w') as f:
        json.dump(pool.to_json_dict(), f, indent=2)
    with open(run_dir / 'candidate_scores.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['score', 'expr'])
        for score, expr in sorted(scores.values(), key=lambda item: rank_score(item[0], args.filter_mode), reverse=True):
            writer.writerow([score, str(expr)])
    print(row, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--random_seeds', default='0')
    p.add_argument('--method_name', default='gva_filter')
    p.add_argument('--source_root', default='/root/alpha_1203/gva_factor_experiments/runs_newdata/main_compare_20260628_211640')
    p.add_argument('--source_pattern', default='full_gva25_s{seed}')
    p.add_argument('--output_root', default='/root/autodl-tmp/gva_factor_experiments/runs_newdata/gva_filter_manual')
    p.add_argument('--qlib_data_path', default='/root/autodl-tmp/cn_data_akshare_2010_2026')
    p.add_argument('--qlib_kernels', type=int, default=1)
    p.add_argument('--device_str', default='auto')
    p.add_argument('--instruments', default='csi300')
    p.add_argument('--pool_capacity', type=int, default=10)
    p.add_argument('--filter_mode', choices=['weak', 'strong'], default='strong')
    p.add_argument('--mutual_threshold', type=float, default=0.99)
    p.add_argument('--l1_alpha', type=float, default=5e-3)
    p.add_argument('--max_snapshots', type=int, default=0)
    p.add_argument('--train_start', default='2010-01-04')
    p.add_argument('--train_end', default='2021-12-31')
    p.add_argument('--valid_start', default='2022-01-04')
    p.add_argument('--valid_end', default='2023-12-29')
    p.add_argument('--test_start', default='2024-01-02')
    p.add_argument('--test_end', default='2026-05-28')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for seed in parse_seeds(args.random_seeds):
        run_one(args, seed)
