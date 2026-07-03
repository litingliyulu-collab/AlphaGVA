#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alphagen_qlib.stock_data import StockData, FeatureType, initialize_qlib
from alphagen_qlib.utils import load_alpha_pool_by_path


METHOD_LABELS = {
    'alphagen_ppo': 'AlphaGen-PPO',
    'critic_gva': 'Critic-GVA',
    'full_gva': 'Full-GVA',
    'ppo_filter': 'PPO_filter',
    'random_search': 'Random Search',
    'gp': 'GP_filter',
    'lightgbm': 'LightGBM',
    'mlp': 'MLP',
}
METHOD_ORDER = ['ppo_filter', 'random_search', 'gp', 'lightgbm', 'mlp', 'alphagen_ppo', 'critic_gva', 'full_gva']


def parse_step(path: Path) -> int:
    m = re.search(r'(\d+)_steps_pool\.json$', path.name)
    return int(m.group(1)) if m else -1


def latest_one(pattern: str) -> str:
    xs = glob.glob(pattern)
    if not xs:
        raise FileNotFoundError(pattern)
    return max(xs, key=os.path.getmtime)


def final_step_pool(pattern: str) -> str:
    xs = [Path(x) for x in glob.glob(pattern)]
    if not xs:
        raise FileNotFoundError(pattern)
    return str(sorted(xs, key=parse_step)[-1])


def make_forward_returns(data: StockData) -> pd.DataFrame:
    close_idx = list(data._features).index(FeatureType.CLOSE)
    bt = data.max_backtrack_days
    n = data.n_days
    close = data.data[:, close_idx, :]
    ret = close[bt + 1: bt + n + 1] / close[bt: bt + n] - 1.0
    df = data.make_dataframe(ret, columns=['ret']).reset_index()
    df.columns = ['datetime', 'instrument', 'ret']
    return df.pivot(index='datetime', columns='instrument', values='ret').sort_index()


def pool_scores(data: StockData, pool_path: str, device: torch.device) -> pd.DataFrame:
    exprs, weights = load_alpha_pool_by_path(pool_path)
    with torch.no_grad():
        if not exprs:
            score = torch.zeros((data.n_days, data.n_stocks), dtype=torch.float32, device=device)
        else:
            factors = [expr.evaluate(data) * float(weights[i]) for i, expr in enumerate(exprs)]
            score = torch.sum(torch.stack(factors, dim=0), dim=0)
    df = data.make_dataframe(score, columns=['score']).reset_index()
    df.columns = ['datetime', 'instrument', 'score']
    return df.pivot(index='datetime', columns='instrument', values='score').sort_index()


def npz_scores(data: StockData, pred_path: str, device: torch.device) -> pd.DataFrame:
    z = np.load(pred_path, allow_pickle=True)
    pred = z['pred']
    if pred.shape != (data.n_days, data.n_stocks):
        raise ValueError(f'{pred_path} pred shape {pred.shape} != {(data.n_days, data.n_stocks)}')
    tensor = torch.tensor(pred, dtype=torch.float32, device=device)
    df = data.make_dataframe(tensor, columns=['score']).reset_index()
    df.columns = ['datetime', 'instrument', 'score']
    return df.pivot(index='datetime', columns='instrument', values='score').sort_index()


def calc_backtest(scores: pd.DataFrame, returns: pd.DataFrame, topk: int, cost_rate: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_idx = scores.index.intersection(returns.index)
    common_cols = scores.columns.intersection(returns.columns)
    scores = scores.loc[common_idx, common_cols]
    returns = returns.loc[common_idx, common_cols]
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        valid = row.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        picks = valid.nlargest(min(topk, len(valid))).index
        weights.loc[dt, picks] = 1.0 / len(picks)
    gross = (weights * returns).sum(axis=1, min_count=1).fillna(0.0)
    bench = returns.mean(axis=1, skipna=True).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * cost_rate
    net = gross - cost
    excess = net - bench
    report = pd.DataFrame({'return_gross': gross, 'cost': cost, 'return': net, 'bench': bench, 'excess_return': excess, 'turnover': turnover})
    positions = weights.stack().rename('weight').reset_index()
    positions.columns = ['datetime', 'instrument', 'weight']
    positions = positions[positions['weight'] != 0]
    return report, positions


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())


def summarize(report: pd.DataFrame) -> Dict[str, float]:
    r = report['return'].fillna(0.0)
    b = report['bench'].fillna(0.0)
    e = report['excess_return'].fillna(0.0)
    n = max(len(r), 1)
    ann = 252
    nav = (1 + r).cumprod()
    bnav = (1 + b).cumprod()
    enav = (1 + e).cumprod()
    return {
        'days': n,
        'cum_return': float(nav.iloc[-1] - 1),
        'bench_cum_return': float(bnav.iloc[-1] - 1),
        'excess_cum_return': float(enav.iloc[-1] - 1),
        'annual_return': float(nav.iloc[-1] ** (ann / n) - 1),
        'bench_annual_return': float(bnav.iloc[-1] ** (ann / n) - 1),
        'excess_annual_return': float(enav.iloc[-1] ** (ann / n) - 1),
        'sharpe': float((r.mean() / r.std(ddof=1)) * math.sqrt(ann)) if r.std(ddof=1) > 0 else np.nan,
        'information_ratio': float((e.mean() / e.std(ddof=1)) * math.sqrt(ann)) if e.std(ddof=1) > 0 else np.nan,
        'max_drawdown': max_drawdown(nav),
        'excess_max_drawdown': max_drawdown(enav),
        'avg_turnover': float(report['turnover'].mean()),
        'avg_cost': float(report['cost'].mean()),
        'win_rate': float((r > 0).mean()),
    }


def discover_inputs(args: argparse.Namespace) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    main = Path(args.main_compare_root)
    for seed in [0, 1, 2]:
        specs.append({'method': 'alphagen_ppo', 'seed': str(seed), 'kind': 'pool', 'path': final_step_pool(str(main / f'ppo_s{seed}' / 'results' / '*' / '*_steps_pool.json'))})
        specs.append({'method': 'critic_gva', 'seed': str(seed), 'kind': 'pool', 'path': final_step_pool(str(main / f'critic_gva25_s{seed}' / 'results' / '*' / '*_steps_pool.json'))})
        specs.append({'method': 'full_gva', 'seed': str(seed), 'kind': 'pool', 'path': final_step_pool(str(main / f'full_gva25_s{seed}' / 'results' / '*' / '*_steps_pool.json'))})
    ppo_root = Path(args.ppo_filter_root)
    if ppo_root.exists():
        for seed in [0, 1, 2]:
            specs.append({'method': 'ppo_filter', 'seed': str(seed), 'kind': 'pool', 'path': latest_one(str(ppo_root / f'ppo_filter_s{seed}' / 'results' / '*' / 'final_pool.json'))})
    hor = Path(args.horizontal_root)
    for seed in [0, 1, 2]:
        specs.append({'method': 'random_search', 'seed': str(seed), 'kind': 'pool', 'path': latest_one(str(hor / f'random_search_s{seed}' / 'results' / '*' / 'final_pool.json'))})
        specs.append({'method': 'gp', 'seed': str(seed), 'kind': 'pool', 'path': latest_one(str(hor / f'gp_s{seed}' / 'results' / '*' / 'final_pool.json'))})
    ml = Path(args.ml_root)
    for seed in [0, 1, 2]:
        specs.append({'method': 'lightgbm', 'seed': str(seed), 'kind': 'npz', 'path': latest_one(str(ml / f'lightgbm_s{seed}' / 'results' / '*' / 'pred_test.npz'))})
        specs.append({'method': 'mlp', 'seed': str(seed), 'kind': 'npz', 'path': latest_one(str(ml / f'mlp_s{seed}' / 'results' / '*' / 'pred_test.npz'))})
    return specs


def plot_method_curves(method_reports: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    for column, fname, title, ylabel in [
        ('return', 'cumulative_return_by_method.png', 'Cumulative Return on Test Set', 'Cumulative Return'),
        ('excess_return', 'cumulative_excess_return_by_method.png', 'Cumulative Excess Return on Test Set', 'Cumulative Excess Return'),
    ]:
        plt.figure(figsize=(12, 6))
        for method in METHOD_ORDER:
            reps = [r for key, r in method_reports.items() if key.startswith(method + '_s')]
            if not reps:
                continue
            ret = pd.concat([r[column].rename(i) for i, r in enumerate(reps)], axis=1).fillna(0.0).mean(axis=1)
            curve = (1 + ret).cumprod() - 1
            plt.plot(curve.index, curve.values, label=METHOD_LABELS.get(method, method), linewidth=2)
        if column == 'return' and method_reports:
            first = next(iter(method_reports.values()))
            bench = (1 + first['bench'].fillna(0.0)).cumprod() - 1
            plt.plot(bench.index, bench.values, label='CSI300 universe EW', color='black', linestyle=':', linewidth=2)
        plt.axhline(0, color='gray', linewidth=0.8)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Date')
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=180)
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='/root/autodl-tmp/gva_factor_experiments/backtests/unified_all_methods_top50')
    p.add_argument('--qlib_data_path', default='/root/autodl-tmp/cn_data_akshare_2010_2026')
    p.add_argument('--main_compare_root', default='/root/alpha_1203/gva_factor_experiments/runs_newdata/main_compare_20260628_211640')
    p.add_argument('--ppo_filter_root', default='')
    p.add_argument('--horizontal_root', default='/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_20260628_223714')
    p.add_argument('--ml_root', default='/root/alpha_1203/gva_factor_experiments/runs_newdata/ml_baselines_20260629_101312')
    p.add_argument('--instruments', default='csi300')
    p.add_argument('--test_start', default='2024-01-02')
    p.add_argument('--test_end', default='2026-05-28')
    p.add_argument('--topk', type=int, default=50)
    p.add_argument('--cost_rate', type=float, default=0.0015)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    initialize_qlib(args.qlib_data_path, kernels=1)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    data = StockData(args.instruments, args.test_start, args.test_end, device=device)
    returns = make_forward_returns(data)
    specs = discover_inputs(args)
    with open(out_dir / 'input_specs.json', 'w') as f:
        json.dump(specs, f, indent=2)

    rows = []
    failures = []
    reports: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        method = spec['method']
        seed = int(spec['seed'])
        key = f'{method}_s{seed}'
        print('BACKTEST', key, spec['kind'], spec['path'], flush=True)
        try:
            if spec['kind'] == 'pool':
                scores = pool_scores(data, spec['path'], device)
            else:
                scores = npz_scores(data, spec['path'], device)
        except Exception as exc:
            print('SKIP_FAILED', key, type(exc).__name__, str(exc), flush=True)
            failed = dict(spec)
            failed.update({'key': key, 'error_type': type(exc).__name__, 'error': str(exc)})
            failures.append(failed)
            continue
        report, positions = calc_backtest(scores, returns, args.topk, args.cost_rate)
        reports[key] = report
        run_dir = out_dir / key
        run_dir.mkdir(exist_ok=True)
        report.to_csv(run_dir / 'daily_report.csv')
        positions.to_csv(run_dir / 'positions.csv', index=False)
        summary = summarize(report)
        summary.update({'method': method, 'method_label': METHOD_LABELS.get(method, method), 'seed': seed, 'kind': spec['kind'], 'source_path': spec['path']})
        rows.append(summary)
    metrics = pd.DataFrame(rows)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / 'failed_specs.csv', index=False)
    if metrics.empty:
        raise RuntimeError('No backtest result was generated. See failed_specs.csv for details.')
    metrics.to_csv(out_dir / 'metrics_by_seed.csv', index=False)
    numeric = [c for c in metrics.columns if c not in ['method', 'method_label', 'seed', 'kind', 'source_path']]
    summary = metrics.groupby(['method', 'method_label'])[numeric].agg(['mean', 'std'])
    summary.to_csv(out_dir / 'metrics_summary.csv')
    plot_method_curves(reports, out_dir)
    print('OUTPUT_DIR', out_dir)
    print(metrics[['method_label', 'seed', 'cum_return', 'annual_return', 'excess_cum_return', 'information_ratio', 'max_drawdown', 'avg_turnover']].to_string(index=False))


if __name__ == '__main__':
    main()

