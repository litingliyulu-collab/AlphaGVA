#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path('/root/alpha_1203/gva_factor_experiments/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import backtest_filter_suite as b

METHOD_LABELS = dict(b.METHOD_LABELS)
METHOD_LABELS['locked_warmpool_gva'] = 'Locked-WarmPool-GVA'
METHOD_ORDER = ['ppo_filter_weak', 'ppo_filter', 'gp_filter_weak', 'gp_filter_strong', 'gva_filter', 'xgboost', 'lightgbm', 'mlp', 'alphagen_ppo', 'critic_gva', 'full_gva', 'locked_warmpool_gva']


def locked_specs():
    return [
        {'method': 'locked_warmpool_gva', 'seed': '0', 'kind': 'pool', 'path': '/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/locked_warmpool_gva_continue_20260629_183501/locked_warm_full_gva25_s0_continue/results/csi300_10_0_20260629183504_rl_original/1920_steps_pool.json'},
        {'method': 'locked_warmpool_gva', 'seed': '1', 'kind': 'pool', 'path': '/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/locked_warmpool_gva_continue_20260629_183501/locked_warm_full_gva25_s1_continue/results/csi300_10_1_20260629183504_rl_original/3520_steps_pool.json'},
        {'method': 'locked_warmpool_gva', 'seed': '2', 'kind': 'pool', 'path': '/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/locked_warmpool_gva_continue_20260629_183501/locked_warm_full_gva25_s2_continue/results/csi300_10_2_20260629183504_rl_original/2752_steps_pool.json'},
    ]


def plot(reports, out_dir):
    for col, fname, title, ylabel in [
        ('return', 'cumulative_return_by_method_with_locked.png', 'Cumulative Return on Test Set', 'Cumulative Return'),
        ('excess_return', 'cumulative_excess_return_by_method_with_locked.png', 'Cumulative Excess Return on Test Set', 'Cumulative Excess Return'),
    ]:
        plt.figure(figsize=(12, 6))
        for method in METHOD_ORDER:
            reps = [r for key, r in reports.items() if key.startswith(method + '_s')]
            if not reps:
                continue
            ret = pd.concat([r[col].rename(i) for i, r in enumerate(reps)], axis=1).fillna(0.0).mean(axis=1)
            curve = (1 + ret).cumprod() - 1
            lw = 3 if method in ['full_gva', 'locked_warmpool_gva'] else 1.8
            plt.plot(curve.index, curve.values, label=METHOD_LABELS.get(method, method), linewidth=lw)
        if col == 'return' and reports:
            first = next(iter(reports.values()))
            bench = (1 + first['bench'].fillna(0.0)).cumprod() - 1
            plt.plot(bench.index, bench.values, label='CSI300 universe EW', color='black', linestyle=':', linewidth=2)
        plt.axhline(0, color='gray', linewidth=0.8)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Date')
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=180)
        plt.close()


def main():
    out_dir = Path('/root/autodl-tmp/gva_factor_experiments/backtests/filter_suite_with_locked_top50_20260629')
    out_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        output_dir=str(out_dir),
        qlib_data_path='/root/autodl-tmp/cn_data_akshare_2010_2026',
        main_compare_root='/root/alpha_1203/gva_factor_experiments/runs_newdata/main_compare_20260628_211640',
        ppo_filter_root='/nonexistent/ppo_filter_root',
        horizontal_root='/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_20260628_223714',
        ml_root='/root/alpha_1203/gva_factor_experiments/runs_newdata/ml_baselines_20260629_101312',
        filter_suite_root='/root/autodl-tmp/gva_factor_experiments/runs_newdata/filter_xgb_suite_20260629_145720',
        instruments='csi300',
        test_start='2024-01-02',
        test_end='2026-05-28',
        topk=50,
        cost_rate=0.0015,
        device='cuda:0',
    )
    b.initialize_qlib(args.qlib_data_path, kernels=1)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data = b.StockData(args.instruments, args.test_start, args.test_end, device=device)
    returns = b.make_forward_returns(data)
    specs = b.discover_inputs(args) + locked_specs()
    (out_dir / 'input_specs.json').write_text(json.dumps(specs, indent=2), encoding='utf-8')
    rows, failures, reports = [], [], {}
    for spec in specs:
        method, seed = spec['method'], int(spec['seed'])
        key = f'{method}_s{seed}'
        print('BACKTEST', key, spec['kind'], spec['path'], flush=True)
        try:
            scores = b.pool_scores(data, spec['path'], device) if spec['kind'] == 'pool' else b.npz_scores(data, spec['path'], device)
            report, positions = b.calc_backtest(scores, returns, args.topk, args.cost_rate)
            reports[key] = report
            run_dir = out_dir / key
            run_dir.mkdir(exist_ok=True)
            report.to_csv(run_dir / 'daily_report.csv')
            positions.to_csv(run_dir / 'positions.csv', index=False)
            row = {'method': method, 'method_label': METHOD_LABELS.get(method, method), 'seed': seed, **b.summarize(report), 'path': spec['path']}
            rows.append(row)
        except Exception as exc:
            print('FAILED', key, type(exc).__name__, exc, flush=True)
            failures.append({**spec, 'key': key, 'error_type': type(exc).__name__, 'error': str(exc)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / 'metrics_by_seed.csv', index=False)
    summary = metrics.groupby(['method', 'method_label']).agg({
        'cum_return': ['mean', 'std'],
        'excess_cum_return': ['mean', 'std'],
        'annual_return': ['mean', 'std'],
        'information_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std'],
        'avg_turnover': ['mean', 'std'],
    })
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns.to_flat_index()]
    summary = summary.reset_index().sort_values('excess_cum_return_mean', ascending=False)
    summary.to_csv(out_dir / 'metrics_summary.csv', index=False)
    pd.DataFrame(failures).to_csv(out_dir / 'failed_specs.csv', index=False)
    plot(reports, out_dir)
    print(summary.to_string(index=False))
    print(out_dir)

if __name__ == '__main__':
    main()

