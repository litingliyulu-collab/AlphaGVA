#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

SCRIPT_DIR = Path('/root/alpha_1203/gva_factor_experiments/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import backtest_filter_suite as b

OUT = Path('/root/autodl-tmp/gva_factor_experiments/backtests/csi300_extra_rl_top50_20260630')
OUT.mkdir(parents=True, exist_ok=True)

SPEC = []
def add(method, seed, path):
    SPEC.append({'method': method, 'seed': str(seed), 'kind': 'pool', 'path': path})

# Actor-GVA
for s in [0, 1, 2]:
    add('actor_gva', s, f'/root/autodl-tmp/gva_factor_experiments/runs_newdata/actor_gva_20260630_193023/actor_gva25_s{s}/results/csi300_10_{s}_20260630193026_rl_original/30000_steps_pool.json')
# A2C
for s in [0, 1, 2]:
    if s == 0:
        path = '/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629/a2c_baseline_s0/results/csi300_10_0_20260629193800_a2c_original/30016_steps_pool.json'
    elif s == 1:
        path = '/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629/a2c_baseline_s1/results/csi300_10_1_20260629205636_a2c_original_resume/final_pool.json'
    else:
        path = '/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629/a2c_baseline_s2/results/csi300_10_2_20260629210123_a2c_original_fresh/final_pool.json'
    add('a2c', s, path)
# REINFORCE
for s in [0, 1, 2]:
    add('reinforce', s, f'/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/reinforce_qfr_20260629_213609/reinforce_s{s}/results/csi300_10_{s}_' + ('20260629213617' if s==0 else '20260629220801' if s==1 else '20260629224136') + '_reinforce/final_pool.json')
# QFR fixed
for s in [0, 1, 2]:
    ts = ['20260629215016', '20260629223934', '20260629232826'][s]
    add('qfr', s, f'/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/qfr_fixed_20260629_215010/qfr_s{s}/results/csi300_10_{s}_{ts}_qfr/final_pool.json')

(OUT / 'input_specs.json').write_text(json.dumps(SPEC, indent=2), encoding='utf-8')
args = SimpleNamespace(
    qlib_data_path='/root/autodl-tmp/cn_data_akshare_2010_2026',
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
rows, reports, failures = [], {}, []
for spec in SPEC:
    key = f"{spec['method']}_s{spec['seed']}"
    print('BACKTEST', key, spec['path'], flush=True)
    try:
        scores = b.pool_scores(data, spec['path'], device)
        report, positions = b.calc_backtest(scores, returns, args.topk, args.cost_rate)
        reports[key] = report
        run_dir = OUT / key
        run_dir.mkdir(exist_ok=True)
        report.to_csv(run_dir / 'daily_report.csv')
        positions.to_csv(run_dir / 'positions.csv', index=False)
        row = {'method': spec['method'], 'method_label': {'actor_gva':'Actor-GVA','a2c':'A2C','reinforce':'REINFORCE','qfr':'QFR'}[spec['method']], 'seed': int(spec['seed']), **b.summarize(report), 'path': spec['path']}
        rows.append(row)
    except Exception as exc:
        print('FAILED', key, type(exc).__name__, exc, flush=True)
        failures.append({**spec, 'key': key, 'error_type': type(exc).__name__, 'error': str(exc)})
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / 'metrics_by_seed.csv', index=False)
if failures:
    pd.DataFrame(failures).to_csv(OUT / 'failed_specs.csv', index=False)
summary = metrics.groupby(['method','method_label']).agg({
    'cum_return':['mean','std'], 'excess_cum_return':['mean','std'], 'annual_return':['mean','std'],
    'information_ratio':['mean','std'], 'max_drawdown':['mean','std'], 'avg_turnover':['mean','std']
})
summary.columns = ['_'.join(c).strip('_') for c in summary.columns.to_flat_index()]
summary.reset_index().to_csv(OUT / 'metrics_summary.csv', index=False)
print(metrics[['method_label','seed','cum_return','excess_cum_return','information_ratio','max_drawdown']].to_string(index=False))
print('OUT', OUT)
