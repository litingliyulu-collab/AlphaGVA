#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('/root/autodl-tmp/gva_factor_experiments/compiled/csi300_chapter4')
OUT.mkdir(parents=True, exist_ok=True)

def last_row(path):
    df = pd.read_csv(path)
    return df.iloc[-1]

def summarize(rows):
    df = pd.DataFrame(rows)
    out = []
    for (group, method), g in df.groupby(['group','method'], sort=False):
        out.append({
            'group': group,
            'method': method,
            'n': len(g),
            'IC_mean': g['ic'].mean(),
            'IC_std': g['ic'].std(ddof=1) if len(g)>1 else np.nan,
            'RankIC_mean': g['rank_ic'].mean(),
            'RankIC_std': g['rank_ic'].std(ddof=1) if len(g)>1 else np.nan,
            'seeds': ','.join(map(str, sorted(g['seed'].tolist()))),
        })
    return pd.DataFrame(out)

rows = []
def add_main(method, group, base, dirs, seeds=(0,1,2)):
    for seed, d in zip(seeds, dirs):
        files = list((Path(base)/d/'results').glob('*/metrics.csv'))
        if not files:
            print('MISS', method, seed, Path(base)/d)
            continue
        r = last_row(files[0])
        rows.append({'group':group,'method':method,'seed':seed,'ic':float(r['test/ic_2']),'rank_ic':float(r['test/rank_ic_2']),'path':str(files[0])})

def add_generic(method, group, paths, seeds=(0,1,2), ic_col='test/ic_mean', rank_col='test/rank_ic_mean'):
    for seed, p in zip(seeds, paths):
        p = Path(p)
        if not p.exists():
            print('MISS', method, seed, p)
            continue
        r = last_row(p)
        rows.append({'group':group,'method':method,'seed':seed,'ic':float(r[ic_col]),'rank_ic':float(r[rank_col]),'path':str(p)})

main = '/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/main_compare_20260628_211640'
add_main('AlphaGen', 'Main RL baselines', main, [f'ppo_s{s}' for s in [0,1,2]])
add_main('GVA', 'Proposed method', main, [f'full_gva25_s{s}' for s in [0,1,2]])
add_main('Critic-GVA', 'Ablation', main, [f'critic_gva25_s{s}' for s in [0,1,2]])

actor = '/root/autodl-tmp/gva_factor_experiments/runs_newdata/actor_gva_20260630_193023'
add_main('Actor-GVA', 'Ablation', actor, [f'actor_gva25_s{s}' for s in [0,1,2]])

suite = Path('/root/autodl-tmp/gva_factor_experiments/runs_newdata/filter_xgb_suite_20260629_145720')
add_generic('PPO-filter', 'Filter baselines', [next((suite/f'ppo_filter_weak_s{s}'/'results').glob('*/metrics.csv')) for s in [0,1,2]])
add_generic('GP', 'Symbolic/ML baselines', [next((suite/f'gp_filter_strong_s{s}'/'results').glob('*/metrics.csv')) for s in [0,1,2]])
add_generic('GVA-filter', 'Ablation', [next((suite/f'gva_filter_s{s}'/'results').glob('*/metrics.csv')) for s in [0,1,2]])
add_generic('XGBoost', 'Symbolic/ML baselines', [next((suite/f'xgboost_s{s}'/'results').glob('*/metrics.csv')) for s in [0,1,2]])

ml = Path('/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/ml_baselines_20260629_101312')
add_generic('LightGBM', 'Symbolic/ML baselines', [next((ml/f'lightgbm_s{s}'/'results').glob('*/metrics.csv')) for s in [0,1,2]])
add_generic('MLP', 'Symbolic/ML baselines', [next((ml/f'mlp_s{s}'/'results').glob('*/metrics.csv')) for s in [0,1,2]])

# A2C uses custom callback mixed fields: test/ic_2 = pure test
add_main('A2C', 'Advanced RL baselines', '/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629', [
    'a2c_baseline_s0', 'a2c_baseline_s1', 'a2c_baseline_s2'
])
# Above path has multiple result subdirs under each method; add_main will pick first metrics in results glob. Replace with explicit latest desired below if needed.
# Correct A2C rows if glob picked stale seed1 partial: remove A2C and re-add explicit files.
rows = [r for r in rows if r['method'] != 'A2C']
a2c_paths = [
'/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629/a2c_baseline_s0/results/csi300_10_0_20260629193800_a2c_original/metrics.csv',
'/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629/a2c_baseline_s1/results/csi300_10_1_20260629205636_a2c_original_resume/metrics.csv',
'/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629/a2c_baseline_s2/results/csi300_10_2_20260629210123_a2c_original_fresh/metrics.csv',
]
for seed,p in enumerate(a2c_paths):
    r=last_row(p); rows.append({'group':'Advanced RL baselines','method':'A2C','seed':seed,'ic':float(r['test/ic_2']),'rank_ic':float(r['test/rank_ic_2']),'path':p})

rein = Path('/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/reinforce_qfr_20260629_213609')
add_generic('REINFORCE', 'Advanced RL baselines', [next((rein/f'reinforce_s{s}'/'results').glob('*/metrics.csv')) for s in [0,1,2]])
qfr = Path('/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/qfr_fixed_20260629_215010')
add_generic('QFR', 'Advanced RL baselines', [next((qfr/f'qfr_s{s}'/'results').glob('*/metrics.csv')) for s in [0,1,2]])

aff = Path('/root/autodl-tmp/gva_factor_experiments/backtests/filter_suite_with_locked_top50_20260629/aff_official_adapted_s0/aff_official_metrics.csv')
r = pd.read_csv(aff).iloc[-1]
rows.append({'group':'Two-stage baselines','method':'AFF','seed':0,'ic':float(r['test/ic_mean']),'rank_ic':float(r['test/rank_ic_mean']),'path':str(aff)})

raw = pd.DataFrame(rows)
raw.to_csv(OUT/'csi300_metric_rows.csv', index=False)
summary = summarize(rows)
comparison_order = ['GVA','AlphaGen','PPO-filter','GP','XGBoost','LightGBM','MLP','A2C','REINFORCE','QFR','AFF']
ablation_order = ['GVA','Critic-GVA','Actor-GVA','GVA-filter','PPO-filter']
comparison = summary[summary['method'].isin(comparison_order)].copy()
comparison['order'] = comparison['method'].map({m:i for i,m in enumerate(comparison_order)})
comparison = comparison.sort_values('order').drop(columns='order')
ablation = summary[summary['method'].isin(ablation_order)].copy()
ablation['order'] = ablation['method'].map({m:i for i,m in enumerate(ablation_order)})
ablation = ablation.sort_values('order').drop(columns='order')
comparison.to_csv(OUT/'table_csi300_comparison_metrics.csv', index=False)
ablation.to_csv(OUT/'table_csi300_ablation_metrics.csv', index=False)
print('COMPARISON')
print(comparison.to_string(index=False))
print('ABLATION')
print(ablation.to_string(index=False))
print('OUT', OUT)
