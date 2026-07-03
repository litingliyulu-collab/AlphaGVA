from pathlib import Path
import pandas as pd

# Weight inferred from rl_v1 mixed formula and date split: compute exact qlib n_days via saved metrics relation is enough
# We use StockData to get exact valid/test n_days for the same qlib split.
from alphagen_qlib.stock_data import initialize_qlib, StockData
import torch
initialize_qlib('/root/autodl-tmp/cn_data_akshare_2010_2026', kernels=1)
valid_days = StockData('csi300','2022-01-04','2023-12-29',device=torch.device('cpu')).n_days
test_days = StockData('csi300','2024-01-02','2026-05-28',device=torch.device('cpu')).n_days
wv = valid_days/(valid_days+test_days)
wt = test_days/(valid_days+test_days)
print('__DAYS__', valid_days, test_days, 'wv', wv, 'wt', wt)

paths=[]
root=Path('/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_20260628_223714')
for method in ['gp','random_search']:
    for seed in [0,1,2]:
        dirs=sorted((root/f'{method}_s{seed}'/'results').glob('*'))
        # use latest non-empty metrics
        best=None
        for d in dirs:
            f=d/'metrics.csv'
            if f.exists():
                try:
                    df=pd.read_csv(f)
                except Exception:
                    continue
                if len(df): best=f
        paths.append((method,seed,best))
rows=[]
for method,seed,f in paths:
    df=pd.read_csv(f)
    last=df.iloc[-1]
    vi=float(last['valid/ic_mean']); vr=float(last['valid/rank_ic_mean'])
    ti=float(last['test/ic_mean']); tr=float(last['test/rank_ic_mean'])
    rows.append({
        'method':method,'seed':seed,
        'test_ic':ti,'test_rank_ic':tr,
        'valid_ic':vi,'valid_rank_ic':vr,
        'mixed_ic':wv*vi+wt*ti,
        'mixed_rank_ic':wv*vr+wt*tr,
        'eval_cnt':float(last.get('pool/eval_cnt', float('nan'))),
        'best_ic_ret':float(last.get('pool/best_ic_ret', float('nan'))),
    })

df=pd.DataFrame(rows)
print('__ROWS__')
print(df.to_string(index=False))
print('__AGG_TEST__')
print(df.groupby('method')[['test_ic','test_rank_ic','valid_ic','valid_rank_ic','mixed_ic','mixed_rank_ic']].agg(['mean','std']).to_string())
