from pathlib import Path
import pandas as pd
roots = [
    Path('/root/alpha_1203/gva_factor_experiments/runs_newdata/main_compare_20260628_211640'),
    Path('/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_20260628_223714'),
    Path('/root/alpha_1203/gva_factor_experiments/runs_newdata/ml_baselines_20260629_101312'),
]
rows=[]
for root in roots:
    for f in root.glob('*/results/*/metrics.csv'):
        method_seed=f.parts[-4]
        if method_seed == 'logs':
            continue
        try:
            df=pd.read_csv(f)
        except Exception:
            continue
        if len(df)==0:
            continue
        # Keep only fixed random batch, ignore earlier failed empty dirs automatically by len(df)==0.
        run_name=f.parent.name
        last=df.iloc[-1]
        method=method_seed.rsplit('_s',1)[0] if '_s' in method_seed else method_seed
        try:
            seed=int(method_seed.rsplit('_s',1)[1]) if '_s' in method_seed else None
        except Exception:
            seed=None
        rows.append({
            'root': root.name,
            'run': run_name,
            'method': method,
            'seed': seed,
            'timestep': last.get('timestep', last.get('attempt', last.get('generation', ''))),
            'eval_cnt': last.get('pool/eval_cnt', ''),
            'best_ic_ret': last.get('pool/best_ic_ret', ''),
            'test_ic_mean': last.get('test/ic_mean', ''),
            'test_rank_ic_mean': last.get('test/rank_ic_mean', ''),
        })
df=pd.DataFrame(rows)
# choose latest per method/seed/root and remove duplicate random old failed via latest run name
if len(df):
    df=df.sort_values(['root','method','seed','run']).groupby(['root','method','seed'], dropna=False).tail(1)
print('__ROWS__')
print(df.sort_values(['root','method','seed']).to_string(index=False))
print('__AGG__')
for col in ['test_ic_mean','test_rank_ic_mean','best_ic_ret','eval_cnt']:
    df[col]=pd.to_numeric(df[col], errors='coerce')
agg=df.groupby('method')[['test_ic_mean','test_rank_ic_mean','best_ic_ret','eval_cnt']].agg(['mean','std'])
print(agg.to_string())
