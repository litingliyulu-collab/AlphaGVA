from pathlib import Path
import pandas as pd
run = Path('/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_20260628_223714')
rows = []
for f in sorted(run.glob('*/results/*/metrics.csv')):
    method_seed = f.parts[-4]
    run_name = f.parent.name
    try:
        df = pd.read_csv(f)
    except Exception as e:
        rows.append([method_seed, run_name, 'read_error', str(e), '', '', '', '', ''])
        continue
    if len(df) == 0:
        rows.append([method_seed, run_name, 'empty', 0, '', '', '', '', ''])
        continue
    last = df.iloc[-1]
    step_col = 'attempt' if 'attempt' in df.columns else 'generation'
    rows.append([
        method_seed,
        run_name,
        'ok',
        len(df),
        last.get(step_col, ''),
        last.get('pool/eval_cnt', ''),
        last.get('pool/best_ic_ret', ''),
        last.get('test/ic_mean', ''),
        last.get('test/rank_ic_mean', ''),
    ])
print('method_seed,run,status,rows,step_or_gen,eval_cnt,best_ic_ret,test_ic_mean,test_rank_ic_mean')
for r in rows:
    print(','.join(map(str, r)))
print('__LATEST_PER_METHOD_SEED__')
latest = {}
for r in rows:
    if r[2] != 'ok':
        continue
    key = r[0]
    prev = latest.get(key)
    if prev is None or str(r[1]) > str(prev[1]):
        latest[key] = r
for r in latest.values():
    print(','.join(map(str, r)))
print('__AGG__')
agg = []
for r in latest.values():
    method_seed = r[0]
    if method_seed.startswith('random_search') and '20260628224314_random' not in r[1]:
        continue
    method = method_seed.rsplit('_s', 1)[0]
    seed = int(method_seed.rsplit('_s', 1)[1])
    agg.append({'method': method, 'seed': seed, 'step': float(r[4]), 'eval_cnt': float(r[5]), 'best_ic_ret': float(r[6]), 'test_ic_mean': float(r[7]), 'test_rank_ic_mean': float(r[8])})
df = pd.DataFrame(agg)
print(df.sort_values(['method','seed']).to_string(index=False))
print('__MEAN_STD__')
if len(df):
    print(df.groupby('method')[['test_ic_mean','test_rank_ic_mean','best_ic_ret','eval_cnt']].agg(['mean','std']).to_string())
