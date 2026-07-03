from pathlib import Path
import pandas as pd
import os, subprocess, time
run = Path('/root/alpha_1203/gva_factor_experiments/runs_newdata/main_compare_20260628_211640')
print('__MAIN_LOG__')
log = Path('/root/alpha_1203/gva_factor_experiments/logs/main_compare_newdata_parallel_20260628_211640.log')
if log.exists():
    print('\n'.join(log.read_text(errors='ignore').splitlines()[-20:]))
print('__RUN_EXISTS__', run.exists())
print('__METRICS__')
rows = []
for method_dir in sorted([p for p in run.iterdir() if p.is_dir()] if run.exists() else []):
    files = list((method_dir / 'results').glob('*/metrics.csv'))
    if not files:
        rows.append((method_dir.name, 'no_metrics', None, None, None, None, None, None, None, None))
        continue
    f = files[0]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        rows.append((method_dir.name, 'read_error', str(e), None, None, None, None, None, None, None))
        continue
    if len(df) == 0:
        rows.append((method_dir.name, 'empty', 0, None, None, None, None, None, None, None))
        continue
    last = df.iloc[-1]
    rows.append((
        method_dir.name,
        'ok',
        int(last.get('timestep', -1)) if pd.notna(last.get('timestep', None)) else None,
        int(last.get('pool/size', -1)) if pd.notna(last.get('pool/size', None)) else None,
        float(last.get('pool/best_ic_ret', float('nan'))),
        float(last.get('test/ic_mean', float('nan'))),
        float(last.get('test/rank_ic_mean', float('nan'))),
        float(last.get('train/baseline_loss', float('nan'))),
        float(last.get('gva/baseline_bank_size', float('nan'))),
        float(last.get('train/actor_weight_std', float('nan'))),
    ))
print('method,status,timestep,pool_size,best_ic_ret,test_ic_mean,test_rank_ic_mean,baseline_loss,baseline_bank,actor_weight_std')
for r in rows:
    print(','.join('' if x is None else str(x) for x in r))
print('__GPU__')
try:
    out = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv'], text=True)
    print(out.strip())
except Exception as e:
    print('gpu_error', e)
print('__ACTIVE_MAIN_PIDS__')
try:
    out = subprocess.check_output("pgrep -af 'scripts/rl_v1.py' | grep main_compare_20260628_211640 | wc -l", shell=True, text=True)
    print(out.strip())
except Exception as e:
    print('pid_error', e)
