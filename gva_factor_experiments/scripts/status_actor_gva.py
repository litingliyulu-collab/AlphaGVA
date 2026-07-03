from pathlib import Path
import pandas as pd
import subprocess

base = Path('/root/autodl-tmp/gva_factor_experiments/runs_newdata')
runs = sorted(base.glob('actor_gva_*'))
run = runs[-1] if runs else None
print('__RUN__', run)
if run is None:
    raise SystemExit(0)
print('__LOGS__')
for log in sorted((run / 'logs').glob('*.log')):
    print('---', log.name)
    try:
        lines = log.read_text(errors='ignore').splitlines()
        print('\n'.join(lines[-8:]))
    except Exception as e:
        print(type(e).__name__, e)
print('__METRICS__')
print('method,status,timestep,pool_size,best_ic_ret,test_ic_1,test_ic_2,test_rank_ic_1,test_rank_ic_2,test_ic_mean,test_rank_ic_mean,baseline_loss,baseline_bank,actor_weight_mean,actor_weight_std')
for method_dir in sorted([p for p in run.iterdir() if p.is_dir() and p.name != 'logs']):
    files = list((method_dir / 'results').glob('*/metrics.csv'))
    if not files:
        print(f'{method_dir.name},no_metrics,,,,,,,,,,,,,')
        continue
    f = files[0]
    try:
        df = pd.read_csv(f)
        if df.empty:
            print(f'{method_dir.name},empty,,,,,,,,,,,,,')
            continue
        last = df.iloc[-1]
        vals = [
            method_dir.name,
            'ok',
            int(last.get('timestep', -1)),
            int(last.get('pool/size', -1)),
            float(last.get('pool/best_ic_ret', float('nan'))),
            float(last.get('test/ic_1', float('nan'))),
            float(last.get('test/ic_2', float('nan'))),
            float(last.get('test/rank_ic_1', float('nan'))),
            float(last.get('test/rank_ic_2', float('nan'))),
            float(last.get('test/ic_mean', float('nan'))),
            float(last.get('test/rank_ic_mean', float('nan'))),
            float(last.get('train/baseline_loss', float('nan'))),
            float(last.get('gva/baseline_bank_size', float('nan'))),
            float(last.get('train/actor_weight_mean', float('nan'))),
            float(last.get('train/actor_weight_std', float('nan'))),
        ]
        print(','.join(map(str, vals)))
    except Exception as e:
        print(f'{method_dir.name},read_error,{type(e).__name__}:{e},,,,,,,,,,,,')
print('__ACTIVE_PIDS__')
try:
    print(subprocess.check_output(['pgrep', '-af', 'scripts/rl_v1.py'], text=True).strip())
except Exception as e:
    print('none_or_error', e)
print('__GPU__')
try:
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv'], text=True).strip())
except Exception as e:
    print('gpu_error', e)
