import csv, glob, os, re
from pathlib import Path
ROOT = Path('/root/alpha_1203/gva_factor_experiments/runs/stage2_confirm')
METHODS = [
    'confirm_ppo',
    'confirm_mse',
    'confirm_critic_gva_bw010',
    'confirm_full_gva_bw010_aw050',
]
TARGET = 20000
rows = []
completed = 0
progress_sum = 0
expected = len(METHODS) * 3 * TARGET
for method in METHODS:
    paths = sorted(glob.glob(str(ROOT / method / 'results' / '*' / 'metrics.csv')), key=os.path.getmtime)
    by_seed = {}
    for path in paths:
        run = Path(path).parts[-2]
        parts = run.split('_')
        seed = parts[2] if len(parts) > 2 else '?'
        with open(path, newline='') as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        last = data[-1]
        step = int(float(last.get('timestep') or 0))
        by_seed[seed] = (step, last, path)
    for seed in ['0','1','2']:
        if seed not in by_seed:
            rows.append((method, seed, 0, 0.0, '', '', 'pending'))
            continue
        step, last, path = by_seed[seed]
        pct = min(step, TARGET) / TARGET * 100
        status = 'complete' if step >= TARGET else 'running'
        if status == 'complete':
            completed += 1
        progress_sum += min(step, TARGET)
        rows.append((method, seed, step, pct, last.get('test/ic_mean',''), last.get('test/rank_ic_mean',''), status))
print(f'TOTAL: {progress_sum}/{expected} steps = {progress_sum/expected*100:.1f}% | completed {completed}/{len(METHODS)*3} experiments')
print('method\tseed\tsteps\tpct\tic_mean\trankic_mean\tstatus')
for r in rows:
    print(f'{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.1f}%\t{r[4]}\t{r[5]}\t{r[6]}')
