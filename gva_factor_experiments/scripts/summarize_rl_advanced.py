import csv
import re
from pathlib import Path
root = Path('/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629')
for method in ['a2c', 'dqn', 'qrdqn']:
    print(f'__{method.upper()}__')
    runs = []
    for f in sorted(root.glob(f'{method}_baseline_s*/results/*/metrics.csv')):
        try:
            with open(f, newline='') as fp:
                data = list(csv.DictReader(fp))
            if not data:
                continue
            last = data[-1]
            seed_match = re.search(rf'{method}_baseline_s(\d+)', str(f))
            seed = int(seed_match.group(1)) if seed_match else -1
            ts = int(float(last.get('timestep') or 0))
            run = f.parent.name
            if ('fresh' in run) or ('resume' in run) or ts >= 30000:
                runs.append((method, seed, ts, run, f, last))
        except Exception as e:
            print('ERR', f, type(e).__name__, e)
    for method, seed, ts, run, f, last in sorted(runs, key=lambda x: (x[1], x[2], x[3])):
        print('\t'.join([
            method,
            str(seed),
            str(ts),
            run,
            last.get('test/ic_2', ''),
            last.get('test/rank_ic_2', ''),
            last.get('test/ic_mean', ''),
            last.get('test/rank_ic_mean', ''),
            last.get('pool/best_ic_ret', ''),
            last.get('pool/size', ''),
            str(f),
        ]))
