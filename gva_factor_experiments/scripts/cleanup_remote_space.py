from pathlib import Path
import re
import shutil

root = Path('/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629')
removed = 0
kept = []

if root.exists():
    for result_dir in root.glob('*/results/*'):
        if not result_dir.is_dir():
            continue
        zips = list(result_dir.glob('*_steps.zip'))
        if not zips:
            continue
        def step(p):
            m = re.search(r'(\d+)_steps\.zip$', p.name)
            return int(m.group(1)) if m else -1
        latest = max(zips, key=step)
        kept.append(str(latest))
        for p in zips:
            if p == latest:
                continue
            try:
                removed += p.stat().st_size
                p.unlink()
            except FileNotFoundError:
                pass

for p in [Path('/root/autodl-tmp/gva_factor_experiments/smoke/rl_advanced_baselines'), Path('/root/autodl-tmp/.Trash-0')]:
    if not p.exists():
        continue
    try:
        if p.is_dir():
            removed += sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
            shutil.rmtree(p)
        else:
            removed += p.stat().st_size
            p.unlink()
    except FileNotFoundError:
        pass

print('REMOVED_BYTES', removed)
print('REMOVED_GB', round(removed / (1024 ** 3), 3))
print('KEPT_LATEST_CHECKPOINTS')
for x in kept:
    print(x)
