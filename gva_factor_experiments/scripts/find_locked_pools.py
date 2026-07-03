from pathlib import Path
import re
root = Path('/root/autodl-tmp/gva_factor_experiments/migrated_from_system/runs_newdata/locked_warmpool_gva_continue_20260629_183501')
for result_dir in sorted(root.glob('locked_warm_full_gva25_s*_continue/results/*')):
    pools = list(result_dir.glob('*_steps_pool.json'))
    if not pools:
        continue
    def step(p):
        m = re.search(r'(\d+)_steps_pool', p.name)
        return int(m.group(1)) if m else -1
    latest = max(pools, key=step)
    print(result_dir.parent.parent.name, step(latest), latest)
