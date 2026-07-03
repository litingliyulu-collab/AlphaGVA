import csv, glob, os, subprocess
run = '/root/alpha_1203/gva_factor_experiments/runs_newdata/warmpool_gva_20260629_115537'
print('__LAUNCH_LOG__')
log = '/root/alpha_1203/gva_factor_experiments/logs/warmpool_gva_launch_20260629.log'
if os.path.exists(log):
    lines = open(log).read().splitlines()
    print('\n'.join(lines[-20:]))
print('__FILES__')
print('\n'.join(sorted(glob.glob(run + '/*'))[:20]))
print('__LAST_METRICS__')
for p in sorted(glob.glob(run + '/*/results/*/metrics.csv')):
    rows = list(csv.DictReader(open(p)))
    if not rows:
        continue
    r = rows[-1]
    name = p.split('/warmpool_gva_20260629_115537/')[1].split('/')[0]
    vals = [name, r.get('timestep'), r.get('pool/size'), r.get('pool/eval_cnt'), r.get('test/ic_1'), r.get('test/rank_ic_1'), r.get('test/ic_2'), r.get('test/rank_ic_2'), r.get('test/ic_mean'), r.get('test/rank_ic_mean'), r.get('time/time_elapsed')]
    print('\t'.join(str(x) for x in vals))
