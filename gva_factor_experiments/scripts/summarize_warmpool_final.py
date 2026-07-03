import csv, glob, os, statistics
run = '/root/alpha_1203/gva_factor_experiments/runs_newdata/warmpool_gva_20260629_115537'
print('__PID_STATUS__')
for pid in [3765,3767,3769]:
    alive = os.system(f'kill -0 {pid} >/dev/null 2>&1') == 0
    print(pid, 'alive' if alive else 'done')
print('__RUN_LOG__')
for p in ['/root/alpha_1203/gva_factor_experiments/logs/warmpool_gva_launch_20260629.log', run + '/run.log']:
    if os.path.exists(p):
        print('FILE', p)
        lines=open(p).read().splitlines()
        print('\n'.join(lines[-20:]))
print('__FINAL_METRICS__')
rows_out=[]
for p in sorted(glob.glob(run + '/*/results/*/metrics.csv')):
    rows=list(csv.DictReader(open(p)))
    if not rows:
        continue
    r=rows[-1]
    method=p.split('/warmpool_gva_20260629_115537/')[1].split('/')[0]
    seed=method.rsplit('_s',1)[-1]
    out={
        'method': method,
        'seed': seed,
        'step': int(float(r.get('timestep') or 0)),
        'eval_cnt': int(float(r.get('pool/eval_cnt') or 0)),
        'pool_size': int(float(r.get('pool/size') or 0)),
        'Test IC': float(r.get('test/ic_2') or 'nan'),
        'Test RankIC': float(r.get('test/rank_ic_2') or 'nan'),
        'Valid IC': float(r.get('test/ic_1') or 'nan'),
        'Valid RankIC': float(r.get('test/rank_ic_1') or 'nan'),
        'Mixed IC': float(r.get('test/ic_mean') or 'nan'),
        'Mixed RankIC': float(r.get('test/rank_ic_mean') or 'nan'),
        'elapsed': float(r.get('time/time_elapsed') or 'nan'),
    }
    rows_out.append(out)
    print('\t'.join(str(out[k]) for k in ['method','seed','step','eval_cnt','pool_size','Test IC','Test RankIC','Valid IC','Valid RankIC','Mixed IC','Mixed RankIC','elapsed']))
print('__MEAN__')
if rows_out:
    keys=['Test IC','Test RankIC','Valid IC','Valid RankIC','Mixed IC','Mixed RankIC']
    print('\t'.join(['mean']+[f'{statistics.mean([r[k] for r in rows_out]):.6f}' for k in keys]))
