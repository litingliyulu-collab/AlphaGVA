import csv, glob, os, statistics
run = '/root/alpha_1203/gva_factor_experiments/runs_newdata/locked_warmpool_gva_20260629_172252'
print('__LAST_METRICS__')
rows=[]
for p in sorted(glob.glob(run + '/*/results/*/metrics.csv')):
    data=list(csv.DictReader(open(p)))
    if not data:
        continue
    r=data[-1]
    name=p.split('/locked_warmpool_gva_20260629_172252/')[1].split('/')[0]
    row={
        'name': name,
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
    rows.append(row)
    print('\t'.join(str(row[k]) for k in ['name','step','eval_cnt','pool_size','Test IC','Test RankIC','Valid IC','Valid RankIC','Mixed IC','Mixed RankIC','elapsed']))
if rows:
    keys=['Test IC','Test RankIC','Valid IC','Valid RankIC','Mixed IC','Mixed RankIC']
    print('__MEAN__')
    print('\t'.join(['mean']+[f'{statistics.mean([r[k] for r in rows]):.6f}' for k in keys]))
print('__PIDS__')
for pid in [2430,2432,2434]:
    alive=os.system(f'kill -0 {pid} >/dev/null 2>&1') == 0
    print(pid, 'alive' if alive else 'done')
