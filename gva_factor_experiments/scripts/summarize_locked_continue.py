import csv, glob, os, statistics
runs=sorted(glob.glob('/root/alpha_1203/gva_factor_experiments/runs_newdata/locked_warmpool_gva_continue_*'))
run=runs[-1]
print('RUN', run)
print('__RUNLOG__')
log=os.path.join(run,'run.log')
if os.path.exists(log):
    print('\n'.join(open(log).read().splitlines()[-30:]))
print('__METRICS__')
rows=[]
for p in sorted(glob.glob(run+'/*/results/*/metrics.csv')):
    data=list(csv.DictReader(open(p)))
    if not data:
        continue
    r=data[-1]
    group=p.split(run+'/')[1].split('/')[0]
    seed=group.split('_s',1)[1].split('_',1)[0]
    base={'0':28096,'1':26496,'2':27264}[seed]
    row={
        'method':group,
        'seed':seed,
        'cont_step':int(float(r.get('timestep') or 0)),
        'total_step':base+int(float(r.get('timestep') or 0)),
        'Test IC':float(r.get('test/ic_2') or 'nan'),
        'Test RankIC':float(r.get('test/rank_ic_2') or 'nan'),
        'Valid IC':float(r.get('test/ic_1') or 'nan'),
        'Valid RankIC':float(r.get('test/rank_ic_1') or 'nan'),
        'Mixed IC':float(r.get('test/ic_mean') or 'nan'),
        'Mixed RankIC':float(r.get('test/rank_ic_mean') or 'nan'),
    }
    rows.append(row)
    print('\t'.join(str(row[k]) for k in ['method','seed','cont_step','total_step','Test IC','Test RankIC','Valid IC','Valid RankIC','Mixed IC','Mixed RankIC']))
if rows:
    keys=['Test IC','Test RankIC','Valid IC','Valid RankIC','Mixed IC','Mixed RankIC']
    print('__MEAN__')
    print('\t'.join(['mean']+[f'{statistics.mean([x[k] for x in rows]):.6f}' for k in keys]))
print('__PIDS__')
for pid in [10699,10714,10724,10734]:
    print(pid, 'alive' if os.system(f'kill -0 {pid} >/dev/null 2>&1')==0 else 'done')
