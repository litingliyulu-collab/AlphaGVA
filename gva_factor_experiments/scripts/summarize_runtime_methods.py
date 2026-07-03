import csv, glob, os

def read_last_metrics(pattern):
    out=[]
    for p in sorted(glob.glob(pattern)):
        rows=list(csv.DictReader(open(p)))
        if not rows:
            continue
        out.append((p, rows[-1]))
    return out

print('__FULL_GVA_METRICS__')
for p,r in read_last_metrics('/root/alpha_1203/gva_factor_experiments/runs_newdata/main_compare_20260628_211640/full_gva25_s*/results/*/metrics.csv'):
    name=p.split('/main_compare_20260628_211640/')[1].split('/')[0]
    print(name, r.get('timestep'), r.get('time/time_elapsed'), r.get('pool/eval_cnt'))

print('__WARMPOOL_METRICS__')
for p,r in read_last_metrics('/root/alpha_1203/gva_factor_experiments/runs_newdata/warmpool_gva_20260629_115537/warm_full_gva25_s*/results/*/metrics.csv'):
    name=p.split('/warmpool_gva_20260629_115537/')[1].split('/')[0]
    print(name, r.get('timestep'), r.get('time/time_elapsed'), r.get('pool/eval_cnt'))

print('__LOCKED_BASE_METRICS__')
for p,r in read_last_metrics('/root/alpha_1203/gva_factor_experiments/runs_newdata/locked_warmpool_gva_20260629_172252/locked_warm_full_gva25_s*/results/*/metrics.csv'):
    name=p.split('/locked_warmpool_gva_20260629_172252/')[1].split('/')[0]
    print(name, r.get('timestep'), r.get('time/time_elapsed'), r.get('pool/eval_cnt'))

print('__LOCKED_CONT_METRICS__')
for p,r in read_last_metrics('/root/alpha_1203/gva_factor_experiments/runs_newdata/locked_warmpool_gva_continue_20260629_183501/locked_warm_full_gva25_s*_continue/results/*/metrics.csv'):
    name=p.split('/locked_warmpool_gva_continue_20260629_183501/')[1].split('/')[0]
    print(name, r.get('timestep'), r.get('time/time_elapsed'), r.get('pool/eval_cnt'))

print('__RANDOM_METRICS__')
for p,r in read_last_metrics('/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_20260628_223714/random_search_s*/results/*/metrics.csv'):
    name=p.split('/horizontal_baselines_20260628_223714/')[1].split('/')[0]
    keys=list(r.keys())
    print(name, 'attempts=', r.get('attempts'), 'elapsed_sec=', r.get('elapsed_sec'), 'eval=', r.get('pool/eval_cnt'), 'keys=', keys)

print('__RANDOM_LOG_TAILS__')
for f in sorted(glob.glob('/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_20260628_223714/logs/random_s*_batch.log')):
    print('FILE', f)
    print('\n'.join(open(f).read().splitlines()[-10:]))

print('__RUN_LOGS__')
for f in [
'/root/alpha_1203/gva_factor_experiments/logs/main_compare_newdata_parallel_20260628_211640.log',
'/root/alpha_1203/gva_factor_experiments/logs/warmpool_gva_launch_20260629.log',
'/root/alpha_1203/gva_factor_experiments/logs/locked_warmpool_gva_launch_20260629.log',
'/root/alpha_1203/gva_factor_experiments/runs_newdata/locked_warmpool_gva_continue_20260629_183501/run.log',
]:
    print('FILE', f)
    if os.path.exists(f):
        print('\n'.join(open(f).read().splitlines()[-25:]))
