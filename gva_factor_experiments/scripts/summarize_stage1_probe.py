from pathlib import Path
import pandas as pd
run = Path('/root/alpha_1203/gva_factor_experiments/runs_newdata/stage1_probe_20260628_201941')
for name in ['ppo_s0', 'critic_gva25_s0']:
    files = list((run / name / 'results').glob('*/metrics.csv'))
    print('__', name, '__')
    if not files:
        print('no metrics')
        continue
    df = pd.read_csv(files[0])
    print('rows', len(df), 'file', files[0])
    if len(df):
        cols = [c for c in ['timestep','pool/size','pool/best_ic_ret','test/ic_mean','test/rank_ic_mean','train/baseline_loss','gva/baseline_bank_size','gva/baseline_hit_rate','gva/greedy_success','gva/greedy_updates'] if c in df.columns]
        print(df[cols].tail(5).to_string(index=False))
