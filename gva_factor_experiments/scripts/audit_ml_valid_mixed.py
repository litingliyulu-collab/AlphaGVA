from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

PROJECT=Path('/root/alpha_1203/AlphaForge-master/alphagen-master')
sys.path.insert(0, str(PROJECT))
from scripts import ml_baselines_newdata as ml
from alphagen_qlib.stock_data import initialize_qlib, StockData

root=Path('/root/alpha_1203/gva_factor_experiments/runs_newdata/ml_baselines_20260629_101312')
initialize_qlib('/root/autodl-tmp/cn_data_akshare_2010_2026', kernels=1)
data_device=torch.device('cpu')
train_data=StockData('csi300','2010-01-04','2021-12-31',device=data_device)
valid_data=StockData('csi300','2022-01-04','2023-12-29',device=data_device)
test_data=StockData('csi300','2024-01-02','2026-05-28',device=data_device)
valid_days=valid_data.n_days; test_days=test_data.n_days
wv=valid_days/(valid_days+test_days); wt=test_days/(valid_days+test_days)
print('__DAYS__', valid_days, test_days, wv, wt)

x_train_raw, y_train_raw=ml.build_arrays(train_data, 60)
x_valid_raw, y_valid_raw=ml.build_arrays(valid_data, 60)
x_test_raw, y_test_raw=ml.build_arrays(test_data, 60)
x_train, x_valid, x_test=ml.normalize_features(x_train_raw, x_valid_raw, x_test_raw)
x_valid_flat=x_valid.reshape(-1, x_valid.shape[-1])
x_test_flat=x_test.reshape(-1, x_test.shape[-1])

rows=[]
for method in ['lightgbm','mlp']:
  for seed in [0,1,2]:
    dirs=sorted((root/f'{method}_s{seed}'/'results').glob('*_ml'))
    if not dirs:
      continue
    d=dirs[-1]
    if method=='lightgbm':
      model=lgb.Booster(model_file=str(d/'model.txt'))
      pred_valid=model.predict(x_valid_flat, num_iteration=model.best_iteration).astype(np.float32)
      pred_test=model.predict(x_test_flat, num_iteration=model.best_iteration).astype(np.float32)
    else:
      device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      model=ml.MLP(x_valid_flat.shape[1]).to(device)
      model.load_state_dict(torch.load(d/'model.pt', map_location=device))
      model.eval()
      def predict(x):
        outs=[]
        bs=8192
        with torch.no_grad():
          for st in range(0,len(x),bs):
            xb=torch.from_numpy(x[st:st+bs]).to(device)
            outs.append(model(xb).detach().cpu().numpy())
        return np.concatenate(outs).astype(np.float32)
      pred_valid=predict(x_valid_flat)
      pred_test=predict(x_test_flat)
    valid=ml.calc_metrics(pred_valid, y_valid_raw)
    test=ml.calc_metrics(pred_test, y_test_raw)
    vi=valid['test/ic_mean']; vr=valid['test/rank_ic_mean']
    ti=test['test/ic_mean']; tr=test['test/rank_ic_mean']
    rows.append({'method':method,'seed':seed,'test_ic':ti,'test_rank_ic':tr,'valid_ic':vi,'valid_rank_ic':vr,'mixed_ic':wv*vi+wt*ti,'mixed_rank_ic':wv*vr+wt*tr})

df=pd.DataFrame(rows)
print('__ROWS__')
print(df.to_string(index=False))
print('__AGG__')
print(df.groupby('method')[['test_ic','test_rank_ic','valid_ic','valid_rank_ic','mixed_ic','mixed_rank_ic']].agg(['mean','std']).to_string())
# save audit csv next to run
out=root/'ml_valid_test_mixed_audit.csv'
df.to_csv(out,index=False)
print('__SAVED__', out)
