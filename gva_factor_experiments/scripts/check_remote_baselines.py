import importlib, json, pathlib, re
print('__IMPORTS__')
for m in ['lightgbm','xgboost','sklearn','tensorflow','gplearn','qlib','sb3_contrib','stable_baselines3']:
    try:
        mod=importlib.import_module(m)
        print(m, 'OK', getattr(mod, '__version__', ''))
    except Exception as e:
        print(m, 'FAIL', type(e).__name__, str(e)[:200])
print('__NOTEBOOKS__')
for p in pathlib.Path('/root/alpha_1203/AlphaForge-master').glob('exp_*.ipynb'):
    print('###', p.name)
    try:
        nb=json.loads(p.read_text(errors='ignore'))
    except Exception as e:
        print('ERR', e); continue
    txt='\n'.join(''.join(c.get('source',[])) for c in nb.get('cells',[]))
    keys=['LightGBM','LGBM','XGBoost','XGB','MLP','RandomForest','sklearn','xgboost','lightgbm','PPO_filter','GP_filter','AlphaGen','QFR','DSO','AFF']
    print('HAS', [k for k in keys if k.lower() in txt.lower()])
    lines=[]
    for i,line in enumerate(txt.splitlines(),1):
        if any(k.lower() in line.lower() for k in keys+['get_data_by_year','train_model','predict','model =','model=','out_']):
            lines.append((i,line[:220]))
    for i,line in lines[:120]:
        print(f'{i}: {line}')
