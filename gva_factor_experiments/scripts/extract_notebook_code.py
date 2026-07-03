import json
from pathlib import Path
root = Path('/root/alpha_1203/AlphaForge-master')
for name, ranges in {
    'exp_ML_train_and_result.ipynb': [(1,70),(70,125),(139,190),(211,285),(312,386)],
    'exp_AFF_calc_result.ipynb': [(1,120)],
    'exp_DSO_calc_result.ipynb': [(1,120)],
    'exp_RL_calc_result.ipynb': [(1,120)],
}.items():
    nb = json.loads((root/name).read_text(errors='ignore'))
    txt = '\n'.join(''.join(c.get('source', [])) for c in nb.get('cells', []))
    lines = txt.splitlines()
    print('###', name)
    for a,b in ranges:
        print(f'__LINES_{a}_{b}__')
        for i in range(a-1, min(b, len(lines))):
            line = lines[i]
            if name == 'exp_ML_train_and_result.ipynb' or any(k in line for k in ['paths =','out_','get_data_by_year','test_ensemble','batch_spearmanr','result','print(']):
                print(f'{i+1}: {line[:240]}')
