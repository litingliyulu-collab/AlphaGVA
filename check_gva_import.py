import ast
import pathlib
for path in ['alphagen/rl/custom_ppo_trainer.py', 'scripts/rl_v1.py']:
    ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))
    print(path, 'syntax ok')
from alphagen.rl.custom_ppo_trainer import train_custom_ppo
print('import train_custom_ppo ok')
