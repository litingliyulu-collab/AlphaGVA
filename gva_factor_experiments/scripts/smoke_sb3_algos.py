import os, sys, traceback
sys.path.insert(0, '/root/alpha_1203/AlphaForge-master/alphagen-master')
from stable_baselines3 import A2C, DQN
from sb3_contrib import QRDQN
from alphagen_qlib.stock_data import initialize_qlib, StockData
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.reward import RewardEvaluator
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
import torch

initialize_qlib('/root/autodl-tmp/cn_data_akshare_2010_2026', kernels=1)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
close=Feature(FeatureType.CLOSE)
target=Ref(close,-20)/close-1
data=StockData('csi300','2019-01-02','2019-12-31',device=device)
calc=QLibStockDataCalculator(data,target)
pool=MseAlphaPool(capacity=3, calculator=calc, ic_lower_bound=None, l1_alpha=5e-3, reward_evaluator=RewardEvaluator.from_mode('original'), device=device)
env=AlphaEnv(pool=pool,device=device,print_expr=False)
print('obs', env.observation_space, 'act', env.action_space, 'mask_sum', env.action_masks().sum())
for Algo in [A2C, DQN, QRDQN]:
    try:
        print('TRY', Algo.__name__)
        model=Algo('MlpPolicy', env, device=device, verbose=0, gamma=1.0, policy_kwargs=dict(features_extractor_class=LSTMSharedNet, features_extractor_kwargs=dict(n_layers=1,d_model=64,dropout=0.0,device=device)))
        model.learn(total_timesteps=64)
        print('OK', Algo.__name__, 'pool', pool.size)
    except Exception as e:
        print('FAIL', Algo.__name__, type(e).__name__, str(e))
        traceback.print_exc(limit=2)
