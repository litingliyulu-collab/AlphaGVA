import numpy as np
import torch
import torch.nn as nn
from alphagen.rl.custom_ppo_trainer import _run_greedy_baseline_update

class DummyPool:
    def __init__(self):
        self.size = 1
        self.exprs = ['old', None]
        self.single_ics = np.zeros(2)
        self._weights = np.zeros(2)
        self._mutual_ics = np.eye(2)
        self._extra_info = [None, None]
        self.best_obj = 0.2
        self.best_ic_ret = 0.2
        self._failure_cache = set()
        self.update_history = []
        self.eval_cnt = 0

class DummyCore:
    def __init__(self):
        self.pool = DummyPool()
        self._tokens = []
        self._builder = []
        self.eval_cnt = 0

class DummyEnv:
    def __init__(self):
        self.env = DummyCore()
        self.state = np.zeros(5, dtype=np.uint8)
        self.counter = 0
        self.size_action = 3
    def reset(self):
        self.state[:] = 0
        self.counter = 0
        return self.state, {}
    def action_masks(self):
        mask = np.zeros(self.size_action, dtype=bool)
        if self.counter < 2:
            mask[1] = True
        else:
            mask[2] = True
        return mask
    def step(self, action):
        if action == 2 or self.counter >= 4:
            self.env.pool.size = 99
            self.env.pool.best_obj = 9.9
            return self.state, 1.0, True, False, {}
        self.state[self.counter] = action
        self.counter += 1
        return self.state, 0.0, False, False, {}

class DummyActor(nn.Module):
    def forward(self, obs):
        probs = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32)
        return probs.repeat(obs.shape[0], 1)

env = DummyEnv()
bank = {}
ok, path_len, reward = _run_greedy_baseline_update(env, DummyActor(), torch.device('cpu'), tuple(), bank, 1.0, 5)
print('ok', ok, 'path_len', path_len, 'reward', reward)
print('bank', bank)
print('env_counter', env.counter, 'pool_size', env.env.pool.size, 'best_obj', env.env.pool.best_obj)
assert ok
assert bank[tuple()] == 1.0
assert bank[(1,)] == 1.0
assert bank[(1, 1)] == 1.0
assert env.counter == 0
assert env.env.pool.size == 1
assert env.env.pool.best_obj == 0.2
print('dummy gva smoke ok')
