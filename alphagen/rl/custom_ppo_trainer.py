import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import math
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.rl.policy import LSTMSharedNet
from alphagen.rl.env.wrapper import AlphaEnv


class ActorNet(nn.Module):
    """
    Actor网络（策略网络）
    基于LSTM特征提取器 + 策略头
    """
    def __init__(
        self,
        observation_space,
        action_space_size: int,
        features_extractor: LSTMSharedNet,
        device: torch.device
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.device = device
        
        # 策略头：从特征到动作概率
        feature_dim = features_extractor._d_model
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 提取特征
        features = self.features_extractor(obs)
        # 输出动作概率
        return self.policy_head(features)


class CriticNet(nn.Module):
    """
    Critic网络（价值函数网络）
    基于LSTM特征提取器 + 价值头
    """
    def __init__(
        self,
        observation_space,
        features_extractor: LSTMSharedNet,
        device: torch.device
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.device = device
        
        # 价值头：从特征到价值估计
        feature_dim = features_extractor._d_model
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 提取特征
        features = self.features_extractor(obs)
        # 输出价值估计
        return self.value_head(features)


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_values: List[float],
    dones: List[bool],
    gamma: float = 1.0,
    gae_lambda: float = 0.95
) -> Tuple[List[float], List[float]]:
    """
    计算GAE (Generalized Advantage Estimation) 和returns
    """
    advantages = []
    returns = []
    
    gae = 0
    for step in reversed(range(len(rewards))):
        if dones[step]:
            delta = rewards[step] - values[step]
            gae = delta
        else:
            delta = rewards[step] + gamma * next_values[step] - values[step]
            gae = delta + gamma * gae_lambda * gae
        
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])
    
    return advantages, returns



def _unwrap_core(env: AlphaEnv) -> Any:
    core = env
    while hasattr(core, "env"):
        core = core.env
    return core


def _state_key(env: AlphaEnv) -> Tuple[int, ...]:
    counter = int(getattr(env, "counter", 0))
    return tuple(int(x) for x in np.asarray(env.state[:counter]).tolist())


def _snapshot_env(env: AlphaEnv) -> Dict[str, Any]:
    core = _unwrap_core(env)
    return {
        "state": np.array(env.state, copy=True),
        "counter": int(getattr(env, "counter", 0)),
        "tokens": copy.deepcopy(getattr(core, "_tokens", None)),
        "builder": copy.deepcopy(getattr(core, "_builder", None)),
        "eval_cnt": getattr(core, "eval_cnt", None),
    }


def _restore_env(env: AlphaEnv, snapshot: Dict[str, Any]) -> None:
    core = _unwrap_core(env)
    env.state = np.array(snapshot["state"], copy=True)
    env.counter = int(snapshot["counter"])
    if snapshot["tokens"] is not None:
        core._tokens = copy.deepcopy(snapshot["tokens"])
    if snapshot["builder"] is not None:
        core._builder = copy.deepcopy(snapshot["builder"])
    if snapshot["eval_cnt"] is not None:
        core.eval_cnt = snapshot["eval_cnt"]


def _snapshot_pool(pool: Any) -> Dict[str, Any]:
    return {
        "size": pool.size,
        "exprs": copy.deepcopy(pool.exprs),
        "single_ics": np.array(pool.single_ics, copy=True),
        "weights": np.array(pool._weights, copy=True),
        "mutual_ics": np.array(pool._mutual_ics, copy=True),
        "extra_info": copy.deepcopy(pool._extra_info),
        "best_obj": getattr(pool, "best_obj", None),
        "best_ic_ret": pool.best_ic_ret,
        "failure_cache": copy.deepcopy(getattr(pool, "_failure_cache", set())),
        "update_history": copy.deepcopy(getattr(pool, "update_history", [])),
        "eval_cnt": pool.eval_cnt,
    }


def _restore_pool(pool: Any, snapshot: Dict[str, Any]) -> None:
    pool.size = snapshot["size"]
    pool.exprs = copy.deepcopy(snapshot["exprs"])
    pool.single_ics = np.array(snapshot["single_ics"], copy=True)
    pool._weights = np.array(snapshot["weights"], copy=True)
    pool._mutual_ics = np.array(snapshot["mutual_ics"], copy=True)
    pool._extra_info = copy.deepcopy(snapshot["extra_info"])
    if snapshot["best_obj"] is not None:
        pool.best_obj = snapshot["best_obj"]
    pool.best_ic_ret = snapshot["best_ic_ret"]
    if hasattr(pool, "_failure_cache"):
        pool._failure_cache = copy.deepcopy(snapshot["failure_cache"])
    if hasattr(pool, "update_history"):
        pool.update_history = copy.deepcopy(snapshot["update_history"])
    pool.eval_cnt = snapshot["eval_cnt"]


def _masked_actor_probs(
    actor: nn.Module,
    state: np.ndarray,
    mask: Optional[np.ndarray],
    device: torch.device
) -> torch.Tensor:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    action_probs = actor(state_tensor)
    if mask is not None:
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
        action_probs = action_probs.masked_fill(~mask_tensor, 0.0)
        denom = action_probs.sum(dim=1, keepdim=True)
        action_probs = torch.where(denom > 0, action_probs / denom.clamp_min(1e-12), action_probs)
    return action_probs


def _run_greedy_baseline_update(
    env: AlphaEnv,
    actor: nn.Module,
    device: torch.device,
    state_key: Tuple[int, ...],
    baseline_bank: Dict[Tuple[int, ...], float],
    gamma: float,
    max_depth: int,
) -> Tuple[bool, int, float]:
    env_snapshot = _snapshot_env(env)
    pool = _unwrap_core(env).pool
    pool_snapshot = _snapshot_pool(pool)

    path: List[Tuple[int, ...]] = []
    terminal_reward = 0.0
    terminated = False

    try:
        env.reset()
        for action in state_key:
            _, _, done, truncated, _ = env.step(int(action))
            if done or truncated:
                return False, 0, 0.0

        path.append(_state_key(env))
        depth = 0
        while depth < max_depth:
            mask = env.action_masks() if hasattr(env, "action_masks") else None
            with torch.no_grad():
                action_probs = _masked_actor_probs(actor, env.state, mask, device)
                if float(action_probs.sum().item()) <= 0:
                    break
                action = int(torch.argmax(action_probs, dim=1).item())

            _, reward, done, truncated, _ = env.step(action)
            depth += 1
            if done or truncated:
                terminal_reward = float(reward)
                terminated = True
                break
            path.append(_state_key(env))
    finally:
        _restore_pool(pool, pool_snapshot)
        _restore_env(env, env_snapshot)

    if not terminated or not path:
        return False, len(path), terminal_reward

    last_idx = len(path) - 1
    for idx, key in enumerate(path):
        target = terminal_reward * (gamma ** (last_idx - idx))
        old = baseline_bank.get(key)
        if old is None or target > old:
            baseline_bank[key] = target
    return True, len(path), terminal_reward


def train_custom_ppo(
    env: AlphaEnv,
    total_timesteps: int,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
    batch_size: int = 128,
    n_epochs: int = 4,
    clip_range: float = 0.2,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    device: torch.device = None,
    callback: Optional[BaseCallback] = None,
    tensorboard_log: Optional[str] = None,
    # 自定义critic损失参数
    critic_loss_type: str = "hybrid",  # "mse", "hybrid", "huber"
    td_weight: float = 0.9,  # TD损失权重（用于hybrid）
    baseline_weight: float = 0.1,  # Baseline损失权重（用于hybrid）
    use_history_best: bool = True,  # 是否使用历史最佳作为baseline
    rollout_length: int = 256,  # ← 新增：控制rollout长度
    log_frequency: int = 1,  # ← 新增：每N个rollout记录一次日志
    actor_gap_weight: float = 0.0,
    actor_gap_clip: float = 2.0,
    gva_budget_ratio: float = 0.25,
    gva_max_updates_per_rollout: int = 32,
    gva_refresh_interval: int = 10,
    gva_max_rollout_depth: Optional[int] = None,
    gva_min_state_len: int = 1,
):
    """
    自定义PPO训练函数，完全控制critic损失函数
    
    参数:
        critic_loss_type: "mse" - 标准MSE, "hybrid" - 混合损失, "huber" - Huber损失
        td_weight: TD损失权重（用于hybrid）
        baseline_weight: Baseline损失权重（用于hybrid）
        use_history_best: 是否使用历史最佳奖励作为baseline
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建网络
    observation_space = env.observation_space
    action_space_size = env.action_space.n
    
    # 创建特征提取器
    features_extractor = LSTMSharedNet(
        observation_space=observation_space,
        n_layers=2,
        d_model=128,
        dropout=0.1,
        device=device
    )
    
    # 创建Actor和Critic网络
    actor = ActorNet(observation_space, action_space_size, features_extractor, device).to(device)
    critic = CriticNet(observation_space, features_extractor, device).to(device)
    
    # 创建优化器
    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    
    # 损失函数
    loss_fn = nn.MSELoss()
    
    # Training history and GVA baseline bank.
    greedy_baseline_bank: Dict[Tuple[int, ...], float] = {}
    greedy_baseline_last_update: Dict[Tuple[int, ...], int] = {}
    timestep = 0
    rollout_count = 0  # ← 新增：记录rollout数量
    
    # 训练循环
    while timestep < total_timesteps:
        # 收集一个rollout的数据
        states, state_keys, actions, rewards, values, next_values, dones, action_masks = [], [], [], [], [], [], [], []
        
        state, info = env.reset()
        done = False
        episode_reward = 0
        rollout_steps = 0
        
        # ========== 修改：累积收集rollout_length个timesteps（可能跨越多个episode）==========
        while rollout_steps < rollout_length and timestep < total_timesteps:
            current_state_key = _state_key(env)
            current_state = np.array(state, copy=True)
            state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 获取动作mask
            if hasattr(env, 'action_masks'):
                mask = env.action_masks()
                mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
            else:
                mask_tensor = None
            
            # Actor选择动作
            with torch.no_grad():
                action_probs = actor(state_tensor)
                if mask_tensor is not None:
                    # 应用mask
                    action_probs = action_probs.masked_fill(~mask_tensor, 0.0)
                    action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
                
                action = torch.multinomial(action_probs, 1).item()
            
            # Critic估计价值
            with torch.no_grad():
                value = critic(state_tensor)
            
            # 执行动作
            next_state, reward, done, truncated, info = env.step(action)
            
            # 存储数据
            states.append(current_state)
            state_keys.append(current_state_key)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            dones.append(done or truncated)
            if mask_tensor is not None:
                action_masks.append(mask)
            
            # 获取下一个状态的价值
            if not (done or truncated):
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    next_value = critic(next_state_tensor)
                next_values.append(next_value.item())
            else:
                next_values.append(0.0)
            
            state = next_state
            episode_reward += reward
            timestep += 1
            rollout_steps += 1
            
            # ========== 如果episode结束，重置环境继续收集 ==========
            if done or truncated:
                state, info = env.reset()
                done = False
                episode_reward = 0
        
        if len(states) == 0:
            continue
        rollout_count += 1

        gva_stats = {
            'baseline_bank_size': len(greedy_baseline_bank),
            'baseline_hit_rate': 0.0,
            'baseline_target_mean': 0.0,
            'baseline_target_std': 0.0,
            'baseline_gap_mean': 0.0,
            'baseline_gap_std': 0.0,
            'greedy_updates': 0,
            'greedy_success': 0,
            'greedy_path_len_mean': 0.0,
            'greedy_terminal_reward_mean': 0.0,
        }

        gva_needs_baseline_bank = (baseline_weight > 0 and critic_loss_type == "hybrid") or (actor_gap_weight != 0)
        if use_history_best and gva_needs_baseline_bank and len(state_keys) > 0:
            max_depth = gva_max_rollout_depth or len(env.state)
            budget_n = int(math.ceil(max(0.0, min(1.0, gva_budget_ratio)) * len(state_keys)))
            if gva_max_updates_per_rollout > 0:
                budget_n = min(budget_n, gva_max_updates_per_rollout)
            budget_n = max(0, budget_n)
            if budget_n > 0:
                eligible_indices = [i for i, key in enumerate(state_keys) if len(key) >= gva_min_state_len]
                if len(eligible_indices) == 0:
                    candidate_indices = np.array([], dtype=int)
                elif budget_n >= len(eligible_indices):
                    candidate_indices = np.array(eligible_indices, dtype=int)
                else:
                    pick_pos = np.linspace(0, len(eligible_indices) - 1, num=budget_n, dtype=int)
                    candidate_indices = np.array([eligible_indices[pos] for pos in pick_pos], dtype=int)
                path_lens, terminal_rewards = [], []
                seen_this_rollout = set()
                for idx in candidate_indices.tolist():
                    key = state_keys[idx]
                    if key in seen_this_rollout:
                        continue
                    seen_this_rollout.add(key)
                    last_update = greedy_baseline_last_update.get(key, -10**9)
                    if key in greedy_baseline_bank and rollout_count - last_update < gva_refresh_interval:
                        continue
                    success, path_len, terminal_reward = _run_greedy_baseline_update(
                        env=env,
                        actor=actor,
                        device=device,
                        state_key=key,
                        baseline_bank=greedy_baseline_bank,
                        gamma=gamma,
                        max_depth=max_depth,
                    )
                    gva_stats['greedy_updates'] += 1
                    greedy_baseline_last_update[key] = rollout_count
                    if success:
                        gva_stats['greedy_success'] += 1
                        path_lens.append(path_len)
                        terminal_rewards.append(terminal_reward)
                if path_lens:
                    gva_stats['greedy_path_len_mean'] = float(np.mean(path_lens))
                if terminal_rewards:
                    gva_stats['greedy_terminal_reward_mean'] = float(np.mean(terminal_rewards))
            gva_stats['baseline_bank_size'] = len(greedy_baseline_bank)
        
        # 计算GAE和returns
        advantages, returns = compute_gae(
            rewards, values, next_values, dones, gamma, gae_lambda
        )
        
        # 转换为tensor
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(device)
        baseline_targets = [greedy_baseline_bank.get(key, ret) for key, ret in zip(state_keys, returns)]
        baseline_hits = [1.0 if key in greedy_baseline_bank else 0.0 for key in state_keys]
        baseline_targets_tensor = torch.tensor(baseline_targets, dtype=torch.float32).unsqueeze(1).to(device)
        baseline_hits_tensor = torch.tensor(baseline_hits, dtype=torch.float32).unsqueeze(1).to(device)
        if len(baseline_targets) > 0:
            gva_stats['baseline_hit_rate'] = float(np.mean(baseline_hits))
            gva_stats['baseline_target_mean'] = float(np.mean(baseline_targets))
            gva_stats['baseline_target_std'] = float(np.std(baseline_targets))
        
        # 获取旧的动作概率
        with torch.no_grad():
            old_action_probs = actor(states_tensor)
            if action_masks:
                masks_tensor = torch.tensor(np.array(action_masks), dtype=torch.bool).to(device)
                old_action_probs = old_action_probs.masked_fill(~masks_tensor, 0.0)
                old_action_probs = old_action_probs / old_action_probs.sum(dim=1, keepdim=True)
            old_action_probs = old_action_probs[range(len(actions)), actions].unsqueeze(1)
        train_stats = {
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'entropy_loss': 0.0,
            'explained_variance': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'td_loss': 0.0,
            'baseline_loss': 0.0,
            'value_mean': 0.0,
            'value_std': 0.0,
            'advantage_mean': float(advantages_tensor.mean().item()),
            'advantage_std': float(advantages_tensor.std(unbiased=False).item()),
            'actor_weight_mean': 1.0,
            'actor_weight_std': 0.0,
            'gva/baseline_bank_size': gva_stats['baseline_bank_size'],
            'gva/baseline_hit_rate': gva_stats['baseline_hit_rate'],
            'gva/baseline_target_mean': gva_stats['baseline_target_mean'],
            'gva/baseline_target_std': gva_stats['baseline_target_std'],
            'gva/baseline_gap_mean': gva_stats['baseline_gap_mean'],
            'gva/baseline_gap_std': gva_stats['baseline_gap_std'],
            'gva/greedy_updates': gva_stats['greedy_updates'],
            'gva/greedy_success': gva_stats['greedy_success'],
            'gva/greedy_path_len_mean': gva_stats['greedy_path_len_mean'],
            'gva/greedy_terminal_reward_mean': gva_stats['greedy_terminal_reward_mean'],
            'total_loss': 0.0,
            'n_updates': 0
        }
        # PPO迭代更新
        for epoch in range(n_epochs):
            # 打乱数据
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Mini-batch更新
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                if len(batch_idx) == 0:
                    continue
                
                b_states = states_tensor[batch_idx]
                b_actions = actions_tensor[batch_idx]
                b_advantages = advantages_tensor[batch_idx]
                b_returns = returns_tensor[batch_idx]
                b_baseline_targets = baseline_targets_tensor[batch_idx]
                b_baseline_hits = baseline_hits_tensor[batch_idx]
                b_old_probs = old_action_probs[batch_idx]
                
                # ========== Critic update with shared greedy baseline ==========
                b_values = critic(b_states)
                td_loss = loss_fn(b_values, b_returns)
                if use_history_best:
                    baseline_tensor = b_baseline_targets.detach()
                else:
                    baseline_tensor = b_values.detach()
                baseline_loss = loss_fn(b_values, baseline_tensor)
                
                if critic_loss_type == "mse":
                    critic_loss = td_loss
                    
                elif critic_loss_type == "hybrid":
                    critic_loss = td_weight * td_loss + baseline_weight * baseline_loss
                    
                elif critic_loss_type == "huber":
                    error = b_returns - b_values
                    delta = 1.0
                    is_small = torch.abs(error) < delta
                    critic_loss = torch.where(
                        is_small,
                        0.5 * error ** 2,
                        delta * (torch.abs(error) - 0.5 * delta)
                    ).mean()
                    
                else:
                    critic_loss = td_loss
                
                # Critic反向传播
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optim.step()
                
                # ========== Actor更新（标准PPO）==========
                new_action_probs = actor(b_states)
                if action_masks:
                    b_masks = masks_tensor[batch_idx]
                    new_action_probs = new_action_probs.masked_fill(~b_masks, 0.0)
                    new_action_probs = new_action_probs / new_action_probs.sum(dim=1, keepdim=True)
                
                new_action_probs = new_action_probs[range(len(b_actions)), b_actions].unsqueeze(1)
                
                # PPO clip
                ratio = new_action_probs / (b_old_probs + 1e-8)
                weighted_advantages = b_advantages
                actor_weights = torch.ones_like(b_advantages)
                if actor_gap_weight != 0:
                    with torch.no_grad():
                        value_gap = baseline_tensor.detach() - b_values.detach()
                        value_gap = torch.clamp(value_gap, min=-actor_gap_clip, max=actor_gap_clip)
                        agreement = torch.sign(b_advantages.detach() * value_gap)
                        actor_delta = actor_gap_weight * agreement * torch.abs(value_gap)
                        actor_weights = torch.clamp(1.0 + actor_delta, min=0.1, max=3.0)
                    weighted_advantages = b_advantages * actor_weights
                surr1 = ratio * weighted_advantages
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * weighted_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 添加熵正则化
                entropy = -(new_action_probs * torch.log(new_action_probs + 1e-8)).sum(dim=1).mean()
                actor_loss = actor_loss - ent_coef * entropy
                
                # Actor反向传播
                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optim.step()
                # ========== 计算并保存训练统计信息 ==========
                with torch.no_grad():
                    # KL散度（近似）
                    log_ratio = torch.log(ratio + 1e-8)
                    approx_kl = (log_ratio * (ratio - 1)).mean().item()
                    
                    # Clip fraction
                    clip_fraction = ((ratio < 1 - clip_range) | (ratio > 1 + clip_range)).float().mean().item()
                    
                    # Explained variance
                    value_pred = critic(b_states)
                    if b_returns.numel() > 1:
                        explained_var = 1 - torch.var(b_returns - value_pred) / (torch.var(b_returns) + 1e-8)
                        explained_var = explained_var.item() if not torch.isnan(explained_var) else 0.0
                    else:
                        explained_var = 0.0
                
                # 更新统计信息（使用最后一次batch的值）
                train_stats['approx_kl'] = approx_kl
                train_stats['clip_fraction'] = clip_fraction
                train_stats['entropy_loss'] = -entropy.item()  # 注意：stable-baselines3记录的是负熵
                train_stats['explained_variance'] = explained_var
                train_stats['policy_loss'] = actor_loss.item()
                train_stats['value_loss'] = critic_loss.item()
                train_stats['td_loss'] = td_loss.item()
                train_stats['baseline_loss'] = baseline_loss.item()
                train_stats['value_mean'] = b_values.mean().item()
                train_stats['value_std'] = b_values.std(unbiased=False).item()
                train_stats['actor_weight_mean'] = actor_weights.mean().item()
                train_stats['actor_weight_std'] = actor_weights.std(unbiased=False).item()
                baseline_gap = baseline_tensor.detach() - b_values.detach()
                train_stats['gva/baseline_bank_size'] = len(greedy_baseline_bank)
                train_stats['gva/baseline_hit_rate'] = b_baseline_hits.mean().item()
                train_stats['gva/baseline_target_mean'] = baseline_tensor.mean().item()
                train_stats['gva/baseline_target_std'] = baseline_tensor.std(unbiased=False).item()
                train_stats['gva/baseline_gap_mean'] = baseline_gap.mean().item()
                train_stats['gva/baseline_gap_std'] = baseline_gap.std(unbiased=False).item()
                train_stats['gva/greedy_updates'] = gva_stats['greedy_updates']
                train_stats['gva/greedy_success'] = gva_stats['greedy_success']
                train_stats['gva/greedy_path_len_mean'] = gva_stats['greedy_path_len_mean']
                train_stats['gva/greedy_terminal_reward_mean'] = gva_stats['greedy_terminal_reward_mean']
                train_stats['total_loss'] = (actor_loss + vf_coef * critic_loss - ent_coef * entropy).item()
                train_stats['n_updates'] += 1
        
        # 回调：每收集rollout_length个timesteps后记录一次（相当于每1个rollout记录一次）
        if callback is not None:
            callback.num_timesteps = timestep
            
            # ========== 记录rollout统计信息（累积所有episode）==========
            if len(rewards) > 0:
                episode_lengths = []
                episode_rewards = []
                current_ep_len = 0
                current_ep_rew = 0
                
                for r, d in zip(rewards, dones):
                    current_ep_len += 1
                    current_ep_rew += r
                    if d:
                        episode_lengths.append(current_ep_len)
                        episode_rewards.append(current_ep_rew)
                        current_ep_len = 0
                        current_ep_rew = 0
                
                if current_ep_len > 0:
                    episode_lengths.append(current_ep_len)
                    episode_rewards.append(current_ep_rew)
                
                if len(episode_lengths) > 0:
                    callback.logger.record('rollout/ep_len_mean', np.mean(episode_lengths))
                    callback.logger.record('rollout/ep_rew_mean', np.mean(episode_rewards))
            
            # ========== 记录训练统计信息 ==========
            callback.logger.record('train/approx_kl', train_stats['approx_kl'])
            callback.logger.record('train/clip_fraction', train_stats['clip_fraction'])
            callback.logger.record('train/clip_range', clip_range)
            callback.logger.record('train/entropy_loss', train_stats['entropy_loss'])
            callback.logger.record('train/explained_variance', train_stats['explained_variance'])
            callback.logger.record('train/learning_rate', actor_lr)
            callback.logger.record('train/loss', train_stats['total_loss'])
            callback.logger.record('train/n_updates', train_stats['n_updates'])
            callback.logger.record('train/policy_gradient_loss', train_stats['policy_loss'])
            callback.logger.record('train/value_loss', train_stats['value_loss'])
            callback.logger.record('train/td_loss', train_stats['td_loss'])
            callback.logger.record('train/baseline_loss', train_stats['baseline_loss'])
            callback.logger.record('train/value_mean', train_stats['value_mean'])
            callback.logger.record('train/value_std', train_stats['value_std'])
            callback.logger.record('train/advantage_mean', train_stats['advantage_mean'])
            callback.logger.record('train/advantage_std', train_stats['advantage_std'])
            callback.logger.record('train/actor_weight_mean', train_stats['actor_weight_mean'])
            callback.logger.record('train/actor_weight_std', train_stats['actor_weight_std'])
            callback.logger.record('gva/baseline_bank_size', train_stats['gva/baseline_bank_size'])
            callback.logger.record('gva/baseline_hit_rate', train_stats['gva/baseline_hit_rate'])
            callback.logger.record('gva/baseline_target_mean', train_stats['gva/baseline_target_mean'])
            callback.logger.record('gva/baseline_target_std', train_stats['gva/baseline_target_std'])
            callback.logger.record('gva/baseline_gap_mean', train_stats['gva/baseline_gap_mean'])
            callback.logger.record('gva/baseline_gap_std', train_stats['gva/baseline_gap_std'])
            callback.logger.record('gva/greedy_updates', train_stats['gva/greedy_updates'])
            callback.logger.record('gva/greedy_success', train_stats['gva/greedy_success'])
            callback.logger.record('gva/greedy_path_len_mean', train_stats['gva/greedy_path_len_mean'])
            callback.logger.record('gva/greedy_terminal_reward_mean', train_stats['gva/greedy_terminal_reward_mean'])
            
            # ========== 记录时间统计信息 ==========
            import time
            if not hasattr(callback, '_start_time'):
                callback._start_time = time.time()
            elapsed = time.time() - callback._start_time
            callback.logger.record('time/iterations', rollout_count)
            callback.logger.record('time/time_elapsed', int(elapsed))
            callback.logger.record('time/total_timesteps', timestep)
            if elapsed > 0:
                callback.logger.record('time/fps', int(timestep / elapsed))
            
            # 调用回调（会记录pool和test指标，并打印）
            callback.on_rollout_end()
    
    return actor, critic