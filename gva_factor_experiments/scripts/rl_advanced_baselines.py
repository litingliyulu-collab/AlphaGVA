#!/usr/bin/env python3
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Tuple

import fire
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C, DQN
from sb3_contrib import QRDQN

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.reward import RewardEvaluator
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib

# Reuse the project callback/checkpoint code so output format matches rl_v1.py.
from scripts.rl_v1 import CustomCallback, read_warm_pool


class LegalActionMapWrapper(gym.Wrapper):
    """Map illegal actions proposed by non-mask RL algorithms to legal actions."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.illegal_action_count = 0
        self.total_action_count = 0

    def action_masks(self):
        return self.env.action_masks()

    def step(self, action):
        self.total_action_count += 1
        mask = self.action_masks()
        legal = np.flatnonzero(mask)
        a = int(action)
        if len(legal) == 0:
            raise RuntimeError("No legal actions available")
        if a < 0 or a >= len(mask) or not mask[a]:
            self.illegal_action_count += 1
            a = int(legal[a % len(legal)])
        obs, reward, terminated, truncated, info = self.env.step(a)
        info = dict(info)
        info["illegal_action_count"] = self.illegal_action_count
        info["total_action_count"] = self.total_action_count
        return obs, reward, terminated, truncated, info


class SparseCheckpointCallback(CustomCallback):
    """Keep metrics/pool snapshots, but save heavyweight model zips sparsely."""

    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        model_checkpoint_interval: int = 5000,
        keep_model_checkpoints: int = 1,
    ):
        super().__init__(save_path, test_calculators=test_calculators, verbose=verbose)
        self.model_checkpoint_interval = int(model_checkpoint_interval)
        self.keep_model_checkpoints = int(keep_model_checkpoints)
        self._last_model_checkpoint_step = -1

    def save_checkpoint(self, force_model: bool = False):
        path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")

        with open(f"{path}_pool.json", "w") as f:
            json.dump(self.pool.to_json_dict(), f)

        should_save_model = force_model
        if self.model_checkpoint_interval > 0:
            should_save_model = should_save_model or (
                self.num_timesteps - self._last_model_checkpoint_step >= self.model_checkpoint_interval
            )
        if not should_save_model:
            return

        if hasattr(self.model, "save"):
            self.model.save(path)
        elif hasattr(self.model, "actor"):
            torch.save({
                "actor": self.model.actor.state_dict(),
                "critic": self.model.critic.state_dict(),
            }, f"{path}.pth")
        self._last_model_checkpoint_step = self.num_timesteps
        self._prune_model_checkpoints()

    def _prune_model_checkpoints(self):
        if self.keep_model_checkpoints <= 0:
            return
        checkpoints = sorted(
            Path(self.save_path).glob("*_steps.zip"),
            key=lambda p: p.stat().st_mtime,
        )
        for old_path in checkpoints[:-self.keep_model_checkpoints]:
            try:
                old_path.unlink()
            except OSError as exc:
                print(f"[WARN] failed to delete old checkpoint {old_path}: {exc}", flush=True)


def build_pool(pool_capacity, calculator, reward_mode, device, warm_exprs=None):
    pool = MseAlphaPool(
        capacity=pool_capacity,
        calculator=calculator,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        reward_evaluator=RewardEvaluator.from_mode(reward_mode),
        device=device,
    )
    if warm_exprs:
        pool.force_load_exprs(warm_exprs)
    return pool


def build_model(algo, env, common_policy, n_steps, device, tb_dir):
    if algo == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            policy_kwargs=common_policy,
            gamma=1.0,
            n_steps=n_steps,
            ent_coef=0.01,
            learning_rate=7e-4,
            device=device,
            tensorboard_log=str(tb_dir),
            verbose=1,
        )
    if algo == "dqn":
        return DQN(
            "MlpPolicy",
            env,
            policy_kwargs=common_policy,
            gamma=1.0,
            learning_rate=1e-4,
            buffer_size=5000,
            learning_starts=512,
            batch_size=64,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            device=device,
            tensorboard_log=str(tb_dir),
            verbose=1,
        )
    return QRDQN(
        "MlpPolicy",
        env,
        policy_kwargs=common_policy,
        gamma=1.0,
        learning_rate=1e-4,
        buffer_size=5000,
        learning_starts=512,
        batch_size=64,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        device=device,
        tensorboard_log=str(tb_dir),
        verbose=1,
    )


def run_one(
    algo: str = "a2c",
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 30000,
    n_steps: int = 64,
    qlib_data_path: str = "/root/autodl-tmp/cn_data_akshare_2010_2026",
    qlib_kernels: int = 1,
    device_str: str = "auto",
    train_start: str = "2010-01-04",
    train_end: str = "2021-12-31",
    valid_start: str = "2022-01-04",
    valid_end: str = "2023-12-29",
    test_start: str = "2024-01-02",
    test_end: str = "2026-05-28",
    output_root: str = "/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629",
    reward_mode: str = "original",
    warm_pool_json: Optional[str] = None,
    warm_sort_by_abs_weight: bool = False,
    resume_checkpoint: Optional[str] = None,
    resume_warm_pool_json: Optional[str] = None,
    model_checkpoint_interval: int = 5000,
    keep_model_checkpoints: int = 1,
):
    algo = algo.lower()
    if algo not in {"a2c", "dqn", "qrdqn"}:
        raise ValueError(f"unsupported algo: {algo}")
    reseed_everything(seed)
    initialize_qlib(qlib_data_path, kernels=qlib_kernels)
    device = torch.device("cuda:0" if device_str == "auto" and torch.cuda.is_available() else device_str)

    print(f"[RL_BASELINE] algo={algo} seed={seed} steps={steps} device={device} output_root={output_root}", flush=True)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    datasets = [
        StockData(instruments, train_start, train_end, device=device),
        StockData(instruments, valid_start, valid_end, device=device),
        StockData(instruments, test_start, test_end, device=device),
    ]
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]
    warm_pool_source = resume_warm_pool_json or warm_pool_json
    warm_exprs = read_warm_pool(warm_pool_source, seed, warm_sort_by_abs_weight) if warm_pool_source else []
    pool = build_pool(pool_capacity, calculators[0], reward_mode, device, warm_exprs)
    env = LegalActionMapWrapper(AlphaEnv(pool=pool, device=device, print_expr=False))

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    method_name = f"{algo}_baseline_s{seed}"
    resume_tag = "resume" if resume_checkpoint else "fresh"
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{algo}_{reward_mode}_{resume_tag}"
    save_path = Path(output_root) / method_name / "results" / name_prefix
    tb_dir = Path(output_root) / method_name / "tensorboard"
    save_path.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    callback = SparseCheckpointCallback(
        str(save_path),
        test_calculators=calculators[1:],
        verbose=1,
        model_checkpoint_interval=model_checkpoint_interval,
        keep_model_checkpoints=keep_model_checkpoints,
    )
    common_policy = dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(n_layers=1, d_model=64, dropout=0.0, device=device),
    )
    if resume_checkpoint:
        algo_class = {"a2c": A2C, "dqn": DQN, "qrdqn": QRDQN}[algo]
        print(f"[RL_BASELINE_RESUME] checkpoint={resume_checkpoint} warm_pool={warm_pool_source}", flush=True)
        model = algo_class.load(
            resume_checkpoint,
            env=env,
            device=device,
            tensorboard_log=str(tb_dir),
        )
        remaining_steps = max(int(steps) - int(getattr(model, "num_timesteps", 0)), 0)
        print(f"[RL_BASELINE_RESUME] loaded_steps={getattr(model, 'num_timesteps', 0)} remaining_steps={remaining_steps}", flush=True)
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            tb_log_name=name_prefix,
            reset_num_timesteps=False,
        )
    else:
        model = build_model(algo, env, common_policy, n_steps, device, tb_dir)
        model.learn(total_timesteps=int(steps), callback=callback, tb_log_name=name_prefix)

    callback.save_checkpoint(force_model=True)
    with open(save_path / "final_pool.json", "w") as f:
        json.dump(pool.to_json_dict(), f, indent=2)

    core_env = model.get_env().envs[0]
    stats = {
        "algo": algo,
        "seed": seed,
        "steps": steps,
        "model_num_timesteps": int(getattr(model, "num_timesteps", -1)),
        "illegal_action_count": int(getattr(core_env, "illegal_action_count", -1)),
        "total_action_count": int(getattr(core_env, "total_action_count", -1)),
        "resume_checkpoint": resume_checkpoint,
        "resume_warm_pool_json": resume_warm_pool_json,
    }
    with open(save_path / "legal_action_map_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("[RL_BASELINE_DONE]", stats, save_path, flush=True)


def main(
    algos: Union[str, Tuple[str, ...]] = ("a2c", "dqn", "qrdqn"),
    random_seeds: Union[int, Tuple[int, ...]] = 0,
    **kwargs,
):
    if isinstance(algos, str):
        algos = tuple(a.strip() for a in algos.split(",") if a.strip())
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds,)
    for algo in algos:
        for seed in random_seeds:
            run_one(algo=algo, seed=int(seed), **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
