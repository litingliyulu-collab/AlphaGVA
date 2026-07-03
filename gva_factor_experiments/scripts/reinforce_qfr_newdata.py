#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from alphagen.data.expression import Feature, Ref
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib


class MaskedPolicy(nn.Module):
    def __init__(
        self,
        observation_space,
        action_dim: int,
        n_layers: int,
        d_model: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.extractor = LSTMSharedNet(
            observation_space=observation_space,
            n_layers=n_layers,
            d_model=d_model,
            dropout=dropout,
            device=device,
        )
        self.actor = nn.Linear(d_model, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.extractor(obs))


@dataclass
class EpisodeResult:
    reward: float
    log_prob_sum: Optional[torch.Tensor]
    entropy_sum: Optional[torch.Tensor]
    action_steps: int
    pool_eval_delta: int


def build_target():
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]


def snapshot_pool(pool: MseAlphaPool) -> Dict[str, object]:
    return {
        "size": pool.size,
        "exprs": list(pool.exprs),
        "single_ics": pool.single_ics.copy(),
        "weights": pool._weights.copy(),
        "mutual_ics": pool._mutual_ics.copy(),
        "extra_info": list(pool._extra_info),
        "best_obj": pool.best_obj,
        "best_ic_ret": pool.best_ic_ret,
        "update_history": list(pool.update_history),
        "failure_cache": set(pool._failure_cache),
        "eval_cnt": pool.eval_cnt,
    }


def restore_pool(pool: MseAlphaPool, state: Dict[str, object]) -> None:
    pool.size = int(state["size"])
    pool.exprs = list(state["exprs"])  # type: ignore[assignment]
    pool.single_ics = state["single_ics"].copy()  # type: ignore[union-attr]
    pool._weights = state["weights"].copy()  # type: ignore[union-attr]
    pool._mutual_ics = state["mutual_ics"].copy()  # type: ignore[union-attr]
    pool._extra_info = list(state["extra_info"])  # type: ignore[arg-type]
    pool.best_obj = float(state["best_obj"])
    pool.best_ic_ret = float(state["best_ic_ret"])
    pool.update_history = list(state["update_history"])  # type: ignore[assignment]
    pool._failure_cache = set(state["failure_cache"])  # type: ignore[arg-type]
    pool.eval_cnt = int(state["eval_cnt"])


def evaluate_pool(pool: MseAlphaPool, valid_calc: QLibStockDataCalculator, test_calc: QLibStockDataCalculator) -> Dict[str, float]:
    if pool.size == 0:
        return {
            "valid/ic_mean": 0.0,
            "valid/rank_ic_mean": 0.0,
            "test/ic_mean": 0.0,
            "test/rank_ic_mean": 0.0,
        }
    valid_ic, valid_ric = pool.test_ensemble(valid_calc)
    test_ic, test_ric = pool.test_ensemble(test_calc)
    return {
        "valid/ic_mean": float(valid_ic),
        "valid/rank_ic_mean": float(valid_ric),
        "test/ic_mean": float(test_ic),
        "test/rank_ic_mean": float(test_ric),
    }


def write_metrics_header(path: str) -> None:
    fields = [
        "timestep",
        "episode",
        "formula_eval_cnt",
        "sample_formula_eval_cnt",
        "baseline_formula_eval_cnt",
        "baseline_action_steps",
        "reward",
        "baseline_reward",
        "advantage",
        "reinforce_baseline",
        "loss",
        "pool/size",
        "pool/eval_cnt",
        "pool/best_ic_ret",
        "valid/ic_mean",
        "valid/rank_ic_mean",
        "test/ic_mean",
        "test/rank_ic_mean",
    ]
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def append_metrics(path: str, row: Dict[str, float]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)


def rollout_episode(
    env,
    policy: MaskedPolicy,
    device: torch.device,
    deterministic: bool = False,
) -> EpisodeResult:
    obs, _ = env.reset()
    done = False
    action_steps = 0
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    reward = 0.0
    pool_eval_before = env.unwrapped.pool.eval_cnt

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(obs_tensor).squeeze(0)
        mask = torch.as_tensor(env.action_masks(), dtype=torch.bool, device=device)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        dist = Categorical(logits=masked_logits)
        action = int(torch.argmax(masked_logits).item()) if deterministic else int(dist.sample().item())
        if not deterministic:
            action_tensor = torch.as_tensor(action, device=device)
            log_probs.append(dist.log_prob(action_tensor))
            entropies.append(dist.entropy())
        obs, reward, done, _, _ = env.step(action)
        action_steps += 1

    pool_eval_delta = int(env.unwrapped.pool.eval_cnt - pool_eval_before)
    return EpisodeResult(
        reward=float(reward),
        log_prob_sum=torch.stack(log_probs).sum() if log_probs else None,
        entropy_sum=torch.stack(entropies).sum() if entropies else None,
        action_steps=action_steps,
        pool_eval_delta=pool_eval_delta,
    )


def build_pool(pool_capacity: int, calculator: QLibStockDataCalculator, l1_alpha: float, device: torch.device) -> MseAlphaPool:
    return MseAlphaPool(
        capacity=pool_capacity,
        calculator=calculator,
        ic_lower_bound=None,
        l1_alpha=l1_alpha,
        device=device,
    )


def make_run_dir(output_root: str, method_name: str, seed: int, instruments: str, pool_capacity: int, algo: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(
        output_root,
        f"{method_name}_s{seed}",
        "results",
        f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{algo}",
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_one(args: argparse.Namespace, seed: int) -> None:
    reseed_everything(seed)
    device = torch.device("cuda:0" if args.device_str == "auto" and torch.cuda.is_available() else args.device_str)
    initialize_qlib(args.qlib_data_path, kernels=args.qlib_kernels)

    target = build_target()
    train = StockData(args.instruments, args.train_start, args.train_end, device=device)
    valid = StockData(args.instruments, args.valid_start, args.valid_end, device=device)
    test = StockData(args.instruments, args.test_start, args.test_end, device=device)
    train_calc = QLibStockDataCalculator(train, target)
    valid_calc = QLibStockDataCalculator(valid, target)
    test_calc = QLibStockDataCalculator(test, target)

    pool = build_pool(args.pool_capacity, train_calc, args.l1_alpha, device)
    env = AlphaEnv(pool=pool, device=device, print_expr=args.print_expr)
    policy = MaskedPolicy(
        observation_space=env.observation_space,
        action_dim=env.action_space.n,
        n_layers=args.n_layers,
        d_model=args.d_model,
        dropout=args.dropout,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    method_name = args.method_name or args.algo
    run_dir = make_run_dir(args.output_root, method_name, seed, args.instruments, args.pool_capacity, args.algo)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    write_metrics_header(metrics_path)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"seed": seed}, f, indent=2)

    print(
        f"[REINFORCE_QFR_START] algo={args.algo} seed={seed} steps={args.steps} "
        f"split={args.train_start}:{args.train_end}/{args.valid_start}:{args.valid_end}/{args.test_start}:{args.test_end} "
        f"run_dir={run_dir}",
        flush=True,
    )

    timestep = 0
    episode = 0
    formula_eval_cnt = 0
    sample_formula_eval_cnt = 0
    baseline_formula_eval_cnt = 0
    baseline_action_steps = 0
    reinforce_baseline = 0.0
    next_eval_step = args.eval_every_steps
    last_row: Dict[str, float] = {}

    while timestep < args.steps:
        episode += 1
        original_state = snapshot_pool(pool)
        sample = rollout_episode(env, policy, device, deterministic=False)
        committed_state = snapshot_pool(pool)
        timestep += sample.action_steps
        sample_formula_eval_cnt += sample.pool_eval_delta
        formula_eval_cnt += sample.pool_eval_delta

        baseline_reward = reinforce_baseline
        if args.algo == "qfr":
            restore_pool(pool, original_state)
            greedy = rollout_episode(env, policy, device, deterministic=True)
            baseline_reward = greedy.reward
            baseline_formula_eval_cnt += greedy.pool_eval_delta
            formula_eval_cnt += greedy.pool_eval_delta
            baseline_action_steps += greedy.action_steps
            restore_pool(pool, committed_state)
        else:
            greedy = None

        advantage = sample.reward - baseline_reward
        if args.advantage_clip > 0:
            advantage = float(np.clip(advantage, -args.advantage_clip, args.advantage_clip))

        if sample.log_prob_sum is not None and sample.entropy_sum is not None:
            loss = -float(advantage) * sample.log_prob_sum - args.entropy_coef * sample.entropy_sum
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
        else:
            loss_value = 0.0

        if args.algo == "reinforce":
            reinforce_baseline = (
                args.baseline_ewma * reinforce_baseline
                + (1.0 - args.baseline_ewma) * sample.reward
            )

        should_eval = timestep >= next_eval_step or timestep >= args.steps
        if should_eval:
            evals = evaluate_pool(pool, valid_calc, test_calc)
            row = {
                "timestep": int(min(timestep, args.steps)),
                "episode": int(episode),
                "formula_eval_cnt": int(formula_eval_cnt),
                "sample_formula_eval_cnt": int(sample_formula_eval_cnt),
                "baseline_formula_eval_cnt": int(baseline_formula_eval_cnt),
                "baseline_action_steps": int(baseline_action_steps),
                "reward": float(sample.reward),
                "baseline_reward": float(baseline_reward),
                "advantage": float(advantage),
                "reinforce_baseline": float(reinforce_baseline),
                "loss": float(loss_value),
                "pool/size": int(pool.size),
                "pool/eval_cnt": int(pool.eval_cnt),
                "pool/best_ic_ret": float(pool.best_ic_ret),
                **evals,
            }
            append_metrics(metrics_path, row)
            with open(os.path.join(run_dir, f"{int(min(timestep, args.steps))}_steps_pool.json"), "w") as f:
                json.dump(pool.to_json_dict(), f, indent=2)
            print("[REINFORCE_QFR_METRIC]", row, flush=True)
            last_row = row
            while next_eval_step <= timestep:
                next_eval_step += args.eval_every_steps

    torch.save(policy.state_dict(), os.path.join(run_dir, "policy_final.pth"))
    with open(os.path.join(run_dir, "final_pool.json"), "w") as f:
        json.dump(pool.to_json_dict(), f, indent=2)
    with open(os.path.join(run_dir, "final_summary.json"), "w") as f:
        json.dump(last_row, f, indent=2)
    print(f"[REINFORCE_QFR_DONE] algo={args.algo} seed={seed} run_dir={run_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["reinforce", "qfr"], default="reinforce")
    parser.add_argument("--random_seeds", default="0")
    parser.add_argument("--method_name", default=None)
    parser.add_argument("--output_root", default="/root/alpha_1203/gva_factor_experiments/runs_newdata/reinforce_qfr_manual")
    parser.add_argument("--qlib_data_path", default="/root/autodl-tmp/cn_data_akshare_2010_2026")
    parser.add_argument("--qlib_kernels", type=int, default=1)
    parser.add_argument("--device_str", default="auto")
    parser.add_argument("--instruments", default="csi300")
    parser.add_argument("--pool_capacity", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--eval_every_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--baseline_ewma", type=float, default=0.9)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l1_alpha", type=float, default=5e-3)
    parser.add_argument("--print_expr", action="store_true")
    parser.add_argument("--train_start", default="2010-01-04")
    parser.add_argument("--train_end", default="2021-12-31")
    parser.add_argument("--valid_start", default="2022-01-04")
    parser.add_argument("--valid_end", default="2023-12-29")
    parser.add_argument("--test_start", default="2024-01-02")
    parser.add_argument("--test_end", default="2026-05-28")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for seed in parse_seeds(args.random_seeds):
        run_one(args, seed)

