#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression, Feature, OutOfDataRangeError, Ref
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib


def build_target() -> Expression:
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


class PPOFilterPool(AlphaPoolBase):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        device: torch.device,
        mutual_threshold: float = 0.99,
        candidate_limit: int = 2000,
        l1_alpha: float = 5e-3,
        filter_mode: str = "strong",
    ) -> None:
        super().__init__(capacity, calculator, device)
        self.mutual_threshold = mutual_threshold
        self.candidate_limit = candidate_limit
        self.l1_alpha = l1_alpha
        self.filter_mode = filter_mode
        self.candidates: Dict[str, Tuple[float, Expression]] = {}
        self.pool = self._build_filtered_pool()

    @property
    def state(self) -> Dict[str, Any]:
        return self.pool.state

    def to_json_dict(self) -> Dict[str, Any]:
        return self.pool.to_json_dict()

    def try_new_expr(self, expr: Expression) -> float:
        key = str(expr)
        if key in self.candidates:
            return self._reward_from_score(self.candidates[key][0])
        try:
            score = float(self.calculator.calc_single_IC_ret(expr))
        except (OutOfDataRangeError, FloatingPointError, RuntimeError, ValueError, ZeroDivisionError, TypeError):
            return 0.0
        if not np.isfinite(score):
            return 0.0
        self.eval_cnt += 1
        self.candidates[key] = (score, expr)
        if len(self.candidates) > self.candidate_limit:
            keep = sorted(self.candidates.items(), key=lambda kv: self._rank_score(kv[1][0]), reverse=True)[: self.candidate_limit]
            self.candidates = dict(keep)
        if self._rank_score(score) > self.best_ic_ret:
            self.best_ic_ret = self._rank_score(score)
        return self._reward_from_score(score)

    def _rank_score(self, score: float) -> float:
        return abs(score) if self.filter_mode == "strong" else max(score, 0.0)

    def _reward_from_score(self, score: float) -> float:
        return abs(score) if self.filter_mode == "strong" else max(score, 0.0)

    def _selected_exprs(self) -> List[Expression]:
        selected: List[Expression] = []
        for score, expr in sorted(self.candidates.values(), key=lambda item: self._rank_score(item[0]), reverse=True):
            if self.filter_mode == "weak" and score <= 0:
                continue
            if len(selected) >= self.capacity:
                break
            redundant = False
            for old in selected:
                try:
                    mutual = float(self.calculator.calc_mutual_IC(expr, old))
                except Exception:
                    redundant = True
                    break
                if np.isfinite(mutual) and abs(mutual) > self.mutual_threshold:
                    redundant = True
                    break
            if not redundant:
                selected.append(expr)
        return selected

    def _build_filtered_pool(self) -> MseAlphaPool:
        pool = MseAlphaPool(
            capacity=self.capacity,
            calculator=self.calculator,
            ic_lower_bound=None,
            l1_alpha=self.l1_alpha,
            device=self.device,
        )
        exprs = self._selected_exprs()
        if exprs:
            if self.filter_mode == "weak":
                pool.force_load_exprs(exprs, weights=[1.0 / len(exprs)] * len(exprs))
            else:
                pool.force_load_exprs(exprs)
        pool.eval_cnt = self.eval_cnt
        return pool

    def refresh_pool(self) -> MseAlphaPool:
        self.pool = self._build_filtered_pool()
        self.size = self.pool.size
        return self.pool

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        self.refresh_pool()
        if self.pool.size == 0:
            return 0.0, 0.0
        return self.pool.test_ensemble(calculator)


class PPOFilterCallback(BaseCallback):
    def __init__(self, save_path: str, pool: PPOFilterPool, valid_calc: QLibStockDataCalculator, test_calc: QLibStockDataCalculator, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.save_path = save_path
        self.pool = pool
        self.valid_calc = valid_calc
        self.test_calc = test_calc
        os.makedirs(save_path, exist_ok=True)
        self.metrics_path = os.path.join(save_path, 'metrics.csv')
        with open(self.metrics_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fields())
            writer.writeheader()

    def _fields(self) -> List[str]:
        return [
            'timestep', 'candidate_count', 'pool/size', 'pool/eval_cnt', 'pool/best_single_abs_ic', 'pool/best_ic_ret',
            'valid/ic_mean', 'valid/rank_ic_mean', 'test/ic_mean', 'test/rank_ic_mean',
            'rollout/ep_len_mean', 'rollout/ep_rew_mean', 'train/approx_kl', 'train/clip_fraction', 'train/entropy_loss',
            'train/explained_variance', 'train/loss', 'train/n_updates', 'train/policy_gradient_loss', 'train/value_loss',
            'time/fps', 'time/iterations', 'time/time_elapsed'
        ]

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        filtered = self.pool.refresh_pool()
        if filtered.size > 0:
            valid_ic, valid_ric = filtered.test_ensemble(self.valid_calc)
            test_ic, test_ric = filtered.test_ensemble(self.test_calc)
        else:
            valid_ic = valid_ric = test_ic = test_ric = 0.0
        row: Dict[str, Any] = {
            'timestep': self.num_timesteps,
            'candidate_count': len(self.pool.candidates),
            'pool/size': filtered.size,
            'pool/eval_cnt': self.pool.eval_cnt,
            'pool/best_single_abs_ic': self.pool.best_ic_ret,
            'pool/best_ic_ret': filtered.best_ic_ret,
            'valid/ic_mean': valid_ic,
            'valid/rank_ic_mean': valid_ric,
            'test/ic_mean': test_ic,
            'test/rank_ic_mean': test_ric,
        }
        if hasattr(self.logger, 'name_to_value'):
            for key in self._fields():
                if key in row:
                    continue
                if key in self.logger.name_to_value:
                    row[key] = self.logger.name_to_value[key]
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fields(), extrasaction='ignore')
            writer.writerow(row)
        with open(os.path.join(self.save_path, f'{self.num_timesteps}_steps_pool.json'), 'w') as f:
            json.dump(filtered.to_json_dict(), f, indent=2)
        if self.verbose:
            print(row, flush=True)


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.replace('[', '').replace(']', '').split(',') if x.strip()]


def build_data(args: argparse.Namespace, device: torch.device) -> Tuple[QLibStockDataCalculator, QLibStockDataCalculator, QLibStockDataCalculator]:
    initialize_qlib(args.qlib_data_path, kernels=args.qlib_kernels)
    target = build_target()
    train = StockData(args.instruments, args.train_start, args.train_end, device=device)
    valid = StockData(args.instruments, args.valid_start, args.valid_end, device=device)
    test = StockData(args.instruments, args.test_start, args.test_end, device=device)
    return QLibStockDataCalculator(train, target), QLibStockDataCalculator(valid, target), QLibStockDataCalculator(test, target)


def run_one(args: argparse.Namespace, seed: int) -> None:
    reseed_everything(seed)
    if args.device_str == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device_str)
    train_calc, valid_calc, test_calc = build_data(args, device)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(args.output_root, f'{args.method_name}_s{seed}', 'results', f'csi300_{args.pool_capacity}_{seed}_{timestamp}_ppo_filter')
    os.makedirs(save_path, exist_ok=True)
    pool = PPOFilterPool(
        capacity=args.pool_capacity,
        calculator=train_calc,
        device=device,
        mutual_threshold=args.mutual_threshold,
        candidate_limit=args.candidate_limit,
        l1_alpha=args.l1_alpha,
        filter_mode=args.filter_mode,
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=args.print_expr)
    callback = PPOFilterCallback(save_path, pool, valid_calc, test_calc, verbose=1)
    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(n_layers=2, d_model=128, dropout=0.1, device=device),
        ),
        gamma=1.0,
        ent_coef=0.01,
        batch_size=args.batch_size,
        tensorboard_log=os.path.join(args.output_root, f'{args.method_name}_s{seed}', 'tensorboard'),
        device=device,
        verbose=1,
        n_steps=args.n_steps,
        n_epochs=args.ppo_epochs,
    )
    model.learn(total_timesteps=args.steps, callback=callback)
    final_pool = pool.refresh_pool()
    with open(os.path.join(save_path, 'final_pool.json'), 'w') as f:
        json.dump(final_pool.to_json_dict(), f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--random_seeds', default='0')
    p.add_argument('--method_name', default='ppo_filter')
    p.add_argument('--output_root', default='/root/alpha_1203/gva_factor_experiments/runs_newdata/ppo_filter_manual')
    p.add_argument('--qlib_data_path', default='/root/autodl-tmp/cn_data_akshare_2010_2026')
    p.add_argument('--qlib_kernels', type=int, default=1)
    p.add_argument('--device_str', default='auto')
    p.add_argument('--instruments', default='csi300')
    p.add_argument('--pool_capacity', type=int, default=10)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--n_steps', type=int, default=64)
    p.add_argument('--ppo_epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--mutual_threshold', type=float, default=0.99)
    p.add_argument('--candidate_limit', type=int, default=2000)
    p.add_argument('--filter_mode', choices=['weak', 'strong'], default='strong')
    p.add_argument('--l1_alpha', type=float, default=5e-3)
    p.add_argument('--train_start', default='2010-01-04')
    p.add_argument('--train_end', default='2021-12-31')
    p.add_argument('--valid_start', default='2022-01-04')
    p.add_argument('--valid_end', default='2023-12-29')
    p.add_argument('--test_start', default='2024-01-02')
    p.add_argument('--test_end', default='2026-05-28')
    p.add_argument('--print_expr', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for seed in parse_seeds(args.random_seeds):
        run_one(args, seed)
