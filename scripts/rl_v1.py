import json
import os
from typing import Optional, Tuple, List, Union
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool, RewardAlignedMseAlphaPool
from alphagen.reward import RewardEvaluator
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction


def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    DIR = "./out_GVA/llm-tests/interaction"
    parser = build_parser()
    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            return [parser.parse(expr) for expr, _ in pool_state]
    return []


def read_warm_pool(path_template: Optional[str], seed: int) -> List[Expression]:
    if not path_template:
        return []
    path = path_template.format(seed=seed)
    parser = build_parser()
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        expr_strings = data.get("exprs", [])
    else:
        expr_strings = data
    exprs = []
    seen = set()
    for expr_str in expr_strings:
        if isinstance(expr_str, (list, tuple)):
            expr_str = expr_str[0]
        if expr_str in seen:
            continue
        seen.add(expr_str)
        try:
            exprs.append(parser.parse(expr_str))
        except Exception as exc:
            print(f"[WarmPool] Skip unparsable expression: {expr_str} ({type(exc).__name__}: {exc})")
    print(f"[WarmPool] Loaded {len(exprs)} expressions from {path}")
    return exprs


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


def build_chat_client(log_dir: str) -> ChatClient:
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url="https://api.ai.cs.ac.cn/v1"),
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=logger
        )
    )

import csv
from pathlib import Path
class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)
        # ========== 新增：保存指标到CSV ==========
        self.metrics_file = os.path.join(self.save_path, 'metrics.csv')
        self.metrics_writer = None
        self.metrics_fieldnames = None
        self._init_metrics_file()
        
        self.llm_use_count = 0
        self.last_llm_use = 0
        self.obj_history: List[Tuple[int, float]] = []
        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n

    def _on_step(self) -> bool:
        return True


    def _init_metrics_file(self):
        """初始化CSV文件，写入表头"""
        self.metrics_fieldnames = [
            'timestep', 'rollout',
            'pool/size', 'pool/significant', 'pool/best_ic_ret', 'pool/eval_cnt',
            'reward/total', 'reward/original_objective', 'reward/quality_score',
            'reward/ic', 'reward/rank_ic', 'reward/rankic_mean', 'reward/ic_std',
            'reward/rank_ic_std', 'reward/ir', 'reward/icir', 'reward/stability',
            'reward/robustness', 'reward/phase_ic_std', 'reward/phase_ic_min',
            'reward/logic_score', 'reward/logic_penalty', 'reward/expr_length',
            'reward/expr_token_count', 'reward/expr_depth', 'reward/expr_complexity',
            'reward/abnormal_ratio', 'reward/low_variance_ratio',
            'reward/sim_ic', 'reward/explicit_redundancy',
            'reward/max_abs_mutual_ic', 'reward/avg_abs_mutual_ic',
            'reward/sim_struct', 'reward/sim_sem', 'reward/sim_dual',
            'reward/implicit_redundancy', 'reward/max_structure_similarity',
            'reward/avg_structure_similarity', 'reward/max_semantic_similarity',
            'reward/avg_semantic_similarity', 'reward/penalty',
            'reward/pool_ic', 'reward/single_ic',
            'rollout/ep_len_mean', 'rollout/ep_rew_mean',
            'test/ic_1', 'test/ic_2', 'test/ic_3', 'test/ic_mean',
            'test/rank_ic_1', 'test/rank_ic_2', 'test/rank_ic_3', 'test/rank_ic_mean',
            'train/approx_kl', 'train/clip_fraction', 'train/entropy_loss',
            'train/explained_variance', 'train/loss', 'train/n_updates',
            'train/policy_gradient_loss', 'train/value_loss',
            'train/td_loss', 'train/baseline_loss',
            'train/value_mean', 'train/value_std',
            'train/advantage_mean', 'train/advantage_std',
            'train/actor_weight_mean', 'train/actor_weight_std',
            'gva/baseline_bank_size', 'gva/baseline_hit_rate',
            'gva/baseline_target_mean', 'gva/baseline_target_std',
            'gva/baseline_gap_mean', 'gva/baseline_gap_std',
            'gva/greedy_updates', 'gva/greedy_success',
            'gva/greedy_path_len_mean', 'gva/greedy_terminal_reward_mean',
            'time/fps', 'time/iterations', 'time/time_elapsed'
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_fieldnames)
            writer.writeheader()
    
    def _save_metrics_to_csv(self, metrics_dict: dict):
        """保存指标到CSV文件"""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_fieldnames, extrasaction='ignore')
            writer.writerow(metrics_dict)
    
    def _reward_metrics(self) -> dict:
        result = getattr(self.pool, 'last_reward_result', None)
        if result is None:
            return {}
        return result.flat_metrics()

    def _record_reward_metrics(self, metrics: dict) -> None:
        for key, value in self._reward_metrics().items():
            self.logger.record(key, value)
            metrics[key] = value

    def _on_rollout_end(self) -> None:
        if self.chat_session is not None:
            self._try_use_llm()

        # ========== 收集所有指标 ==========
        metrics = {
            'timestep': self.num_timesteps,
            'rollout': getattr(self, '_rollout_count', 0),
        }
        
        self.logger.record('pool/size', self.pool.size)
        metrics['pool/size'] = self.pool.size
        
        # 如果pool为空，记录默认值并跳过测试
        if self.pool.size == 0:
            metrics.update({
                'pool/significant': 0,
                'pool/best_ic_ret': -1.0,
                'pool/eval_cnt': self.pool.eval_cnt,
            })
            self._record_reward_metrics(metrics)
            n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
            for i in range(1, len(self.test_calculators) + 1):
                metrics[f'test/ic_{i}'] = 0.0
                metrics[f'test/rank_ic_{i}'] = 0.0
            metrics['test/ic_mean'] = 0.0
            metrics['test/rank_ic_mean'] = 0.0
            if hasattr(self.logger, 'name_to_value'):
                for key in ['rollout/ep_len_mean', 'rollout/ep_rew_mean',
                           'train/approx_kl', 'train/clip_fraction', 'train/entropy_loss',
                           'train/explained_variance', 'train/loss', 'train/n_updates',
                           'train/policy_gradient_loss', 'train/value_loss',
                           'train/td_loss', 'train/baseline_loss',
                           'train/value_mean', 'train/value_std',
                           'train/advantage_mean', 'train/advantage_std',
                           'train/actor_weight_mean', 'train/actor_weight_std',
                           'gva/baseline_bank_size', 'gva/baseline_hit_rate',
                           'gva/baseline_target_mean', 'gva/baseline_target_std',
                           'gva/baseline_gap_mean', 'gva/baseline_gap_std',
                           'gva/greedy_updates', 'gva/greedy_success',
                           'gva/greedy_path_len_mean', 'gva/greedy_terminal_reward_mean',
                           'time/fps', 'time/iterations', 'time/time_elapsed']:
                    if key in self.logger.name_to_value:
                        metrics[key] = self.logger.name_to_value[key]
            self.save_checkpoint()
            if self.verbose >= 1:
                self.logger.dump(step=self.num_timesteps)
            # 保存指标
            self._save_metrics_to_csv(metrics)
            return
        
        # Pool不为空时，正常记录和测试
        metrics['pool/significant'] = (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum()
        metrics['pool/best_ic_ret'] = self.pool.best_ic_ret
        metrics['pool/eval_cnt'] = self.pool.eval_cnt
        self._record_reward_metrics(metrics)
        
        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
            metrics[f'test/ic_{i}'] = ic_test
            metrics[f'test/rank_ic_{i}'] = rank_ic_test
        
        self.logger.record(f'test/ic_mean', ic_test_mean)
        self.logger.record(f'test/rank_ic_mean', rank_ic_test_mean)
        metrics['test/ic_mean'] = ic_test_mean
        metrics['test/rank_ic_mean'] = rank_ic_test_mean
        
        # 从logger获取训练指标（如果存在）
        if hasattr(self.logger, 'name_to_value'):
            for key in ['rollout/ep_len_mean', 'rollout/ep_rew_mean',
                       'train/approx_kl', 'train/clip_fraction', 'train/entropy_loss',
                       'train/explained_variance', 'train/loss', 'train/n_updates',
                       'train/policy_gradient_loss', 'train/value_loss',
                       'train/td_loss', 'train/baseline_loss',
                       'train/value_mean', 'train/value_std',
                       'train/advantage_mean', 'train/advantage_std',
                       'train/actor_weight_mean', 'train/actor_weight_std',
                       'gva/baseline_bank_size', 'gva/baseline_hit_rate',
                       'gva/baseline_target_mean', 'gva/baseline_target_std',
                       'gva/baseline_gap_mean', 'gva/baseline_gap_std',
                       'gva/greedy_updates', 'gva/greedy_success',
                       'gva/greedy_path_len_mean', 'gva/greedy_terminal_reward_mean',
                       'time/fps', 'time/iterations', 'time/time_elapsed']:
                if key in self.logger.name_to_value:
                    metrics[key] = self.logger.name_to_value[key]
        
        self.save_checkpoint()
        
        if self.verbose >= 1:
            self.logger.dump(step=self.num_timesteps)
        
        # ========== 保存指标到CSV ==========
        self._save_metrics_to_csv(metrics)


    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        # 检查是MaskablePPO还是自定义训练
        if hasattr(self.model, 'save'):
            # 原来的MaskablePPO
            self.model.save(path)
        elif hasattr(self.model, 'actor'):
            # 自定义训练的包装对象
            torch.save({
                'actor': self.model.actor.state_dict(),
                'critic': self.model.critic.state_dict(),
            }, f"{path}.pth")
    
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        
        # Pool保存保持不变
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_json_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        print('---------------------------------------------')
        for i in range(self.pool.size):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    def _try_use_llm(self) -> None:
        n_steps = self.num_timesteps
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1
        
        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        logger.debug(
            f"[Step: {n_steps}] Trying to invoke LLM (#{self.llm_use_count}): "
            f"IC={self.pool.best_ic_ret:.4f}, obj={self.pool.best_ic_ret:.4f}")

        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed due to {type(e)}: {str(e)}")

    @property
    def pool(self) -> LinearAlphaPool:
        assert(isinstance(self.env_core.pool, LinearAlphaPool))
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        """
        获取AlphaEnvCore，正确处理所有环境包装层
        环境包装层次：DummyVecEnv -> AlphaEnvWrapper -> AlphaEnvCore
        """
        # 获取起始环境
        if hasattr(self.model, 'env') and self.model.env is not None:
            # 自定义训练模式
            env = self.model.env
        else:
            # MaskablePPO模式
            env = self.training_env
        
        # 递归unwrap，直到找到AlphaEnvCore（有pool属性）
        max_iterations = 10  # 防止无限循环
        for _ in range(max_iterations):
            # 如果是VecEnv（如DummyVecEnv），访问envs[0]
            if hasattr(env, 'envs') and len(env.envs) > 0:
                env = env.envs[0]
                continue
            
            # 如果是Wrapper（如AlphaEnvWrapper），访问env属性
            if hasattr(env, 'env'):
                inner_env = env.env
                # 检查内层环境是否是AlphaEnvCore
                if hasattr(inner_env, 'pool'):
                    return inner_env
                env = inner_env
                continue
            
            # 如果有unwrapped属性
            if hasattr(env, 'unwrapped'):
                unwrapped = env.unwrapped
                if hasattr(unwrapped, 'pool'):
                    return unwrapped
                env = unwrapped
                continue
            
            # 如果当前env就是AlphaEnvCore（有pool属性）
            if hasattr(env, 'pool'):
                return env
            
            # 无法继续unwrap
            break
        
        # 如果循环结束还没找到，抛出详细错误
        raise AttributeError(
            f"Cannot find AlphaEnvCore. Current env type: {type(env)}, "
            f"has envs: {hasattr(env, 'envs')}, "
            f"has env: {hasattr(env, 'env')}, "
            f"has unwrapped: {hasattr(env, 'unwrapped')}, "
            f"has pool: {hasattr(env, 'pool')}"
        )


def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3,
    n_steps: int = 256,
    ppo_epochs: int = 4,
    # 新增参数：选择训练方法
    use_custom_ppo: bool = False,  # False=使用原来的MaskablePPO, True=使用自定义训练
    custom_critic_loss: str = "hybrid",  # "mse", "hybrid", "huber"
    td_weight: float = 0.9,
    baseline_weight: float = 0.1,
    actor_gap_weight: float = 0.0,
    actor_gap_clip: float = 2.0,
    gva_budget_ratio: float = 0.25,
    gva_max_updates_per_rollout: int = 32,
    gva_refresh_interval: int = 10,
    gva_max_rollout_depth: Optional[int] = None,
    gva_min_state_len: int = 1,
    qlib_data_path: str = "/root/autodl-tmp/cn_data_akshare_2010_2026",
    qlib_kernels: int = 1,
    device_str: str = "auto",
    train_start: str = "2010-01-04",
    train_end: str = "2021-12-31",
    valid_start: str = "2022-01-04",
    valid_end: str = "2023-12-29",
    test_start: str = "2024-01-02",
    test_end: str = "2026-05-28",
    method_name: str = None,
    output_root: str = None,
    reward_mode: str = "original",
    reward_align_pool: bool = False,
    warm_pool_json: Optional[str] = None,
):
    reseed_everything(seed)
    initialize_qlib(qlib_data_path, kernels=qlib_kernels)

    llm_replace_n = 0 if not use_llm else llm_replace_n
    print(f"""[Main] Starting training process
    Seed: {seed}
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Total Iteration Steps: {steps}
    PPO Epochs: {ppo_epochs}
    Method Name: {method_name or 'default'}
    Output Root: {output_root or 'legacy out_* directories'}
    Qlib Data Path: {qlib_data_path}
    Qlib Kernels: {qlib_kernels}
    Device: {device_str}
    Train Segment: {train_start} to {train_end}
    Valid Segment: {valid_start} to {valid_end}
    Test Segment: {test_start} to {test_end}
    Critic TD Weight: {td_weight}
    Critic Baseline Weight: {baseline_weight}
    Actor Gap Weight: {actor_gap_weight}
    GVA Budget Ratio: {gva_budget_ratio}
    GVA Max Updates Per Rollout: {gva_max_updates_per_rollout}
    GVA Refresh Interval: {gva_refresh_interval}
    GVA Max Rollout Depth: {gva_max_rollout_depth}
    GVA Min State Length: {gva_min_state_len}
    Reward Mode: {reward_mode}
    Reward Align Pool: {reward_align_pool}
    AlphaGPT-Like Init-Only LLM Usage: {alphagpt_init}
    Use LLM: {use_llm}
    Invoke LLM every N steps: {llm_every_n_steps}
    Replace N alphas with LLM: {llm_replace_n}
    Drop N alphas before LLM: {drop_rl_n}""")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    tag = (
        "agpt" if alphagpt_init else
        "rl" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}_{reward_mode}{'_ralign' if reward_align_pool else ''}"
    
    # ========== 修改：根据 method_name 设置输出目录 ==========
    if output_root:
        run_group = method_name or 'default'
        results_dir = os.path.join(output_root, run_group, 'results')
        tensorboard_dir = os.path.join(output_root, run_group, 'tensorboard')
    elif method_name:
        # 使用指定的方法名称作为输出目录前缀
        results_dir = f"./out_{method_name}/results"
        tensorboard_dir = f"./out_{method_name}/tensorboard"
    else:
        # 默认输出目录
        results_dir = "./out/results"
        tensorboard_dir = "./out/tensorboard"
    
    save_path = os.path.join(results_dir, name_prefix)
    os.makedirs(save_path, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            device=device
        )

    segments = [
        (train_start, train_end),
        (valid_start, valid_end),
        (test_start, test_end),
    ]
    datasets = [get_dataset(*s) for s in segments]
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]
    reward_evaluator = RewardEvaluator.from_mode(reward_mode)
    use_reward_align = reward_align_pool and reward_mode.lower() not in ("original", "v0")
    pool_cls = RewardAlignedMseAlphaPool if use_reward_align else MseAlphaPool

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = pool_cls(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            reward_evaluator=reward_evaluator,
            device=device
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    warm_exprs = read_warm_pool(warm_pool_json, seed)
    chat, inter, pool = None, None, build_pool(warm_exprs)
    if alphagpt_init:
        pool = build_pool(read_alphagpt_init_pool(seed))
    elif use_llm:
        chat = build_chat_client(save_path)
        inter = DefaultInteraction(
            build_parser(), chat, build_pool,
            calculator_train=calculators[0], calculators_test=calculators[1:],
            replace_k=llm_replace_n, forgetful=True
        )
        pool = inter.run()

    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=True
    )
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n
    )
    
    # ========== 根据参数选择训练方法 ==========
    if use_custom_ppo:
        # 使用自定义PPO训练
        from alphagen.rl.custom_ppo_trainer import train_custom_ppo
        
        # 创建包装对象（在训练前创建）
        class ModelWrapper:
            def __init__(self, env=None):
                self.actor = None
                self.critic = None
                self.env = env
                self.num_timesteps = 0
                from stable_baselines3.common.logger import configure
                # 使用新的 tensorboard 目录
                tensorboard_path = os.path.join(tensorboard_dir, name_prefix)
                self.logger = configure(tensorboard_path, ["stdout", "tensorboard"])
            
            def get_env(self):
                """返回环境，用于BaseCallback的training_env属性"""
                # 如果env是单个环境，需要包装成VecEnv格式
                # 为了兼容BaseCallback，我们需要返回一个类似VecEnv的对象
                if self.env is not None:
                    # 创建一个简单的包装，使其看起来像VecEnv
                    class EnvWrapper:
                        def __init__(self, env):
                            self.envs = [env]  # BaseCallback期望envs属性
                    return EnvWrapper(self.env)
                return None
            
            def save(self, path):
                """保存模型"""
                if self.actor is not None and self.critic is not None:
                    torch.save({
                        'actor': self.actor.state_dict(),
                        'critic': self.critic.state_dict(),
                    }, f"{path}.pth")
        
        # 在训练前创建wrapper并设置model
                # 在训练前创建wrapper并设置model
        wrapper = ModelWrapper(env=env)  # ← 传递env
        checkpoint_callback.model = wrapper  # ← 关键：在训练前设置
        
        # 训练
        actor, critic = train_custom_ppo(
            env=env,
            total_timesteps=steps,
            device=device,
            callback=checkpoint_callback,
            critic_loss_type=custom_critic_loss,
            td_weight=td_weight,
            baseline_weight=baseline_weight,
            actor_gap_weight=actor_gap_weight,
            actor_gap_clip=actor_gap_clip,
            gva_budget_ratio=gva_budget_ratio,
            gva_max_updates_per_rollout=gva_max_updates_per_rollout,
            gva_refresh_interval=gva_refresh_interval,
            gva_max_rollout_depth=gva_max_rollout_depth,
            gva_min_state_len=gva_min_state_len,
            rollout_length=n_steps,
            n_epochs=ppo_epochs,
            log_frequency=1,
        )
        
        # 训练完成后更新wrapper
        wrapper.actor = actor
        wrapper.critic = critic
        wrapper.num_timesteps = steps
        
        # 最终保存
        checkpoint_callback.save_checkpoint()
        
    else:
         # 使用原来的MaskablePPO
        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                features_extractor_class=LSTMSharedNet,
                features_extractor_kwargs=dict(
                    n_layers=2,
                    d_model=128,
                    dropout=0.1,
                    device=device,
                ),
            ),
            gamma=1.,
            ent_coef=0.01,
            batch_size=128,
            n_steps=n_steps,
            tensorboard_log=tensorboard_dir,  # 使用新的 tensorboard 目录
            device=device,
            verbose=1,
        )
        model.learn(
            total_timesteps=steps,
            callback=checkpoint_callback,
            tb_log_name=name_prefix,
        )


def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000,
    n_steps=256,
    ppo_epochs: int = 4,
    # 新增参数
    use_custom_ppo: bool = False,  # 是否使用自定义PPO
    custom_critic_loss: str = "hybrid",  # 自定义critic损失类型
    td_weight: float = 0.9,
    baseline_weight: float = 0.1,
    actor_gap_weight: float = 0.0,
    actor_gap_clip: float = 2.0,
    gva_budget_ratio: float = 0.25,
    gva_max_updates_per_rollout: int = 32,
    gva_refresh_interval: int = 10,
    gva_max_rollout_depth: Optional[int] = None,
    gva_min_state_len: int = 1,
    qlib_data_path: str = "/root/autodl-tmp/cn_data_akshare_2010_2026",
    qlib_kernels: int = 1,
    device_str: str = "auto",
    train_start: str = "2010-01-04",
    train_end: str = "2021-12-31",
    valid_start: str = "2022-01-04",
    valid_end: str = "2023-12-29",
    test_start: str = "2024-01-02",
    test_end: str = "2026-05-28",
    method_name: str = None,
    output_root: str = None,
    reward_mode: str = "original",
    reward_align_pool: bool = False,
    warm_pool_json: Optional[str] = None,
    
):
    """
    :param random_seeds: Random seeds
    :param pool_capacity: Maximum size of the alpha pool
    :param instruments: Stock subset name
    :param alphagpt_init: Use an alpha set pre-generated by LLM as the initial pool
    :param use_llm: Enable LLM usage
    :param drop_rl_n: Drop n worst alphas before invoke the LLM
    :param steps: Total iteration steps
    :param llm_every_n_steps: Invoke LLM every n steps
    """
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds, )
    default_steps = {
        10: 200_000,
        20: 250_000,
        50: 300_000,
        100: 350_000
    }
    for s in random_seeds:
        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=pool_capacity,
            steps=default_steps[int(pool_capacity)] if steps is None else int(steps),
            alphagpt_init=alphagpt_init,
            drop_rl_n=drop_rl_n,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps,
            n_steps=n_steps,
            ppo_epochs=ppo_epochs,
            use_custom_ppo=use_custom_ppo,
            custom_critic_loss=custom_critic_loss,
            td_weight=td_weight,
            baseline_weight=baseline_weight,
            actor_gap_weight=actor_gap_weight,
            actor_gap_clip=actor_gap_clip,
            gva_budget_ratio=gva_budget_ratio,
            gva_max_updates_per_rollout=gva_max_updates_per_rollout,
            gva_refresh_interval=gva_refresh_interval,
            gva_max_rollout_depth=gva_max_rollout_depth,
            gva_min_state_len=gva_min_state_len,
            qlib_data_path=qlib_data_path,
            qlib_kernels=qlib_kernels,
            device_str=device_str,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            method_name=method_name,  # 传递方法名称
            output_root=output_root,
            reward_mode=reward_mode,
            reward_align_pool=reward_align_pool,
            warm_pool_json=warm_pool_json,
        )


if __name__ == '__main__':
    fire.Fire(main)
