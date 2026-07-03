from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from alphagen.data.expression import Expression
from alphagen.reward.config import RewardConfig
from alphagen.reward.expression_adapter import ExpressionAdapter, ExpressionFeatures, jaccard
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr


class PositiveEvaluator:
    def __init__(self, config: RewardConfig):
        self.config = config

    def evaluate(self, values: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        values, target = _align_tensors(values, target)
        if values.numel() == 0 or target.numel() == 0:
            return _zero_quality()
        with torch.no_grad():
            ics = batch_pearsonr(values, target)
            rankics = batch_spearmanr(values, target)
            ic_mean = _safe_item(ics.mean())
            rankic_mean = _safe_item(rankics.mean())
            ic_std = _safe_item(ics.std(unbiased=False))
            rankic_std = _safe_item(rankics.std(unbiased=False))
            phases = _phase_means(ics, self.config.n_phases)

        ir = ic_mean / (ic_std + self.config.eps)
        phase_std = float(np.std(phases)) if phases else 0.0
        phase_min = float(np.min(phases)) if phases else 0.0
        stability = -ic_std - self.config.phase_std_weight * phase_std
        robustness = phase_min
        quality_score = (
            self.config.w_ic * ic_mean
            + self.config.w_rankic * rankic_mean
            + self.config.w_ir * ir
            + self.config.w_stability * stability
            + self.config.w_robustness * robustness
        )
        return {
            "ic": ic_mean,
            "rank_ic": rankic_mean,
            "rankic_mean": rankic_mean,
            "ic_std": ic_std,
            "rank_ic_std": rankic_std,
            "rankic_std": rankic_std,
            "ir": ir,
            "icir": ir,
            "stability": stability,
            "robustness": robustness,
            "phase_ic_std": phase_std,
            "phase_ic_min": phase_min,
            "quality_score": float(quality_score),
        }


class LogicEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(
        self,
        expr: Expression,
        features: ExpressionFeatures,
        values: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        ...


class RuleLogicEvaluator(LogicEvaluator):
    def __init__(self, config: RewardConfig):
        self.config = config

    def evaluate(
        self,
        expr: Expression,
        features: ExpressionFeatures,
        values: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        del expr
        length_penalty = min(1.0, features.node_count / max(1.0, self.config.max_logic_nodes))
        depth_penalty = min(1.0, features.depth / max(1.0, self.config.max_logic_depth))
        abnormal_ratio = 0.0
        low_variance_ratio = 0.0
        if values is not None:
            with torch.no_grad():
                bad = values.isnan() | values.isinf() | (values.abs() > self.config.max_abs_value)
                abnormal_ratio = _safe_item(bad.float().mean())
                finite = values[~bad]
                if finite.numel() <= 1 or float(finite.std(unbiased=False).item()) < self.config.low_variance_eps:
                    low_variance_ratio = 1.0
        penalty = (
            self.config.logic_length_weight * length_penalty
            + self.config.logic_depth_weight * depth_penalty
            + self.config.logic_abnormal_weight * abnormal_ratio
            + self.config.logic_low_variance_weight * low_variance_ratio
        )
        logic_score = float(max(0.0, min(1.0, 1.0 - penalty)))
        return {
            "logic_score": logic_score,
            "logic_penalty": float(penalty),
            "expr_length": float(features.node_count),
            "expr_token_count": float(features.node_count),
            "expr_depth": float(features.depth),
            "expr_complexity": float(features.node_count + features.depth),
            "abnormal_ratio": abnormal_ratio,
            "low_variance_ratio": low_variance_ratio,
        }


class LearnedLogicScorer(LogicEvaluator):
    def evaluate(
        self,
        expr: Expression,
        features: ExpressionFeatures,
        values: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError("Plug in a trained Tree-LSTM/GNN/GRU scorer here.")


class ExplicitSimilarityEvaluator:
    def evaluate(self, mutual_ics: Sequence[float]) -> Dict[str, float]:
        vals = [abs(float(v)) for v in mutual_ics if np.isfinite(v)]
        max_corr = max(vals) if vals else 0.0
        avg_corr = float(np.mean(vals)) if vals else 0.0
        return {
            "sim_ic": max_corr,
            "max_abs_mutual_ic": max_corr,
            "avg_abs_mutual_ic": avg_corr,
        }


class DualSimilarityEvaluator:
    def __init__(self, config: RewardConfig, adapter: Optional[ExpressionAdapter] = None):
        self.config = config
        self.adapter = adapter or ExpressionAdapter()

    def evaluate(self, features: ExpressionFeatures, pool_exprs: Sequence[Expression]) -> Dict[str, float]:
        struct_scores = []
        sem_scores = []
        for expr in pool_exprs:
            other = self.adapter.extract(expr)
            struct_scores.append(self._structure_similarity(features, other))
            sem_scores.append(self._semantic_similarity(features, other))

        sim_struct = float(np.mean(struct_scores)) if struct_scores else 0.0
        sim_sem = float(np.mean(sem_scores)) if sem_scores else 0.0
        max_struct = max(struct_scores) if struct_scores else 0.0
        max_sem = max(sem_scores) if sem_scores else 0.0
        sim_dual = (
            self.config.alpha_struct * sim_struct
            + (1.0 - self.config.alpha_struct) * sim_sem
        )
        return {
            "sim_struct": sim_struct,
            "sim_sem": sim_sem,
            "sim_dual": float(sim_dual),
            "max_structure_similarity": max_struct,
            "avg_structure_similarity": sim_struct,
            "max_semantic_similarity": max_sem,
            "avg_semantic_similarity": sim_sem,
        }

    def _structure_similarity(self, lhs: ExpressionFeatures, rhs: ExpressionFeatures) -> float:
        token_sim = jaccard(lhs.operator_tokens, rhs.operator_tokens)
        depth_gap = abs(lhs.depth - rhs.depth) / max(lhs.depth, rhs.depth, 1)
        size_gap = abs(lhs.node_count - rhs.node_count) / max(lhs.node_count, rhs.node_count, 1)
        return float(max(0.0, min(1.0, 0.6 * token_sim + 0.2 * (1 - depth_gap) + 0.2 * (1 - size_gap))))

    def _semantic_similarity(self, lhs: ExpressionFeatures, rhs: ExpressionFeatures) -> float:
        return jaccard(lhs.tokens, rhs.tokens)


def _safe_item(value: torch.Tensor) -> float:
    result = float(value.item())
    return result if np.isfinite(result) else 0.0


def _phase_means(values: torch.Tensor, n_phases: int):
    n = int(values.shape[0])
    if n == 0:
        return []
    n_phases = max(1, min(int(n_phases), n))
    return [_safe_item(chunk.mean()) for chunk in torch.chunk(values, n_phases) if chunk.numel() > 0]


def _align_tensors(values: torch.Tensor, target: torch.Tensor):
    if values.ndim != 2 or target.ndim != 2:
        return values.reshape(0, 0), target.reshape(0, 0)
    days = min(values.shape[0], target.shape[0])
    stocks = min(values.shape[1], target.shape[1])
    if days <= 0 or stocks <= 0:
        return values[:0, :0], target[:0, :0]
    return values[:days, :stocks], target[:days, :stocks]


def quality_score_torch(
    values: torch.Tensor,
    target: torch.Tensor,
    config: RewardConfig,
) -> torch.Tensor:
    """Differentiable R4 quality term for ensemble weight optimization."""
    values, target = _align_tensors(values, target)
    if values.numel() == 0 or target.numel() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)

    ics = batch_pearsonr(values, target)
    ic_mean = ics.mean()
    ic_std = ics.std(unbiased=False)
    ir = ic_mean / (ic_std + config.eps)

    n_phases = max(1, min(int(config.n_phases), int(ics.shape[0])))
    phase_means = torch.stack([chunk.mean() for chunk in torch.chunk(ics, n_phases) if chunk.numel() > 0])
    if phase_means.numel() > 1:
        phase_std = phase_means.std(unbiased=False)
    else:
        phase_std = torch.zeros((), device=values.device, dtype=values.dtype)
    phase_min = phase_means.min() if phase_means.numel() > 0 else torch.zeros((), device=values.device, dtype=values.dtype)

    stability = -ic_std - config.phase_std_weight * phase_std
    robustness = phase_min
    rankic_mean = batch_spearmanr(values, target).mean()
    # Spearman is non-differentiable; route RankIC weight through Pearson mean.
    rankic_proxy = ic_mean + (rankic_mean - ic_mean).detach()

    return (
        config.w_ic * ic_mean
        + config.w_rankic * rankic_proxy
        + config.w_ir * ir
        + config.w_stability * stability
        + config.w_robustness * robustness
    )


def _zero_quality() -> Dict[str, float]:
    return {
        "ic": 0.0,
        "rank_ic": 0.0,
        "rankic_mean": 0.0,
        "ic_std": 0.0,
        "rank_ic_std": 0.0,
        "rankic_std": 0.0,
        "ir": 0.0,
        "icir": 0.0,
        "stability": 0.0,
        "robustness": 0.0,
        "phase_ic_std": 0.0,
        "phase_ic_min": 0.0,
        "quality_score": 0.0,
    }