from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import math
import torch

from alphagen.data.expression import Expression
from alphagen.reward.aggregator import RewardAggregator
from alphagen.reward.config import RewardConfig, default_config
from alphagen.reward.evaluators import (
    DualSimilarityEvaluator,
    ExplicitSimilarityEvaluator,
    PositiveEvaluator,
    RuleLogicEvaluator,
)
from alphagen.reward.expression_adapter import ExpressionAdapter


@dataclass
class RewardResult:
    total: float
    components: Dict[str, float]
    diagnostics: Dict[str, float]

    def flat_metrics(self, prefix: str = "reward/") -> Dict[str, float]:
        values = {f"{prefix}total": self.total}
        values.update({f"{prefix}{k}": v for k, v in self.components.items()})
        values.update({f"{prefix}{k}": v for k, v in self.diagnostics.items()})
        return values


@dataclass
class RewardContext:
    expr: Expression
    calculator: Any
    pool_exprs: Sequence[Expression]
    pool_weights: Sequence[float]
    old_pool_exprs: Sequence[Expression]
    single_ic: float
    mutual_ics: Sequence[float]
    pool_ic: float
    original_objective: float
    candidate_value: Optional[torch.Tensor] = None
    pool_value: Optional[torch.Tensor] = None


class RewardEvaluator:
    """Online terminal reward evaluator aligned with the thesis R0-R4 design."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.adapter = ExpressionAdapter()
        self.positive = PositiveEvaluator(self.config)
        self.logic = RuleLogicEvaluator(self.config)
        self.explicit = ExplicitSimilarityEvaluator()
        self.dual = DualSimilarityEvaluator(self.config, self.adapter)
        self.aggregator = RewardAggregator(self.config)

    @classmethod
    def from_mode(cls, mode: str) -> "RewardEvaluator":
        return cls(default_config(mode))

    def evaluate(self, ctx: RewardContext) -> RewardResult:
        mode = self.config.mode.lower()
        if mode == "original":
            return RewardResult(
                total=float(ctx.original_objective),
                components={"original_objective": float(ctx.original_objective)},
                diagnostics={"pool_ic": float(ctx.pool_ic), "single_ic": float(ctx.single_ic)},
            )

        values = self._reward_values(ctx)
        target = ctx.calculator.target
        quality = self.positive.evaluate(values, target)
        if mode == "rank":
            quality = dict(quality)
            quality["quality_score"] = 0.5 * quality["ic"] + 0.5 * quality["rank_ic"]
            logic = _zero_logic()
            explicit = _zero_explicit()
            dual = _zero_dual()
        elif mode == "stability":
            logic = _zero_logic()
            explicit = _zero_explicit()
            dual = _zero_dual()
        elif mode == "logic":
            features = self.adapter.extract(ctx.expr)
            logic_values = ctx.candidate_value if ctx.candidate_value is not None else values
            logic = self.logic.evaluate(ctx.expr, features, logic_values)
            explicit = _zero_explicit()
            dual = _zero_dual()
        elif mode in ("redundancy", "multi"):
            features = self.adapter.extract(ctx.expr)
            logic_values = ctx.candidate_value if ctx.candidate_value is not None else values
            logic = self.logic.evaluate(ctx.expr, features, logic_values)
            explicit = self.explicit.evaluate(ctx.mutual_ics)
            dual = self.dual.evaluate(features, ctx.old_pool_exprs)
        else:
            raise ValueError(f"Unknown reward mode: {mode}")

        aggregated = self.aggregator.aggregate(quality, logic, explicit, dual)
        diagnostics = dict(aggregated.diagnostics)
        diagnostics.update({"pool_ic": float(ctx.pool_ic), "single_ic": float(ctx.single_ic)})
        total = aggregated.reward
        if total is None or not math.isfinite(float(total)):
            total = 0.0
        else:
            total = float(total)
        return RewardResult(
            total=total,
            components=aggregated.components,
            diagnostics=diagnostics,
        )

    def _reward_values(self, ctx: RewardContext) -> torch.Tensor:
        if self.config.use_pool_metrics and ctx.pool_value is not None:
            return ctx.pool_value
        if self.config.use_pool_metrics and len(ctx.pool_exprs) > 0:
            return ctx.calculator.make_ensemble_alpha(ctx.pool_exprs, ctx.pool_weights)
        if ctx.candidate_value is not None:
            return ctx.candidate_value
        return ctx.calculator.evaluate_alpha(ctx.expr)


def _zero_logic() -> Dict[str, float]:
    return {
        "logic_score": 0.0,
        "logic_penalty": 0.0,
        "expr_length": 0.0,
        "expr_depth": 0.0,
        "abnormal_ratio": 0.0,
        "low_variance_ratio": 0.0,
    }


def _zero_explicit() -> Dict[str, float]:
    return {"sim_ic": 0.0, "max_abs_mutual_ic": 0.0, "avg_abs_mutual_ic": 0.0}


def _zero_dual() -> Dict[str, float]:
    return {
        "sim_struct": 0.0,
        "sim_sem": 0.0,
        "sim_dual": 0.0,
        "max_structure_similarity": 0.0,
        "avg_structure_similarity": 0.0,
        "max_semantic_similarity": 0.0,
        "avg_semantic_similarity": 0.0,
    }
