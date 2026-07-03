import math
import types

import pytest
import torch

from alphagen.data.expression import Add, Feature
from alphagen.reward import RewardContext, RewardEvaluator, RuleLogicEvaluator, RewardConfig, ExpressionAdapter
from alphagen_qlib.stock_data import FeatureType


class DummyCalculator:
    def __init__(self, values=None, target=None):
        self.values = values if values is not None else torch.randn(12, 5)
        self._target = target if target is not None else torch.randn(12, 5)

    @property
    def target(self):
        return self._target

    def evaluate_alpha(self, _):
        return self.values

    def make_ensemble_alpha(self, _exprs, _weights):
        return self.values


def make_context(values=None, target=None, old_pool=None, mutual_ics=None):
    expr = Feature(FeatureType.CLOSE)
    return RewardContext(
        expr=expr,
        calculator=DummyCalculator(values, target),
        pool_exprs=[expr],
        pool_weights=[1.0],
        old_pool_exprs=old_pool or [],
        single_ic=0.1,
        mutual_ics=mutual_ics or [],
        pool_ic=0.1,
        original_objective=0.1,
    )


def test_normal_reward_has_details():
    result = RewardEvaluator.from_mode("multi").evaluate(make_context())
    metrics = result.flat_metrics()
    assert math.isfinite(result.total)
    assert "reward/quality_score" in metrics
    assert "reward/logic_score" in metrics
    assert "reward/sim_ic" in metrics
    assert "reward/sim_dual" in metrics


def test_empty_pool_similarity_is_zero():
    result = RewardEvaluator.from_mode("redundancy").evaluate(make_context(old_pool=[], mutual_ics=[]))
    assert result.diagnostics["sim_ic"] == 0.0
    assert result.diagnostics["sim_dual"] == 0.0


@pytest.mark.parametrize(
    "values,target",
    [
        (torch.ones(12, 5), torch.randn(12, 5)),
        (torch.full((12, 5), torch.nan), torch.randn(12, 5)),
        (torch.randn(7, 3), torch.randn(12, 5)),
    ],
)
def test_bad_series_does_not_crash(values, target):
    result = RewardEvaluator.from_mode("stability").evaluate(make_context(values, target))
    assert math.isfinite(result.total)


def test_high_similarity_gets_stronger_penalty():
    low = RewardEvaluator.from_mode("redundancy").evaluate(make_context(mutual_ics=[0.1]))
    high = RewardEvaluator.from_mode("redundancy").evaluate(make_context(mutual_ics=[0.95]))
    assert high.diagnostics["sim_ic"] > low.diagnostics["sim_ic"]
    assert high.total < low.total


def test_logic_score_in_unit_interval():
    evaluator = RuleLogicEvaluator(RewardConfig(mode="logic"))
    expr = Add(Feature(FeatureType.CLOSE), Feature(FeatureType.OPEN))
    features = ExpressionAdapter().extract(expr)
    score = evaluator.evaluate(expr, features, torch.randn(12, 5))["logic_score"]
    assert 0.0 <= score <= 1.0


def test_weight_changes_final_reward():
    ctx = make_context()
    cfg_a = RewardConfig(mode="multi", w_ic=1.0, eta_logic=0.0)
    cfg_b = RewardConfig(mode="multi", w_ic=1.0, eta_logic=1.0)
    a = RewardEvaluator(cfg_a).evaluate(ctx).total
    b = RewardEvaluator(cfg_b).evaluate(ctx).total
    assert a != b
