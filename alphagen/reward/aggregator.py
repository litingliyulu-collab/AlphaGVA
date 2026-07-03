from dataclasses import dataclass
from typing import Dict, Optional

from alphagen.reward.config import RewardConfig


@dataclass
class AggregatedReward:
    reward: float
    components: Dict[str, float]
    diagnostics: Dict[str, float]


class RewardAggregator:
    def __init__(self, config: RewardConfig):
        self.config = config

    def aggregate(
        self,
        quality: Dict[str, float],
        logic: Dict[str, float],
        explicit: Dict[str, float],
        dual: Dict[str, float],
    ) -> AggregatedReward:
        cfg = self.config
        quality_score = float(quality.get("quality_score", 0.0))
        logic_score = float(logic.get("logic_score", 0.0))
        sim_ic = float(explicit.get("sim_ic", 0.0))
        sim_dual = float(dual.get("sim_dual", 0.0))

        components = {
            "quality_score": quality_score,
            "logic_score": cfg.eta_logic * logic_score,
            "explicit_redundancy": -cfg.lambda_ic * sim_ic,
            "implicit_redundancy": -cfg.lambda_dual * sim_dual,
        }
        reward = float(sum(components.values()))
        reward = _clip(reward, cfg.reward_clip_min, cfg.reward_clip_max)

        diagnostics: Dict[str, float] = {}
        diagnostics.update(quality)
        diagnostics.update(logic)
        diagnostics.update(explicit)
        diagnostics.update(dual)
        diagnostics["penalty"] = cfg.lambda_ic * sim_ic + cfg.lambda_dual * sim_dual
        return AggregatedReward(reward=reward, components=components, diagnostics=diagnostics)


def _clip(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is not None:
        value = max(float(low), value)
    if high is not None:
        value = min(float(high), value)
    return float(value)
