from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    mode: str = "original"

    # Positive quality Q(f)
    w_ic: float = 1.0
    w_rankic: float = 0.0
    w_ir: float = 0.0
    w_stability: float = 0.0
    w_robustness: float = 0.0
    phase_std_weight: float = 1.0

    # Logic score P_logic(f), scaled into [0, 1]
    eta_logic: float = 0.0
    logic_length_weight: float = 0.10
    logic_depth_weight: float = 0.10
    logic_abnormal_weight: float = 0.50
    logic_low_variance_weight: float = 0.30
    max_logic_nodes: float = 30.0
    max_logic_depth: float = 8.0

    # Redundancy penalties
    lambda_ic: float = 0.0
    lambda_dual: float = 0.0
    alpha_struct: float = 0.5

    # Numerical guards
    eps: float = 1e-8
    low_variance_eps: float = 1e-6
    max_abs_value: float = 1e6
    n_phases: int = 4
    use_pool_metrics: bool = True
    reward_clip_min: Optional[float] = None
    reward_clip_max: Optional[float] = None


def default_config(mode: str) -> RewardConfig:
    mode = mode.lower()
    if mode in ("v0", "original"):
        return RewardConfig(mode="original")
    if mode in ("rank", "rankic"):
        # Extra ablation, not the thesis R1.
        return RewardConfig(mode="rank", w_ic=0.5, w_rankic=0.5)
    if mode in ("v1", "stability"):
        return RewardConfig(
            mode="stability",
            w_ic=1.0,
            w_ir=0.05,
            w_stability=0.20,
            w_robustness=0.05,
        )
    if mode in ("v2", "logic"):
        return RewardConfig(
            mode="logic",
            w_ic=1.0,
            w_ir=0.05,
            w_stability=0.20,
            w_robustness=0.05,
            eta_logic=0.05,
        )
    if mode in ("v3", "redundancy"):
        return RewardConfig(
            mode="redundancy",
            w_ic=1.0,
            w_ir=0.05,
            w_stability=0.20,
            w_robustness=0.05,
            eta_logic=0.05,
            lambda_ic=0.15,
            lambda_dual=0.05,
            alpha_struct=0.6,
        )
    if mode in ("v4", "multi"):
        return RewardConfig(
            mode="multi",
            w_ic=0.70,
            w_rankic=0.20,
            w_ir=0.05,
            w_stability=0.15,
            w_robustness=0.05,
            eta_logic=0.05,
            lambda_ic=0.15,
            lambda_dual=0.08,
            alpha_struct=0.6,
        )
    raise ValueError(f"Unknown reward mode: {mode}")
