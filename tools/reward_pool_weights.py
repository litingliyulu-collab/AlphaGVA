"""Fit R4 (multi) ensemble weights on train data for backtest comparison."""
from __future__ import annotations

import sys
from itertools import count
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_PATCH = _ROOT / "_remote_reward_patch"
if _PATCH.exists() and str(_PATCH) not in sys.path:
    sys.path.insert(0, str(_PATCH))

from alphagen.data.expression import Expression, Feature, Ref  # noqa: E402
from alphagen.reward.config import default_config  # noqa: E402
from alphagen.reward.evaluators import quality_score_torch  # noqa: E402
from alphagen.utils.pytorch_utils import normalize_by_day  # noqa: E402
from alphagen_qlib.calculator import QLibStockDataCalculator  # noqa: E402
from alphagen_qlib.stock_data import FeatureType, StockData  # noqa: E402


def build_target_expr() -> Expression:
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


def stack_alpha_values(
    exprs: Sequence[Expression],
    data: StockData,
) -> torch.Tensor:
    with torch.no_grad():
        values = [normalize_by_day(expr.evaluate(data)) for expr in exprs]
    return torch.stack(values, dim=0)


def optimize_multi_weights(
    exprs: Sequence[Expression],
    train_data: StockData,
    init_weights: Sequence[float],
    *,
    l1_alpha: float = 5e-3,
    lr: float = 5e-4,
    max_steps: int = 2000,
    tolerance: int = 200,
    device: torch.device | None = None,
) -> np.ndarray:
    if not exprs:
        return np.array([], dtype=np.float64)
    if device is None:
        device = train_data.device

    target_expr = build_target_expr()
    calculator = QLibStockDataCalculator(train_data, target_expr)
    alpha_values = stack_alpha_values(exprs, train_data).to(device)
    target = calculator.target.to(device)
    config = default_config("multi")

    init = np.asarray(init_weights, dtype=np.float64)
    if init.shape[0] != len(exprs):
        init = np.full(len(exprs), 1.0 / len(exprs))

    weights = torch.tensor(init, device=device, dtype=alpha_values.dtype, requires_grad=True)
    optim = torch.optim.Adam([weights], lr=lr)

    min_loss = float("inf")
    best_weights = weights.detach().clone()
    tol_count = 0
    for step in count():
        weighted = (weights[:, None, None] * alpha_values).sum(dim=0)
        obj = quality_score_torch(weighted, target, config)
        loss = l1_alpha * torch.norm(weights, p=1) - obj
        curr_loss = float(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        if min_loss - curr_loss > 1e-6:
            tol_count = 0
        else:
            tol_count += 1
        if curr_loss < min_loss:
            best_weights = weights.detach().clone()
            min_loss = curr_loss
        if tol_count >= tolerance or step >= max_steps:
            break

    return best_weights.cpu().numpy()
