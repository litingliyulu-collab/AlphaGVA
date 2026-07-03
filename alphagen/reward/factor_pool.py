from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from alphagen.data.expression import Expression
from alphagen.reward.expression_adapter import ExpressionAdapter, ExpressionFeatures


@dataclass
class RewardFactorRecord:
    expression: Expression
    score: float = 0.0
    values: Optional[torch.Tensor] = None
    ic_series: Optional[torch.Tensor] = None
    features: Optional[ExpressionFeatures] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RewardFactorPool:
    """Reward-side lightweight pool view. It does not replace AlphaGen's pool."""

    def __init__(self, adapter: Optional[ExpressionAdapter] = None):
        self.adapter = adapter or ExpressionAdapter()
        self.records: List[RewardFactorRecord] = []

    def add(self, record: RewardFactorRecord) -> None:
        if record.features is None:
            record.features = self.adapter.extract(record.expression)
        self.records.append(record)

    def remove(self, index: int) -> None:
        del self.records[index]

    def top_k(self, k: int) -> List[RewardFactorRecord]:
        return sorted(self.records, key=lambda item: item.score, reverse=True)[:k]

    def periodic_filter(self) -> None:
        """Reserved for offline GRS or significance filtering."""
        return None
