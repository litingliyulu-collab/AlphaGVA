from dataclasses import dataclass
from typing import List

from alphagen.data.expression import Expression, Operator


@dataclass(frozen=True)
class ExpressionFeatures:
    tokens: List[str]
    operator_tokens: List[str]
    depth: int
    node_count: int
    expression: str


class ExpressionAdapter:
    """Lightweight adapter around the project's existing Expression tree."""

    def extract(self, expr: Expression) -> ExpressionFeatures:
        tokens: List[str] = []
        operators: List[str] = []

        def walk(node: Expression, depth: int) -> int:
            name = type(node).__name__
            tokens.append(name)
            if isinstance(node, Operator):
                operators.append(name)
                child_depths = [walk(child, depth + 1) for child in node.operands]
                return max(child_depths, default=depth)
            tokens.append(str(node))
            return depth

        max_depth = walk(expr, 1)
        return ExpressionFeatures(
            tokens=tokens,
            operator_tokens=operators,
            depth=max_depth,
            node_count=len(tokens),
            expression=str(expr),
        )


def jaccard(lhs: List[str], rhs: List[str]) -> float:
    left, right = set(lhs), set(rhs)
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))
