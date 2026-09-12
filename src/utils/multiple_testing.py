from math import isfinite
from typing import List, Optional, Tuple


def benjamini_hochberg(
    p_values: List[Optional[float]],
    alpha: float = 0.05,
) -> Tuple[List[Optional[float]], List[bool]]:
    indexed = [
        (index, float(p_value))
        for index, p_value in enumerate(p_values)
        if p_value is not None and isfinite(float(p_value))
    ]
    q_values: List[Optional[float]] = [None for _ in p_values]
    rejected = [False for _ in p_values]
    if not indexed:
        return q_values, rejected

    count = len(indexed)
    ordered = sorted(indexed, key=lambda item: item[1])
    running_min = 1.0

    for rank, (index, p_value) in reversed(list(enumerate(ordered, start=1))):
        q_value = min(running_min, max(0.0, min(1.0, p_value * count / rank)))
        running_min = q_value
        q_values[index] = q_value

    for index, q_value in enumerate(q_values):
        rejected[index] = q_value is not None and q_value <= alpha

    return q_values, rejected


def false_discovery_control(p_values, alpha=0.05, method='by'):
    if method not in ('bh', 'by') or not 0 < alpha < 1:
        raise ValueError('FDR method must be bh or by and alpha must be between zero and one')
    if any(value is not None and (not isfinite(value) or not 0 <= value <= 1) for value in p_values):
        raise ValueError('P-values must be finite and between zero and one')
    adjusted, _ = benjamini_hochberg(p_values, alpha)
    if method == 'by':
        count = sum(value is not None for value in p_values)
        factor = sum(1 / rank for rank in range(1, count + 1))
        adjusted = [min(1.0, value * factor) if value is not None else None for value in adjusted]
    return adjusted, [value is not None and value <= alpha for value in adjusted]
