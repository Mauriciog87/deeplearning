from dataclasses import dataclass
from enum import Enum

import numpy as np

from .settlement import validate_number


FAIR_PROBABILITY = 1 / 37


class Availability(str, Enum):
    READY = 'ready'
    UNTRAINED = 'untrained'
    INSUFFICIENT_DATA = 'insufficient_data'
    UNAVAILABLE = 'unavailable'
    FAILED = 'failed'


@dataclass(frozen=True)
class ModelStatus:
    state: Availability
    reason: str = ''


@dataclass(frozen=True)
class PolicyDecision:
    policy_id: str
    action: int | None
    stake: float | None
    status: ModelStatus = ModelStatus(Availability.READY)


def validate_probabilities(values):
    if isinstance(values, dict):
        if set(values) != set(range(37)):
            raise ValueError('Probability distribution must include exactly numbers 0 through 36')
        values = [values[n] for n in range(37)]
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (37,) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError('Expected 37 finite, nonnegative probabilities')
    total = float(probabilities.sum())
    if not np.isclose(total, 1.0, rtol=0, atol=1e-6):
        raise ValueError('Probabilities must sum to one')
    return probabilities / total


def frequency_probabilities(history, alpha=1.0):
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('Smoothing alpha must be positive and finite')
    numbers = [validate_number(n) for n in history]
    counts = np.bincount(numbers, minlength=37)
    return (counts + alpha) / (len(numbers) + 37 * alpha)


def ranked_numbers(probabilities):
    vector = validate_probabilities(probabilities)
    return [(int(n), float(vector[n])) for n in np.argsort(-vector, kind='stable')]
