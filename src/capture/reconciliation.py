from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from src.settlement import validate_number


@dataclass
class CaptureObservation:
    numbers: list[int]
    confidence: float
    reason: str = ''
    event_id: str = field(default_factory=lambda: str(uuid4()))
    observed_at: datetime = field(default_factory=datetime.now)
    status: str = 'pending'


def reconcile_history(known, observed_newest_first, min_overlap=2):
    if min_overlap < 2:
        raise ValueError('Reconciliation requires at least two matching observations')
    known = [validate_number(number) for number in known]
    observed = list(reversed([validate_number(number) for number in observed_newest_first]))
    candidates = {tuple(observed[length:]) for length in range(min_overlap, min(len(known), len(observed)) + 1)
                  if known[-length:] == observed[:length]}
    if not candidates:
        return None, 'No sequence overlap of at least two numbers'
    if len(candidates) > 1:
        return None, 'Repeated sequences allow different alignments'
    return list(candidates.pop()), ''
