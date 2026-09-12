from dataclasses import dataclass
from math import isfinite

import numpy as np

from src.probabilities import validate_probabilities
from src.settlement import action_bet, settle_action, validate_number, validate_stake


@dataclass(frozen=True)
class ExpectedValueDecision:
    action: int
    stake: float
    expected_profit: float
    lower_expected_profit: float | None
    action_values: tuple[float, ...]
    reason: str


def counterfactual_profits(actual, unit_stake=1.0):
    actual = validate_number(actual)
    validate_stake(unit_stake, unit_stake)
    return tuple(settle_action(actual, action, unit_stake, unit_stake).net_profit for action in range(47))


_UNIT_REWARDS = np.asarray([counterfactual_profits(number) for number in range(37)])


def expected_action_values(probabilities, unit_stake=1.0):
    vector = validate_probabilities(probabilities)
    validate_stake(unit_stake, unit_stake)
    return tuple(float(value) for value in vector @ _UNIT_REWARDS * unit_stake)


def choose_expected_value_action(probabilities, balance, unit_stake=1.0, *, probability_bounds=None):
    validate_stake(0, balance)
    if not isfinite(unit_stake) or unit_stake <= 0:
        raise ValueError('Unit stake must be positive and finite')
    values = expected_action_values(probabilities, unit_stake)
    if balance < unit_stake:
        return ExpectedValueDecision(46, 0, 0, 0 if probability_bounds is not None else None, values, 'Insufficient bankroll')
    scores = list(values)
    if probability_bounds is not None:
        for action in range(46):
            numbers, payout = action_bet(action)
            lower, upper = probability_bounds(numbers)
            if not isfinite(lower) or not isfinite(upper) or not 0 <= lower <= upper <= 1:
                raise ValueError('Probability bounds must be finite and ordered within [0,1]')
            scores[action] = unit_stake * ((payout + 1) * lower - 1)
    action = max(range(47), key=lambda candidate: (scores[candidate], candidate == 46, -candidate))
    if scores[action] <= unit_stake * 1e-12:
        action = 46
    return ExpectedValueDecision(
        action, 0 if action == 46 else unit_stake, values[action],
        scores[action] if probability_bounds is not None else None, values,
        'Maximum lower expected profit under the supplied simultaneous probability bounds'
        if probability_bounds is not None else 'Maximum expected profit under the supplied forecast; no uncertainty guarantee',
    )
