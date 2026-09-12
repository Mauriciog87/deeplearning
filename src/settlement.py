from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Iterable


def validate_number(number: int) -> int:
    if isinstance(number, bool) or not isinstance(number, Integral) or not 0 <= number <= 36:
        raise ValueError(f'Invalid roulette number: {number!r}')
    return int(number)


@dataclass(frozen=True)
class Settlement:
    total_stake: float
    net_profit: float
    balance_before: float
    balance_after: float
    won: bool | None


def validate_stake(stake: float, balance: float) -> None:
    if not isfinite(balance) or balance < 0:
        raise ValueError('Balance must be finite and nonnegative')
    if not isfinite(stake) or stake < 0 or stake > balance:
        raise ValueError('Stake must be finite, nonnegative and affordable')


def action_bet(action: int):
    if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action <= 46:
        raise ValueError('Action must be an integer between 0 and 46')
    if action <= 36:
        return (int(action),), 35
    red = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    outside = {
        37: red, 38: set(range(1, 37)) - red,
        39: set(range(1, 37, 2)), 40: set(range(2, 37, 2)),
        41: set(range(1, 19)), 42: set(range(19, 37)),
        43: set(range(1, 13)), 44: set(range(13, 25)), 45: set(range(25, 37)),
        46: set(),
    }
    return tuple(sorted(outside[action])), 0 if action == 46 else 2 if action >= 43 else 1


def settle_action(actual: int, action: int, stake: float, balance: float) -> Settlement:
    numbers, payout = action_bet(action)
    return settle_bet(actual, numbers, stake, balance, payout=payout, separate_straights=False)


def settle_bet(actual: int, numbers: Iterable[int], unit_stake: float,
               balance: float, *, payout: float = 35,
               separate_straights: bool = True) -> Settlement:
    actual = validate_number(actual)
    selected = tuple(validate_number(n) for n in numbers)
    if len(set(selected)) != len(selected):
        raise ValueError('Bet numbers must be unique')
    if not isfinite(unit_stake) or unit_stake < 0:
        raise ValueError('Unit stake must be finite and nonnegative')
    if not isfinite(payout) or payout < 0:
        raise ValueError('Payout must be finite and nonnegative')
    total = unit_stake * (len(selected) if separate_straights else bool(selected))
    validate_stake(total, balance)
    if not total:
        return Settlement(0.0, 0.0, balance, balance, None)
    won = actual in selected
    returned = unit_stake * (36 if separate_straights else payout + 1) if won else 0
    profit = returned - total
    return Settlement(total, profit, balance, balance + profit, won)
