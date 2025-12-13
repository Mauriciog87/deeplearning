from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import Counter
from datetime import datetime


EUROPEAN_HOUSE_EDGE = 2.7  # 2.7% house edge
PAYOUT_STRAIGHT = 35  # 35:1 for straight up bet


class BetType(Enum):
    STRAIGHT_UP = "straight_up"  # Single number, 35:1
    SPLIT = "split"  # Two numbers, 17:1
    STREET = "street"  # Three numbers, 11:1
    CORNER = "corner"  # Four numbers, 8:1
    LINE = "line"  # Six numbers, 5:1
    DOZEN = "dozen"  # 12 numbers, 2:1
    COLUMN = "column"  # 12 numbers, 2:1
    RED_BLACK = "red_black"  # 18 numbers, 1:1
    ODD_EVEN = "odd_even"  # 18 numbers, 1:1
    HIGH_LOW = "high_low"  # 18 numbers, 1:1


PAYOUTS = {
    BetType.STRAIGHT_UP: 35,
    BetType.SPLIT: 17,
    BetType.STREET: 11,
    BetType.CORNER: 8,
    BetType.LINE: 5,
    BetType.DOZEN: 2,
    BetType.COLUMN: 2,
    BetType.RED_BLACK: 1,
    BetType.ODD_EVEN: 1,
    BetType.HIGH_LOW: 1,
}


@dataclass
class BetResult:
    spin_number: int
    actual_outcome: int
    bet_numbers: List[int]
    bet_type: BetType
    bet_amount: float
    won: bool
    payout: float
    profit: float
    balance_after: float


@dataclass 
class BacktestResult:
    initial_balance: float
    final_balance: float
    total_profit: float
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    sharpe_ratio: float
    max_consecutive_losses: int
    max_consecutive_wins: int
    consecutive_loss_histogram: Dict[int, int]
    equity_curve: List[float]
    bets: List[BetResult] = field(default_factory=list)
    
    @property
    def return_pct(self) -> float:
        return ((self.final_balance - self.initial_balance) / self.initial_balance) * 100
    
    @property
    def profit_factor(self) -> float:
        total_wins = sum(b.profit for b in self.bets if b.profit > 0)
        total_losses = abs(sum(b.profit for b in self.bets if b.profit < 0))
        return total_wins / total_losses if total_losses > 0 else float('inf')


@dataclass
class WalkForwardResult:
    training_periods: int
    testing_periods: int
    in_sample_return: float
    out_of_sample_return: float
    robustness_ratio: float
    period_results: List[Dict[str, Any]]
    aggregate_backtest: BacktestResult


def compute_max_drawdown(equity_curve: List[float]) -> Tuple[float, float]:
    """
    Calculate Maximum Drawdown (MDD) from equity curve.
    
    MDD = (Peak - Trough) / Peak
    
    Returns both absolute and percentage drawdown.
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = peak - value
        drawdown_pct = drawdown / peak if peak > 0 else 0
        
        if drawdown > max_dd:
            max_dd = drawdown
        if drawdown_pct > max_dd_pct:
            max_dd_pct = drawdown_pct
    
    return max_dd, max_dd_pct


def compute_calmar_ratio(
    equity_curve: List[float],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio = Annualized Return / Max Drawdown
    
    Higher is better - shows return per unit of drawdown risk.
    From Salirrosas (2016) paper metrics.
    """
    if len(equity_curve) < 2:
        return 0.0
    
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    periods = len(equity_curve)
    annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1
    
    _, max_dd_pct = compute_max_drawdown(equity_curve)
    
    if max_dd_pct == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return annualized_return / max_dd_pct


def compute_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe Ratio = (Mean Return - Risk Free) / Std Dev
    
    Measures risk-adjusted return.
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_return


def compute_consecutive_losses(results: List[bool]) -> Dict[str, Any]:
    """
    Analyze consecutive loss streaks.
    
    Returns histogram of streak lengths and statistics.
    """
    if not results:
        return {"max_losses": 0, "max_wins": 0, "histogram": {}}
    
    loss_streaks = []
    win_streaks = []
    current_streak = 0
    current_is_loss = not results[0]
    
    for won in results:
        if won:
            if current_is_loss and current_streak > 0:
                loss_streaks.append(current_streak)
                current_streak = 0
            current_is_loss = False
            current_streak += 1
        else:
            if not current_is_loss and current_streak > 0:
                win_streaks.append(current_streak)
                current_streak = 0
            current_is_loss = True
            current_streak += 1
    
    if current_is_loss and current_streak > 0:
        loss_streaks.append(current_streak)
    elif not current_is_loss and current_streak > 0:
        win_streaks.append(current_streak)
    
    loss_histogram = Counter(loss_streaks)
    
    return {
        "max_consecutive_losses": max(loss_streaks) if loss_streaks else 0,
        "max_consecutive_wins": max(win_streaks) if win_streaks else 0,
        "avg_loss_streak": np.mean(loss_streaks) if loss_streaks else 0,
        "loss_histogram": dict(loss_histogram),
        "total_loss_streaks": len(loss_streaks)
    }


def kelly_criterion(
    win_probability: float,
    payout: float,
    fraction: float = 0.25
) -> float:
    """
    Calculate Kelly Criterion bet size.
    
    f* = (bp - q) / b
    
    Where:
    - b: odds (payout)
    - p: probability of winning
    - q: probability of losing (1 - p)
    
    Note: Salirrosas (2016) found flat betting outperformed Kelly,
    so we use fractional Kelly (default 25%) for safety.
    """
    q = 1 - win_probability
    kelly = (payout * win_probability - q) / payout
    
    kelly = max(0, kelly)
    
    return kelly * fraction


def flat_bet_backtest(
    numbers: List[int],
    bet_numbers: List[int],
    bet_amount: float = 1.0,
    initial_balance: float = 1000.0,
    bet_type: BetType = BetType.STRAIGHT_UP
) -> BacktestResult:
    """
    Backtest a flat betting strategy on historical data.
    
    From Salirrosas (2016): Flat betting consistently outperformed
    Kelly Criterion due to lower variance.
    """
    balance = initial_balance
    equity_curve = [balance]
    bets = []
    results = []
    
    payout = PAYOUTS[bet_type]
    
    for i, actual in enumerate(numbers):
        if balance < bet_amount:
            break
        
        won = actual in bet_numbers
        results.append(won)
        
        if won:
            profit = bet_amount * payout
        else:
            profit = -bet_amount
        
        balance += profit
        equity_curve.append(balance)
        
        bets.append(BetResult(
            spin_number=i + 1,
            actual_outcome=actual,
            bet_numbers=bet_numbers,
            bet_type=bet_type,
            bet_amount=bet_amount,
            won=won,
            payout=payout if won else 0,
            profit=profit,
            balance_after=balance
        ))
    
    if not bets:
        return BacktestResult(
            initial_balance=initial_balance,
            final_balance=balance,
            total_profit=0,
            total_bets=0,
            wins=0,
            losses=0,
            win_rate=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            calmar_ratio=0,
            sharpe_ratio=0,
            max_consecutive_losses=0,
            max_consecutive_wins=0,
            consecutive_loss_histogram={},
            equity_curve=equity_curve,
            bets=bets
        )
    
    wins = sum(1 for b in bets if b.won)
    losses = len(bets) - wins
    win_rate = wins / len(bets) if bets else 0
    
    max_dd, max_dd_pct = compute_max_drawdown(equity_curve)
    
    returns = [b.profit / initial_balance for b in bets]
    sharpe = compute_sharpe_ratio(returns)
    calmar = compute_calmar_ratio(equity_curve)
    
    streak_analysis = compute_consecutive_losses(results)
    
    return BacktestResult(
        initial_balance=initial_balance,
        final_balance=balance,
        total_profit=balance - initial_balance,
        total_bets=len(bets),
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        calmar_ratio=calmar,
        sharpe_ratio=sharpe,
        max_consecutive_losses=streak_analysis["max_consecutive_losses"],
        max_consecutive_wins=streak_analysis["max_consecutive_wins"],
        consecutive_loss_histogram=streak_analysis["loss_histogram"],
        equity_curve=equity_curve,
        bets=bets
    )


def walk_forward_optimization(
    numbers: List[int],
    strategy_fn: Callable[[List[int]], List[int]],
    training_window: int = 500,
    testing_window: int = 100,
    bet_amount: float = 1.0,
    initial_balance: float = 1000.0
) -> WalkForwardResult:
    """
    Walk-Forward Optimization from Salirrosas (2016).
    
    1. Train on 'training_window' spins to select bet numbers
    2. Test on next 'testing_window' spins
    3. Roll forward and repeat
    
    This validates that the strategy works out-of-sample,
    not just on historical data (in-sample).
    """
    if len(numbers) < training_window + testing_window:
        raise ValueError(f"Need at least {training_window + testing_window} numbers")
    
    period_results = []
    all_test_bets = []
    in_sample_profits = []
    out_sample_profits = []
    
    start = 0
    while start + training_window + testing_window <= len(numbers):
        train_data = numbers[start:start + training_window]
        test_data = numbers[start + training_window:start + training_window + testing_window]
        
        bet_numbers = strategy_fn(train_data)
        
        if bet_numbers:
            train_backtest = flat_bet_backtest(
                train_data, bet_numbers, bet_amount, initial_balance
            )
            
            test_backtest = flat_bet_backtest(
                test_data, bet_numbers, bet_amount, initial_balance
            )
            
            period_results.append({
                "period": len(period_results) + 1,
                "train_start": start,
                "train_end": start + training_window,
                "test_start": start + training_window,
                "test_end": start + training_window + testing_window,
                "bet_numbers": bet_numbers,
                "in_sample_return": train_backtest.return_pct,
                "out_sample_return": test_backtest.return_pct,
                "in_sample_win_rate": train_backtest.win_rate,
                "out_sample_win_rate": test_backtest.win_rate
            })
            
            in_sample_profits.append(train_backtest.total_profit)
            out_sample_profits.append(test_backtest.total_profit)
            all_test_bets.extend(test_backtest.bets)
        
        start += testing_window
    
    if not period_results:
        raise ValueError("No valid periods found for walk-forward optimization")
    
    test_numbers = []
    for i, result in enumerate(period_results):
        test_numbers.extend(
            numbers[result["test_start"]:result["test_end"]]
        )
    
    if all_test_bets:
        equity = [initial_balance]
        balance = initial_balance
        for bet in all_test_bets:
            balance += bet.profit
            equity.append(balance)
        
        wins = sum(1 for b in all_test_bets if b.won)
        losses = len(all_test_bets) - wins
        max_dd, max_dd_pct = compute_max_drawdown(equity)
        returns = [b.profit / initial_balance for b in all_test_bets]
        results_list = [b.won for b in all_test_bets]
        streak_analysis = compute_consecutive_losses(results_list)
        
        aggregate = BacktestResult(
            initial_balance=initial_balance,
            final_balance=balance,
            total_profit=balance - initial_balance,
            total_bets=len(all_test_bets),
            wins=wins,
            losses=losses,
            win_rate=wins / len(all_test_bets),
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=compute_calmar_ratio(equity),
            sharpe_ratio=compute_sharpe_ratio(returns),
            max_consecutive_losses=streak_analysis["max_consecutive_losses"],
            max_consecutive_wins=streak_analysis["max_consecutive_wins"],
            consecutive_loss_histogram=streak_analysis["loss_histogram"],
            equity_curve=equity,
            bets=all_test_bets
        )
    else:
        aggregate = BacktestResult(
            initial_balance=initial_balance,
            final_balance=initial_balance,
            total_profit=0,
            total_bets=0,
            wins=0,
            losses=0,
            win_rate=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            calmar_ratio=0,
            sharpe_ratio=0,
            max_consecutive_losses=0,
            max_consecutive_wins=0,
            consecutive_loss_histogram={},
            equity_curve=[initial_balance],
            bets=[]
        )
    
    avg_in_sample = np.mean(in_sample_profits) if in_sample_profits else 0
    avg_out_sample = np.mean(out_sample_profits) if out_sample_profits else 0
    
    robustness = avg_out_sample / avg_in_sample if avg_in_sample != 0 else 0
    
    return WalkForwardResult(
        training_periods=len(period_results),
        testing_periods=len(period_results),
        in_sample_return=sum(in_sample_profits),
        out_of_sample_return=sum(out_sample_profits),
        robustness_ratio=robustness,
        period_results=period_results,
        aggregate_backtest=aggregate
    )


def create_bias_strategy(min_probability: float = 0.03) -> Callable[[List[int]], List[int]]:
    """
    Create a strategy function that selects biased numbers.
    
    This is the core strategy from Salirrosas (2016):
    Select numbers with observed probability >= 3%.
    """
    def strategy(train_data: List[int]) -> List[int]:
        counter = Counter(train_data)
        n = len(train_data)
        
        biased = [
            num for num, count in counter.items()
            if count / n >= min_probability
        ]
        
        return biased if biased else []
    
    return strategy


def create_hot_numbers_strategy(top_n: int = 5) -> Callable[[List[int]], List[int]]:
    """
    Create a strategy that bets on the N hottest numbers.
    """
    def strategy(train_data: List[int]) -> List[int]:
        counter = Counter(train_data)
        return [num for num, _ in counter.most_common(top_n)]
    
    return strategy


def create_cold_numbers_strategy(top_n: int = 5) -> Callable[[List[int]], List[int]]:
    """
    Create a strategy that bets on cold numbers (Gambler's Fallacy).
    
    Note: This is included for comparison - academic papers show
    it does NOT work, but it's a common player belief.
    """
    def strategy(train_data: List[int]) -> List[int]:
        counter = Counter(train_data)
        all_counts = [(i, counter.get(i, 0)) for i in range(37)]
        sorted_counts = sorted(all_counts, key=lambda x: x[1])
        return [num for num, _ in sorted_counts[:top_n]]
    
    return strategy


def compare_strategies(
    numbers: List[int],
    initial_balance: float = 1000.0,
    bet_amount: float = 1.0
) -> Dict[str, BacktestResult]:
    """
    Compare multiple betting strategies on the same data.
    """
    strategies = {
        "bias_3pct": create_bias_strategy(0.03),
        "hot_5": create_hot_numbers_strategy(5),
        "cold_5": create_cold_numbers_strategy(5),
        "random_5": lambda x: list(np.random.choice(37, 5, replace=False))
    }
    
    results = {}
    
    for name, strategy_fn in strategies.items():
        try:
            bet_numbers = strategy_fn(numbers[:500])  # Use first 500 for training
            test_numbers = numbers[500:]  # Rest for testing
            
            if bet_numbers and test_numbers:
                backtest = flat_bet_backtest(
                    test_numbers, 
                    bet_numbers, 
                    bet_amount, 
                    initial_balance
                )
                results[name] = backtest
        except Exception:
            continue
    
    return results


def format_backtest_report(result: BacktestResult) -> str:
    """Format backtest results for display."""
    lines = [
        "═" * 50,
        "📊 BACKTEST REPORT",
        "═" * 50,
        f"Initial Balance:    ${result.initial_balance:,.2f}",
        f"Final Balance:      ${result.final_balance:,.2f}",
        f"Total Profit:       ${result.total_profit:,.2f} ({result.return_pct:+.2f}%)",
        "",
        f"Total Bets:         {result.total_bets}",
        f"Wins:               {result.wins} ({result.win_rate*100:.1f}%)",
        f"Losses:             {result.losses}",
        "",
        "─" * 50,
        "📈 RISK METRICS",
        "─" * 50,
        f"Max Drawdown:       ${result.max_drawdown:,.2f} ({result.max_drawdown_pct*100:.1f}%)",
        f"Calmar Ratio:       {result.calmar_ratio:.3f}",
        f"Sharpe Ratio:       {result.sharpe_ratio:.3f}",
        f"Profit Factor:      {result.profit_factor:.2f}",
        "",
        "─" * 50,
        "📉 STREAK ANALYSIS",
        "─" * 50,
        f"Max Consecutive Losses: {result.max_consecutive_losses}",
        f"Max Consecutive Wins:   {result.max_consecutive_wins}",
    ]
    
    if result.consecutive_loss_histogram:
        lines.append("")
        lines.append("Loss Streak Distribution:")
        for streak, count in sorted(result.consecutive_loss_histogram.items()):
            lines.append(f"  {streak} losses in a row: {count} times")
    
    lines.append("═" * 50)
    
    return "\n".join(lines)
