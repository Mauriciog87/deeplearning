from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import Counter
from scipy import stats as scipy_stats


FAIR_PROBABILITY = 1 / 37  # 2.7027%
PROBABILITY_THRESHOLD = 0.03  # 3% filter from Salirrosas paper
CHI_SQUARE_DF = 36  # Degrees of freedom for European roulette


class BiasSignificance(Enum):
    NOT_SIGNIFICANT = "not_significant"
    SIGNIFICANT_95 = "significant_95"
    SIGNIFICANT_99 = "significant_99"
    HIGHLY_SIGNIFICANT = "highly_significant"


@dataclass
class ChiSquareResult:
    statistic: float
    p_value: float
    degrees_of_freedom: int
    critical_95: float
    critical_99: float
    significance: BiasSignificance
    observed_frequencies: Dict[int, int]
    expected_frequency: float
    
    @property
    def is_biased(self) -> bool:
        return self.significance != BiasSignificance.NOT_SIGNIFICANT


@dataclass
class ProbabilityEstimate:
    number: int
    observed_probability: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_99: Tuple[float, float]
    passes_threshold: bool
    z_score: float
    
    @property
    def advantage(self) -> float:
        return (self.observed_probability - FAIR_PROBABILITY) * 100


@dataclass
class OrnsteinUhlenbeckParams:
    theta: float  # Mean reversion speed
    mu: float  # Long-term mean (should be ~1/37)
    sigma: float  # Volatility
    current_probability: float
    expected_next: float


def chi_square_formal_test(
    numbers: List[int],
    confidence_level: float = 0.95
) -> ChiSquareResult:
    """
    Formal Chi-Square test following Salirrosas (2016) methodology.
    
    Tests H0: All numbers are equally likely (fair wheel)
    Tests H1: At least one number has different probability (biased wheel)
    
    Uses 36 degrees of freedom for European roulette (37 outcomes - 1).
    """
    if len(numbers) < 100:
        raise ValueError("Chi-square test requires at least 100 observations for reliability")
    
    counter = Counter(numbers)
    n_total = len(numbers)
    expected = n_total / 37
    
    observed = {i: counter.get(i, 0) for i in range(37)}
    
    chi_sq = sum(
        ((observed[i] - expected) ** 2) / expected 
        for i in range(37)
    )
    
    p_value = 1 - scipy_stats.chi2.cdf(chi_sq, CHI_SQUARE_DF)
    critical_95 = scipy_stats.chi2.ppf(0.95, CHI_SQUARE_DF)  # ~50.998
    critical_99 = scipy_stats.chi2.ppf(0.99, CHI_SQUARE_DF)  # ~58.619
    critical_999 = scipy_stats.chi2.ppf(0.999, CHI_SQUARE_DF)  # ~65.247
    
    if chi_sq >= critical_999:
        significance = BiasSignificance.HIGHLY_SIGNIFICANT
    elif chi_sq >= critical_99:
        significance = BiasSignificance.SIGNIFICANT_99
    elif chi_sq >= critical_95:
        significance = BiasSignificance.SIGNIFICANT_95
    else:
        significance = BiasSignificance.NOT_SIGNIFICANT
    
    return ChiSquareResult(
        statistic=chi_sq,
        p_value=p_value,
        degrees_of_freedom=CHI_SQUARE_DF,
        critical_95=critical_95,
        critical_99=critical_99,
        significance=significance,
        observed_frequencies=observed,
        expected_frequency=expected
    )


def compute_probability_with_confidence(
    numbers: List[int],
    target_number: int,
    confidence_level: float = 0.95
) -> ProbabilityEstimate:
    """
    Compute probability estimate with Wilson score confidence intervals.
    
    Wilson score intervals are preferred over normal approximation for proportions
    as they work better near 0 and 1, which is important for roulette (p ~ 0.027).
    """
    n = len(numbers)
    if n == 0:
        raise ValueError("Need at least one observation")
    
    successes = sum(1 for x in numbers if x == target_number)
    p_hat = successes / n
    
    z_95 = scipy_stats.norm.ppf(0.975)  # ~1.96
    z_99 = scipy_stats.norm.ppf(0.995)  # ~2.576
    
    ci_95 = _wilson_interval(successes, n, z_95)
    ci_99 = _wilson_interval(successes, n, z_99)
    
    z_score = (p_hat - FAIR_PROBABILITY) / np.sqrt(FAIR_PROBABILITY * (1 - FAIR_PROBABILITY) / n)
    
    passes = p_hat >= PROBABILITY_THRESHOLD and ci_95[0] > FAIR_PROBABILITY
    
    return ProbabilityEstimate(
        number=target_number,
        observed_probability=p_hat,
        confidence_interval_95=ci_95,
        confidence_interval_99=ci_99,
        passes_threshold=passes,
        z_score=z_score
    )


def _wilson_interval(successes: int, n: int, z: float) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    p_hat = successes / n
    
    denominator = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    
    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    
    return (max(0, lower), min(1, upper))


def estimate_ornstein_uhlenbeck_params(
    numbers: List[int],
    target_number: int,
    window_size: int = 100
) -> OrnsteinUhlenbeckParams:
    """
    Fit Ornstein-Uhlenbeck process parameters to probability time series.
    
    The O-U process models probability as:
    dP = θ(μ - P)dt + σdW
    
    Where:
    - θ: mean reversion speed (how fast probability returns to fair value)
    - μ: long-term mean (should be ~1/37 for fair wheel)
    - σ: volatility of the process
    - W: Wiener process (Brownian motion)
    
    From Salirrosas (2016), this models how observed probabilities
    fluctuate around the theoretical value.
    """
    if len(numbers) < window_size * 2:
        raise ValueError(f"Need at least {window_size * 2} observations")
    
    probabilities = []
    for i in range(window_size, len(numbers) + 1):
        window = numbers[i - window_size:i]
        p = sum(1 for x in window if x == target_number) / window_size
        probabilities.append(p)
    
    P = np.array(probabilities)
    
    if len(P) < 2:
        return OrnsteinUhlenbeckParams(
            theta=0.1,
            mu=FAIR_PROBABILITY,
            sigma=0.01,
            current_probability=P[-1] if len(P) > 0 else FAIR_PROBABILITY,
            expected_next=FAIR_PROBABILITY
        )
    
    dP = np.diff(P)
    P_lag = P[:-1]
    
    mu = np.mean(P)
    
    try:
        cov_matrix = np.cov(dP, P_lag - mu)
        if cov_matrix.shape == (2, 2):
            theta = -cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0.1
        else:
            theta = 0.1
    except Exception:
        theta = 0.1
    
    theta = max(0.01, min(theta, 2.0))
    
    residuals = dP + theta * (P_lag - mu)
    sigma = np.std(residuals) if len(residuals) > 0 else 0.01
    
    current_p = P[-1]
    expected_next = current_p + theta * (mu - current_p)
    
    return OrnsteinUhlenbeckParams(
        theta=theta,
        mu=mu,
        sigma=sigma,
        current_probability=current_p,
        expected_next=expected_next
    )


def filter_profitable_numbers(
    numbers: List[int],
    min_probability: float = PROBABILITY_THRESHOLD,
    require_statistical_significance: bool = True
) -> List[ProbabilityEstimate]:
    """
    Apply 3% probability filter from Salirrosas (2016).
    
    Returns numbers where:
    1. Observed probability >= 3% (vs 2.7% fair)
    2. Lower bound of 95% CI > fair probability (if require_statistical_significance)
    
    This filter is the key insight from the paper - only bet on numbers
    that show statistically significant positive bias.
    """
    profitable = []
    
    for num in range(37):
        try:
            estimate = compute_probability_with_confidence(numbers, num)
            
            if estimate.observed_probability >= min_probability:
                if require_statistical_significance:
                    if estimate.confidence_interval_95[0] > FAIR_PROBABILITY:
                        profitable.append(estimate)
                else:
                    profitable.append(estimate)
        except ValueError:
            continue
    
    profitable.sort(key=lambda x: x.observed_probability, reverse=True)
    
    return profitable


def compute_entropy(numbers: List[int]) -> Dict[str, float]:
    """
    Compute Shannon entropy and relative entropy of the distribution.
    
    For a fair wheel, entropy should be log2(37) ≈ 5.21 bits.
    Lower entropy indicates concentration on fewer numbers.
    """
    counter = Counter(numbers)
    n = len(numbers)
    
    max_entropy = np.log2(37)
    
    probabilities = [count / n for count in counter.values()]
    observed_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    for i in range(37):
        if i not in counter:
            pass
    
    relative_entropy = observed_entropy / max_entropy
    
    return {
        "observed_entropy": observed_entropy,
        "max_entropy": max_entropy,
        "relative_entropy": relative_entropy,
        "uniformity": relative_entropy,  # 1.0 = perfectly uniform
        "concentration": 1 - relative_entropy  # Higher = more concentrated
    }


def compute_sector_probabilities(
    numbers: List[int],
    sector_type: str = "wheel"
) -> Dict[str, Any]:
    """
    Compute probabilities for wheel sectors.
    
    sector_type options:
    - "wheel": Physical wheel sectors (neighbors on the wheel)
    - "grid": Table layout sectors (rows, columns)
    """
    WHEEL_ORDER = [
        0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
        5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
    ]
    
    n = len(numbers)
    if n == 0:
        return {"error": "No data"}
    
    if sector_type == "wheel":
        sector_size = 6  # ~6 numbers per sector
        sectors = {}
        for i in range(0, 37, sector_size):
            sector_nums = WHEEL_ORDER[i:i + sector_size]
            count = sum(1 for x in numbers if x in sector_nums)
            sector_name = f"sector_{i // sector_size + 1}"
            sectors[sector_name] = {
                "numbers": sector_nums,
                "count": count,
                "probability": count / n,
                "expected": len(sector_nums) / 37
            }
        return {"type": "wheel", "sectors": sectors}
    
    elif sector_type == "grid":
        return {
            "type": "grid",
            "dozen_1": sum(1 for x in numbers if 1 <= x <= 12) / n,
            "dozen_2": sum(1 for x in numbers if 13 <= x <= 24) / n,
            "dozen_3": sum(1 for x in numbers if 25 <= x <= 36) / n,
            "column_1": sum(1 for x in numbers if x > 0 and x % 3 == 1) / n,
            "column_2": sum(1 for x in numbers if x > 0 and x % 3 == 2) / n,
            "column_3": sum(1 for x in numbers if x > 0 and x % 3 == 0) / n,
        }
    
    return {"error": f"Unknown sector_type: {sector_type}"}


def probability_time_series(
    numbers: List[int],
    target_number: int,
    window_size: int = 100,
    step: int = 1
) -> Dict[str, Any]:
    """
    Generate probability time series for a specific number.
    
    Useful for visualizing how probability evolves over time
    and detecting patterns or trends.
    """
    if len(numbers) < window_size:
        return {"error": f"Need at least {window_size} observations"}
    
    time_points = []
    probabilities = []
    ci_lower = []
    ci_upper = []
    
    for i in range(window_size, len(numbers) + 1, step):
        window = numbers[:i]
        estimate = compute_probability_with_confidence(window, target_number)
        
        time_points.append(i)
        probabilities.append(estimate.observed_probability)
        ci_lower.append(estimate.confidence_interval_95[0])
        ci_upper.append(estimate.confidence_interval_95[1])
    
    return {
        "number": target_number,
        "time_points": time_points,
        "probabilities": probabilities,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "fair_probability": FAIR_PROBABILITY,
        "threshold": PROBABILITY_THRESHOLD
    }


def summary_statistics(numbers: List[int]) -> Dict[str, Any]:
    """
    Comprehensive statistical summary following paper methodologies.
    """
    if len(numbers) < 100:
        return {"error": "Need at least 100 observations for reliable statistics"}
    
    chi_sq = chi_square_formal_test(numbers)
    entropy = compute_entropy(numbers)
    profitable = filter_profitable_numbers(numbers, require_statistical_significance=False)
    
    counter = Counter(numbers)
    most_frequent = counter.most_common(5)
    least_frequent = counter.most_common()[:-6:-1]
    
    probs = [counter.get(i, 0) / len(numbers) for i in range(37)]
    std_dev = np.std(probs)
    cv = std_dev / np.mean(probs) if np.mean(probs) > 0 else 0
    
    return {
        "total_spins": len(numbers),
        "unique_numbers": len(counter),
        "chi_square": {
            "statistic": chi_sq.statistic,
            "p_value": chi_sq.p_value,
            "significance": chi_sq.significance.value,
            "is_biased": chi_sq.is_biased
        },
        "entropy": entropy,
        "profitable_numbers": len(profitable),
        "most_frequent": [(n, c, c/len(numbers)*100) for n, c in most_frequent],
        "least_frequent": [(n, c, c/len(numbers)*100) for n, c in least_frequent],
        "coefficient_of_variation": cv,
        "above_threshold_count": sum(1 for e in profitable if e.observed_probability >= PROBABILITY_THRESHOLD)
    }
