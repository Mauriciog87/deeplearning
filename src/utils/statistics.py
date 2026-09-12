from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import Counter
from scipy import stats as scipy_stats
from .randomization import uniformity_test
from src.settlement import validate_number


FAIR_PROBABILITY = 1 / 37  # 2.7027%
BREAK_EVEN_PROBABILITY = 1 / 36
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
    method: str = 'chi-square asymptotic'
    
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
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    confidence_level: float = .95
    family_size: int = 37
    inference: str = 'Fixed-sample exact binomial intervals with Bonferroni allocation; IID outcomes required, not valid for unrestricted repeated looks'
    
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
    iid_overlap_lag1: float = 0.0
    iid_overlap_theta: float = 0.0
    raw_indicator_lag1: Optional[float] = None
    rolling_frequency_lag1: Optional[float] = None
    interpretation: str = 'Descriptive regression of overlapping window frequencies. expected_next concerns the next window frequency, not the next spin probability. Overlap alone creates apparent mean reversion under IID outcomes.'


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
    
    chi_sq, p_value, method = uniformity_test(numbers)
    critical_95 = scipy_stats.chi2.ppf(0.95, CHI_SQUARE_DF)  # ~50.998
    critical_99 = scipy_stats.chi2.ppf(0.99, CHI_SQUARE_DF)  # ~58.619
    
    if p_value <= .001:
        significance = BiasSignificance.HIGHLY_SIGNIFICANT
    elif p_value <= .01:
        significance = BiasSignificance.SIGNIFICANT_99
    elif p_value <= .05:
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
        expected_frequency=expected,
        method=method
    )


def compute_probability_with_confidence(
    numbers: List[int],
    target_number: int,
    confidence_level: float = 0.95,
    *, family_size: int = 37
) -> ProbabilityEstimate:
    """
    Compute exact simultaneous fixed-sample intervals and descriptive Wilson intervals.
    """
    n = len(numbers)
    if n == 0:
        raise ValueError("Need at least one observation")
    target_number = validate_number(target_number)
    numbers = [validate_number(number) for number in numbers]
    if not 0 < confidence_level < 1 or isinstance(family_size, bool) or not isinstance(family_size, int) or family_size < 1:
        raise ValueError('Invalid confidence level or family size')
    
    successes = sum(1 for x in numbers if x == target_number)
    p_hat = successes / n
    
    z_95 = scipy_stats.norm.ppf(0.975)  # ~1.96
    z_99 = scipy_stats.norm.ppf(0.995)  # ~2.576
    
    ci_95 = _wilson_interval(successes, n, z_95)
    ci_99 = _wilson_interval(successes, n, z_99)
    
    z_score = (p_hat - FAIR_PROBABILITY) / np.sqrt(FAIR_PROBABILITY * (1 - FAIR_PROBABILITY) / n)
    
    interval = scipy_stats.binomtest(successes, n).proportion_ci(
        confidence_level=1 - (1 - confidence_level) / family_size, method='exact')
    passes = p_hat >= PROBABILITY_THRESHOLD and interval.low > BREAK_EVEN_PROBABILITY
    
    return ProbabilityEstimate(
        number=target_number,
        observed_probability=p_hat,
        confidence_interval_95=ci_95,
        confidence_interval_99=ci_99,
        passes_threshold=passes,
        z_score=z_score,
        confidence_interval=(interval.low, interval.high),
        confidence_level=confidence_level,
        family_size=family_size
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
    Fit a descriptive linear drift regression to overlapping window frequencies.
    The IID overlap null has lag-one correlation (window_size-1)/window_size
    and regression coefficient theta=1/window_size. Neither is predictive evidence.
    """
    target_number = validate_number(target_number)
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
        raise ValueError('Window size must be an integer of at least two')
    if len(numbers) < window_size * 2:
        raise ValueError(f"Need at least {window_size * 2} observations")
    
    indicators = np.asarray([validate_number(number) == target_number for number in numbers], dtype=float)
    cumulative = np.concatenate(([0], np.cumsum(indicators)))
    P = (cumulative[window_size:] - cumulative[:-window_size]) / window_size
    
    dP = np.diff(P)
    P_lag = P[:-1]
    
    mu = np.mean(P)
    
    variance = np.var(P_lag)
    theta = -float(np.mean((dP - dP.mean()) * (P_lag - P_lag.mean()))) / variance if variance > 0 else 0.0
    
    residuals = dP + theta * (P_lag - mu)
    sigma = np.std(residuals) if len(residuals) > 0 else 0.01
    
    current_p = P[-1]
    expected_next = current_p + theta * (mu - current_p)
    
    return OrnsteinUhlenbeckParams(
        theta=theta,
        mu=mu,
        sigma=sigma,
        current_probability=current_p,
        expected_next=expected_next,
        iid_overlap_lag1=(window_size - 1) / window_size,
        iid_overlap_theta=1 / window_size,
        raw_indicator_lag1=float(np.corrcoef(indicators[:-1], indicators[1:])[0, 1])
        if np.var(indicators[:-1]) > 0 and np.var(indicators[1:]) > 0 else None,
        rolling_frequency_lag1=float(np.corrcoef(P[:-1], P[1:])[0, 1])
        if np.var(P[:-1]) > 0 and np.var(P[1:]) > 0 else None
    )


def filter_profitable_numbers(
    numbers: List[int],
    min_probability: float = PROBABILITY_THRESHOLD,
    require_statistical_significance: bool = True
) -> List[ProbabilityEstimate]:
    """
    Apply an observed-frequency filter, optionally requiring a simultaneous
    fixed-sample lower bound above the straight-bet break-even probability.
    """
    if not np.isfinite(min_probability) or not 0 <= min_probability <= 1:
        raise ValueError('Minimum probability must lie within [0,1]')
    if not numbers:
        return []
    profitable = []
    
    for num in range(37):
        estimate = compute_probability_with_confidence(numbers, num)
        if estimate.observed_probability >= min_probability:
            if not require_statistical_significance or estimate.confidence_interval[0] > BREAK_EVEN_PROBABILITY:
                profitable.append(estimate)
    
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
        "threshold": PROBABILITY_THRESHOLD,
        "break_even_probability": BREAK_EVEN_PROBABILITY,
        "interpretation": "Expanding-prefix frequencies with pointwise Wilson intervals; descriptive across repeated looks"
    }


def summary_statistics(numbers: List[int]) -> Dict[str, Any]:
    """
    Comprehensive statistical summary following paper methodologies.
    """
    if len(numbers) < 100:
        return {"error": "Need at least 100 observations for reliable statistics"}
    
    chi_sq = chi_square_formal_test(numbers)
    entropy = compute_entropy(numbers)
    profitable = filter_profitable_numbers(numbers)
    
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
