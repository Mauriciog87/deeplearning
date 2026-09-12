"""
Wheel Bias Detection Module

Statistical analysis tools to detect potential physical biases
in roulette wheels from historical spin data.

Based on statistical methods including:
- Chi-square goodness-of-fit tests
- Sector frequency analysis  
- Run tests for randomness
- Wheel section hot/cold zone detection

Note: While modern casino wheels are highly engineered,
physical biases can still occur due to:
- Manufacturing imperfections
- Wear and tear over time
- Environmental factors (temperature, humidity)
- Dealer signature patterns
"""

import numpy as np
from scipy import stats
from .randomization import categorical_goodness_of_fit, uniformity_test
from .multiple_testing import false_discovery_control
from .sequential_inference import MonitoringBudget, MultinomialMonitor
from ..settlement import validate_number
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass
from enum import Enum

# Import wheel layout
from ..utils.near_miss import (
    EUROPEAN_WHEEL_ORDER,
    WHEEL_POSITION,
    compute_wheel_sector_frequencies
)


class BiasLevel(Enum):
    """Levels of detected bias significance."""
    NONE = "none"           # No statistically significant bias
    WEAK = "weak"           # p < 0.10, suggestive but not conclusive
    MODERATE = "moderate"   # p < 0.05, statistically significant
    STRONG = "strong"       # p < 0.01, highly significant
    EXTREME = "extreme"     # p < 0.001, very strong evidence


@dataclass
class BiasResult:
    """Result of a bias detection analysis."""
    test_name: str
    statistic: float
    p_value: float
    bias_level: BiasLevel
    details: Dict
    interpretation: str


@dataclass
class WheelBiasReport:
    """Comprehensive wheel bias analysis report."""
    total_spins: int
    chi_square_result: BiasResult
    sector_result: BiasResult
    runs_result: BiasResult
    hot_numbers: List[Tuple[int, float]]   # (number, excess_frequency)
    cold_numbers: List[Tuple[int, float]]  # (number, deficit_frequency)
    hot_sectors: List[Tuple[int, float]]   # (sector_idx, excess_frequency)
    overall_bias: BiasLevel
    recommendations: List[str]
    sequential_evidence: Optional[Dict] = None


def classify_bias(p_value: float) -> BiasLevel:
    """
    Classify bias level based on p-value.
    
    Args:
        p_value: Statistical test p-value
        
    Returns:
        BiasLevel classification
    """
    if p_value < 0.001:
        return BiasLevel.EXTREME
    elif p_value < 0.01:
        return BiasLevel.STRONG
    elif p_value < 0.05:
        return BiasLevel.MODERATE
    elif p_value < 0.10:
        return BiasLevel.WEAK
    else:
        return BiasLevel.NONE


def chi_square_test(
    spin_data: List[int],
    categories: int = 37
) -> BiasResult:
    """
    Chi-square goodness-of-fit test for uniform distribution.
    
    Tests whether the observed frequencies of each number
    differ significantly from expected uniform frequencies.
    
    Args:
        spin_data: List of winning numbers (0-36)
        categories: Number of categories (37 for European roulette)
        
    Returns:
        BiasResult with chi-square statistic and p-value
    """
    spin_data = [validate_number(number) for number in spin_data]
    n = len(spin_data)
    
    # Count occurrences of each number
    observed = np.zeros(categories)
    for num in spin_data:
        if 0 <= num < categories:
            observed[num] += 1
    
    # Expected frequency under uniform distribution
    expected = np.full(categories, n / categories)
    
    # Chi-square test
    chi2, p_value, method = categorical_goodness_of_fit(observed, expected / expected.sum())
    
    # Find most biased numbers
    excess = (observed - expected) / expected
    most_biased_idx = np.argmax(np.abs(excess))
    
    bias_level = classify_bias(p_value)
    
    return BiasResult(
        test_name="Chi-Square Goodness of Fit",
        statistic=chi2,
        p_value=p_value,
        bias_level=bias_level,
        details={
            "method": method,
            "observed": observed.tolist(),
            "expected": expected.tolist(),
            "excess_frequency": excess.tolist(),
            "most_biased_number": int(most_biased_idx),
            "most_biased_excess": float(excess[most_biased_idx]),
            "degrees_of_freedom": categories - 1
        },
        interpretation=f"Chi-square = {chi2:.2f}, p = {p_value:.4f}. "
                       f"{'Significant deviation from uniform distribution.' if bias_level != BiasLevel.NONE else 'No significant deviation detected.'}"
    )


def sector_bias_test(
    spin_data: List[int],
    n_sectors: int = 8
) -> BiasResult:
    """
    Test for bias in wheel sectors.
    
    The wheel is divided into n_sectors equal sections,
    and chi-square test is applied to sector frequencies.
    
    Args:
        spin_data: List of winning numbers
        n_sectors: Number of sectors to divide wheel into
        
    Returns:
        BiasResult for sector analysis
    """
    if not 2 <= n_sectors <= 37:
        raise ValueError('Sector count must be between 2 and 37')
    spin_data = [validate_number(number) for number in spin_data]
    sector_size = 37 / n_sectors
    
    # Assign each spin to a sector
    sector_counts = np.zeros(n_sectors)
    for num in spin_data:
        if num in WHEEL_POSITION:
            pos = WHEEL_POSITION[num]
            sector = min(int(pos / sector_size), n_sectors - 1)
            sector_counts[sector] += 1
    
    # Expected counts
    n = len(spin_data)
    pocket_counts = np.bincount([min(int(pos / sector_size), n_sectors - 1) for pos in range(37)], minlength=n_sectors)
    expected = n * pocket_counts / 37
    
    # Chi-square test on sectors
    chi2, p_value, method = categorical_goodness_of_fit(sector_counts, pocket_counts / 37)
    
    # Identify hot/cold sectors
    excess = (sector_counts - expected) / expected
    hot_sector = int(np.argmax(excess))
    cold_sector = int(np.argmin(excess))
    
    bias_level = classify_bias(p_value)
    
    # Get numbers in hot sector
    hot_sector_numbers = [number for pos, number in enumerate(EUROPEAN_WHEEL_ORDER)
                          if min(int(pos / sector_size), n_sectors - 1) == hot_sector]
    
    return BiasResult(
        test_name="Sector Bias Test",
        statistic=chi2,
        p_value=p_value,
        bias_level=bias_level,
        details={
            "method": method,
            "expected_counts": expected.tolist(),
            "sector_counts": sector_counts.tolist(),
            "excess_frequency": excess.tolist(),
            "hot_sector": hot_sector,
            "cold_sector": cold_sector,
            "hot_sector_numbers": hot_sector_numbers,
            "hot_sector_excess": float(excess[hot_sector]),
            "cold_sector_excess": float(excess[cold_sector])
        },
        interpretation=f"Sector chi-square = {chi2:.2f}, p = {p_value:.4f}. "
                       f"Hot sector: {hot_sector} (numbers: {hot_sector_numbers[:5]}...), "
                       f"excess = {excess[hot_sector]*100:.1f}%"
    )


def runs_test(spin_data: List[int], threshold: int = 18) -> BiasResult:
    """
    Wald-Wolfowitz runs test for randomness.
    
    Tests whether the sequence of "high" (>18) and "low" (≤18) numbers
    shows clustering that deviates from random expectation.
    
    Args:
        spin_data: List of winning numbers
        threshold: Threshold for binary conversion (default 18, median)
        
    Returns:
        BiasResult for runs test
    """
    # Convert to binary sequence (excluding 0)
    binary = []
    for num in spin_data:
        if num > 0:  # Exclude 0
            binary.append(1 if num > threshold else 0)
    
    if len(binary) < 10:
        return BiasResult(
            test_name="Wald-Wolfowitz Runs Test",
            statistic=0,
            p_value=1.0,
            bias_level=BiasLevel.NONE,
            details={"error": "Insufficient data"},
            interpretation="Not enough data for runs test"
        )
    
    # Count runs
    n_runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            n_runs += 1
    
    # Calculate expected runs and variance
    n1 = sum(binary)
    n0 = len(binary) - n1
    n = len(binary)
    
    if n1 == 0 or n0 == 0:
        return BiasResult(
            test_name="Wald-Wolfowitz Runs Test",
            statistic=0,
            p_value=1.0,
            bias_level=BiasLevel.NONE,
            details={"error": "All values same"},
            interpretation="Cannot perform runs test - all values identical"
        )
    
    expected_runs = 1 + (2 * n1 * n0) / n
    variance = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
    
    if variance <= 0:
        variance = 0.001  # Avoid division by zero
    
    # Z-score
    z = (n_runs - expected_runs) / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed
    
    bias_level = classify_bias(p_value)
    
    # Interpretation: too few runs = clustering, too many = alternating
    if n_runs < expected_runs:
        pattern = "clustering (numbers tend to stay high or low)"
    else:
        pattern = "alternating (numbers tend to switch between high and low)"
    
    return BiasResult(
        test_name="Wald-Wolfowitz Runs Test",
        statistic=z,
        p_value=p_value,
        bias_level=bias_level,
        details={
            "observed_runs": n_runs,
            "expected_runs": expected_runs,
            "z_score": z,
            "n_high": n1,
            "n_low": n0,
            "pattern": pattern if bias_level != BiasLevel.NONE else "random"
        },
        interpretation=f"Z = {z:.2f}, p = {p_value:.4f}. "
                       f"{'Detected ' + pattern if bias_level != BiasLevel.NONE else 'Sequence appears random'}"
    )


def color_bias_test(spin_data: List[int]) -> BiasResult:
    """
    Test for bias in color outcomes (Red vs Black).
    
    Args:
        spin_data: List of winning numbers
        
    Returns:
        BiasResult for color analysis
    """
    from ..environment.roulette_env import RED_NUMBERS, BLACK_NUMBERS
    
    red_count = sum(1 for n in spin_data if n in RED_NUMBERS)
    black_count = sum(1 for n in spin_data if n in BLACK_NUMBERS)
    green_count = sum(1 for n in spin_data if n == 0)
    
    # Expected: 18/37 for each color, 1/37 for green
    n = len(spin_data)
    expected_red = n * 18 / 37
    expected_black = n * 18 / 37
    expected_green = n * 1 / 37
    
    observed = np.array([red_count, black_count, green_count])
    expected = np.array([expected_red, expected_black, expected_green])
    
    chi2, p_value = stats.chisquare(observed, expected)
    
    bias_level = classify_bias(p_value)
    
    red_excess = (red_count - expected_red) / expected_red if expected_red > 0 else 0
    black_excess = (black_count - expected_black) / expected_black if expected_black > 0 else 0
    
    return BiasResult(
        test_name="Color Bias Test",
        statistic=chi2,
        p_value=p_value,
        bias_level=bias_level,
        details={
            "red_count": red_count,
            "black_count": black_count,
            "green_count": green_count,
            "expected_red": expected_red,
            "expected_black": expected_black,
            "red_excess": red_excess,
            "black_excess": black_excess
        },
        interpretation=f"Chi-square = {chi2:.2f}, p = {p_value:.4f}. "
                       f"Red: {red_count} ({red_excess*100:+.1f}%), "
                       f"Black: {black_count} ({black_excess*100:+.1f}%)"
    )


def neighbor_clustering_test(
    spin_data: List[int],
    max_distance: int = 3
) -> BiasResult:
    """
    Test for clustering of consecutive spins on the wheel.
    
    Measures whether consecutive winning numbers tend to be
    physically close on the wheel (dealer signature or wheel bias).
    
    Args:
        spin_data: List of winning numbers
        max_distance: Maximum distance to consider "close"
        
    Returns:
        BiasResult for clustering analysis
    """
    from ..utils.near_miss import get_wheel_distance
    
    if len(spin_data) < 2:
        return BiasResult(
            test_name="Neighbor Clustering Test",
            statistic=0,
            p_value=1.0,
            bias_level=BiasLevel.NONE,
            details={"error": "Insufficient data"},
            interpretation="Not enough data"
        )
    
    # Count near-consecutive pairs
    near_pairs = 0
    total_pairs = len(spin_data) - 1
    distances = []
    
    for i in range(total_pairs):
        dist = get_wheel_distance(spin_data[i], spin_data[i+1])
        if dist >= 0:
            distances.append(dist)
            if dist <= max_distance:
                near_pairs += 1
    
    if not distances:
        return BiasResult(
            test_name="Neighbor Clustering Test",
            statistic=0,
            p_value=1.0,
            bias_level=BiasLevel.NONE,
            details={"error": "No valid pairs"},
            interpretation="Could not compute distances"
        )
    
    # Expected probability of being within max_distance
    # On a 37-number wheel, max_distance of 3 means 6 neighbors out of 36 = 6/36
    expected_prob = (2 * max_distance) / 36
    expected_near = total_pairs * expected_prob
    
    # Binomial test (using newer API)
    result = stats.binomtest(near_pairs, total_pairs, expected_prob, alternative='two-sided')
    p_value = result.pvalue
    
    # Effect size
    observed_prob = near_pairs / total_pairs if total_pairs > 0 else 0
    
    bias_level = classify_bias(p_value)
    
    return BiasResult(
        test_name="Neighbor Clustering Test",
        statistic=observed_prob,
        p_value=p_value,
        bias_level=bias_level,
        details={
            "near_pairs": near_pairs,
            "total_pairs": total_pairs,
            "observed_probability": observed_prob,
            "expected_probability": expected_prob,
            "mean_distance": np.mean(distances),
            "median_distance": np.median(distances)
        },
        interpretation=f"Clustering probability = {observed_prob*100:.1f}% (expected {expected_prob*100:.1f}%), "
                       f"p = {p_value:.4f}. "
                       f"{'Consecutive spins show clustering' if observed_prob > expected_prob and bias_level != BiasLevel.NONE else 'No significant clustering'}"
    )


def detect_hot_cold_numbers(
    spin_data: List[int],
    threshold_std: float = 1.5
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Identify "hot" and "cold" numbers based on frequency.
    
    Args:
        spin_data: List of winning numbers
        threshold_std: Standard deviations from mean to consider hot/cold
        
    Returns:
        Tuple of (hot_numbers, cold_numbers) with excess/deficit frequency
    """
    counts = Counter(spin_data)
    n = len(spin_data)
    expected = n / 37
    
    # Calculate frequencies and excess
    frequencies = {}
    for num in range(37):
        count = counts.get(num, 0)
        excess = (count - expected) / expected if expected > 0 else 0
        frequencies[num] = excess
    
    # Calculate threshold
    excesses = list(frequencies.values())
    mean_excess = np.mean(excesses)
    std_excess = np.std(excesses)
    
    hot_threshold = mean_excess + threshold_std * std_excess
    cold_threshold = mean_excess - threshold_std * std_excess
    
    hot_numbers = [(num, exc) for num, exc in frequencies.items() if exc > hot_threshold]
    cold_numbers = [(num, exc) for num, exc in frequencies.items() if exc < cold_threshold]
    
    # Sort by magnitude
    hot_numbers.sort(key=lambda x: x[1], reverse=True)
    cold_numbers.sort(key=lambda x: x[1])
    
    return hot_numbers, cold_numbers


def generate_bias_report(spin_data: List[int]) -> WheelBiasReport:
    """
    Generate comprehensive wheel bias analysis report.
    
    Args:
        spin_data: List of winning numbers
        
    Returns:
        WheelBiasReport with all analysis results
    """
    # Run all tests
    chi_result = chi_square_test(spin_data)
    sector_result = sector_bias_test(spin_data)
    runs_result = runs_test(spin_data)
    tests = [chi_result, sector_result, runs_result]
    adjusted, _ = false_discovery_control([result.p_value for result in tests], method='by')
    for result, q_value in zip(tests, adjusted):
        result.details['q_value'] = q_value
        result.bias_level = classify_bias(q_value)
        result.interpretation = f'Raw p = {result.p_value:.4f}; BY q = {q_value:.4f}. Descriptive evidence only.'
    
    # Find hot/cold numbers and sectors
    hot_numbers, cold_numbers = detect_hot_cold_numbers(spin_data)
    
    # Hot sectors from sector test
    sector_excess = sector_result.details.get("excess_frequency", [])
    hot_sectors = [(i, exc) for i, exc in enumerate(sector_excess) if exc > 0.1]
    hot_sectors.sort(key=lambda x: x[1], reverse=True)
    
    # Determine overall bias level
    bias_levels = [chi_result.bias_level, sector_result.bias_level, runs_result.bias_level]
    if BiasLevel.STRONG in bias_levels or BiasLevel.EXTREME in bias_levels:
        overall_bias = BiasLevel.STRONG
    elif BiasLevel.MODERATE in bias_levels:
        overall_bias = BiasLevel.MODERATE
    elif BiasLevel.WEAK in bias_levels:
        overall_bias = BiasLevel.WEAK
    else:
        overall_bias = BiasLevel.NONE
    
    # Generate recommendations
    recommendations = []
    if overall_bias in [BiasLevel.STRONG, BiasLevel.EXTREME]:
        recommendations.append("Distributional deviation detected; validate it on a later independent session")
        if hot_numbers:
            recommendations.append(f"Hot numbers: {[n for n, _ in hot_numbers[:5]]}")
        if hot_sectors:
            recommendations.append(f"Hot wheel sector: {hot_sectors[0][0]} ({hot_sectors[0][1]*100:.1f}% excess)")
    elif overall_bias == BiasLevel.MODERATE:
        recommendations.append("A distributional signal needs independent confirmation")
        recommendations.append("Recommend gathering more data to confirm")
    elif overall_bias == BiasLevel.WEAK:
        recommendations.append("Weak bias signal - insufficient for reliable exploitation")
        recommendations.append("Continue data collection")
    else:
        recommendations.append("No significant bias detected")
        recommendations.append("Non-rejection does not establish fairness")
    
    return WheelBiasReport(
        total_spins=len(spin_data),
        chi_square_result=chi_result,
        sector_result=sector_result,
        runs_result=runs_result,
        hot_numbers=hot_numbers,
        cold_numbers=cold_numbers,
        hot_sectors=hot_sectors,
        overall_bias=overall_bias,
        recommendations=recommendations
    )


def print_bias_report(report: WheelBiasReport):
    """Pretty print a bias report."""
    print("=" * 60)
    print("WHEEL BIAS ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nTotal Spins Analyzed: {report.total_spins}")
    print(f"Fixed-sample evidence level: {report.overall_bias.value.upper()}")
    if report.sequential_evidence is not None:
        evidence = report.sequential_evidence
        print(f"Sequential evidence: anytime p={evidence['anytime_p_value']:.6g}; allocated alpha={evidence['allocated_alpha']:.6g}; rejected={evidence['rejected']}")
        print(evidence['interpretation'])
    
    print("\n--- Individual Test Results ---")
    
    for result in [report.chi_square_result, report.sector_result, report.runs_result]:
        print(f"\n{result.test_name}:")
        print(f"  Statistic: {result.statistic:.4f}")
        print(f"  P-value: {result.p_value:.6f}; BY q-value: {result.details.get('q_value', result.p_value):.6f}")
        print(f"  Bias Level: {result.bias_level.value}")
        print(f"  {result.interpretation}")
    
    print("\n--- Hot Numbers ---")
    if report.hot_numbers:
        for num, excess in report.hot_numbers[:5]:
            print(f"  Number {num:2d}: {excess*100:+.1f}% excess")
    else:
        print("  No numbers above the descriptive frequency threshold")
    
    print("\n--- Cold Numbers ---")
    if report.cold_numbers:
        for num, excess in report.cold_numbers[:5]:
            print(f"  Number {num:2d}: {excess*100:+.1f}% deficit")
    else:
        print("  No numbers below the descriptive frequency threshold")
    
    print("\n--- Recommendations ---")
    for rec in report.recommendations:
        print(f"  • {rec}")
    
    print("\n" + "=" * 60)


class WheelBiasAnalyzer:
    """
    Real-time wheel bias analyzer for ongoing analysis.
    
    Maintains running statistics and can update analysis
    as new spin data comes in.
    """
    
    def __init__(
        self,
        min_spins_for_analysis: int = 100,
        analysis_window: Optional[int] = None,
        *, stream_id: str = 'wheel', alpha: float = .05, streams=None, descriptive_reports: bool = True
    ):
        """
        Initialize the analyzer.
        
        Args:
            min_spins_for_analysis: Minimum spins before analysis
            analysis_window: Only consider last N spins (None = all)
        """
        self.min_spins = min_spins_for_analysis
        self.analysis_window = analysis_window
        self.descriptive_reports = descriptive_reports
        if self.min_spins < 1 or (analysis_window is not None and analysis_window < 1):
            raise ValueError('Analysis windows must be positive')
        self.stream_id = stream_id
        self.budget = MonitoringBudget(alpha, tuple(streams) if streams is not None else (stream_id,))
        self.restart_index = 0
        self.monitor = MultinomialMonitor(stream_id, alpha=self.budget.allocation(stream_id))
        self._seen_events = {}
        self.spin_history = []
        self._last_report = None
    
    @property
    def total_spins(self) -> int:
        """Total number of spins recorded."""
        return len(self.spin_history)
    
    def add_spin(self, number: int, *, event_id: Optional[str] = None) -> Optional[WheelBiasReport]:
        """
        Add a new spin and optionally update analysis.
        
        Args:
            number: Winning number (0-36)
            
        Returns:
            Updated report if enough data, None otherwise
        """
        number = validate_number(number)
        if event_id is None:
            index = len(self._seen_events)
            event_id = f'{self.stream_id}:auto:{index}'
            while event_id in self._seen_events:
                index += 1
                event_id = f'{self.stream_id}:auto:{index}'
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('A nonempty event ID is required')
        if event_id in self._seen_events:
            if self._seen_events[event_id] != number:
                raise ValueError('The event ID was already recorded with a different outcome')
            return self._last_report
        self.monitor.update(number, event_id=event_id)
        self._seen_events[event_id] = number
        self.spin_history.append(number)
        
        # Only analyze if we have enough data
        if self.descriptive_reports and len(self.spin_history) >= self.min_spins:
            data = self.spin_history
            if self.analysis_window:
                data = self.spin_history[-self.analysis_window:]
            self._last_report = generate_bias_report(data)
            self._last_report.sequential_evidence = self.sequential_evidence()
            return self._last_report
        
        return None
    
    def add_spins(self, numbers: List[int]) -> Optional[WheelBiasReport]:
        """Add multiple spins at once."""
        for n in numbers:
            self.add_spin(n)
        return self._last_report
    
    def get_report(self) -> Optional[WheelBiasReport]:
        """Get the most recent analysis report."""
        return self._last_report
    
    def get_hot_numbers(self, top_n: int = 5) -> List[int]:
        """Get the top N hot numbers."""
        if self._last_report and self._last_report.hot_numbers:
            return [num for num, _ in self._last_report.hot_numbers[:top_n]]
        return []
    
    def get_recommended_bets(self) -> List[int]:
        if not self.monitor.snapshot()['rejected']:
            return []
        candidates = [(number, self.monitor.probability_bounds([number])[0]) for number in range(37)]
        return [number for number, lower in sorted(candidates, key=lambda item: (-item[1], item[0])) if lower > 1 / 36]

    def sequential_evidence(self):
        return dict(self.monitor.snapshot(), total_alpha=self.budget.alpha,
                    streams=list(self.budget.streams), restart_index=self.restart_index)

    def to_state(self):
        return {'schema_version': 1, 'stream_id': self.stream_id, 'alpha': self.budget.alpha,
                'streams': list(self.budget.streams), 'restart_index': self.restart_index,
                'min_spins': self.min_spins, 'analysis_window': self.analysis_window, 'descriptive_reports': self.descriptive_reports,
                'seen_events': list(self._seen_events.items()), 'monitor': self.monitor.to_state()}

    @classmethod
    def from_state(cls, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported wheel analyzer state')
        analyzer = cls(state['min_spins'], state['analysis_window'], stream_id=state['stream_id'],
                       alpha=state['alpha'], streams=state['streams'], descriptive_reports=state['descriptive_reports'])
        allocated = analyzer.budget.allocation(analyzer.stream_id, state['restart_index'])
        analyzer.restart_index = state['restart_index']
        analyzer.monitor = MultinomialMonitor.from_state(state['monitor'], stream_id=analyzer.stream_id)
        if analyzer.monitor.alpha != allocated:
            raise ValueError('The saved monitor does not match the declared error budget')
        for event_id, number in state['seen_events']:
            if not isinstance(event_id, str) or not event_id or event_id in analyzer._seen_events:
                raise ValueError('Invalid event registry in wheel analyzer state')
            analyzer._seen_events[event_id] = validate_number(number)
        for event_id, number in state['monitor']['events']:
            if analyzer._seen_events.get(event_id) != number:
                raise ValueError('The saved monitor is inconsistent with its event registry')
            analyzer.spin_history.append(number)
        current = [tuple(item) for item in state['monitor']['events']]
        seen = list(analyzer._seen_events.items())
        if (current and current != seen[-len(current):]) or (analyzer.restart_index == 0 and current != seen):
            raise ValueError('The saved segment is not the expected suffix of the event registry')
        if analyzer.descriptive_reports and len(analyzer.spin_history) >= analyzer.min_spins:
            data = analyzer.spin_history[-analyzer.analysis_window:] if analyzer.analysis_window else analyzer.spin_history
            analyzer._last_report = generate_bias_report(data)
            analyzer._last_report.sequential_evidence = analyzer.sequential_evidence()
        return analyzer
    
    def reset(self):
        """Reset the analyzer."""
        self.spin_history = []
        self._last_report = None
        self.restart_index += 1
        self.monitor = MultinomialMonitor(self.stream_id, alpha=self.budget.allocation(self.stream_id, self.restart_index))


def formal_chi_square_test(
    spin_data: List[int],
    confidence_level: float = 0.95
) -> Dict:
    """
    Formal Chi-Square test following Salirrosas (2016) methodology.
    
    Uses 36 degrees of freedom for European roulette (37 - 1).
    Returns critical values at 95% and 99% confidence.
    """
    if len(spin_data) < 100:
        return {
            "error": "Need at least 100 spins for reliable chi-square test",
            "spins": len(spin_data)
        }
    
    n = len(spin_data)
    counter = Counter(spin_data)
    expected = n / 37
    
    chi_sq = sum(
        ((counter.get(i, 0) - expected) ** 2) / expected
        for i in range(37)
    )
    
    df = 36
    
    critical_95 = stats.chi2.ppf(0.95, df)  # ~50.998
    critical_99 = stats.chi2.ppf(0.99, df)  # ~58.619
    critical_999 = stats.chi2.ppf(0.999, df)  # ~65.247
    chi_sq, p_value, method = uniformity_test(spin_data)
    
    if p_value <= .001:
        significance = "highly_significant"
        conclusion = "Very strong evidence of wheel bias (p < 0.001)"
    elif p_value <= .01:
        significance = "significant_99"
        conclusion = "Strong evidence of wheel bias (p < 0.01)"
    elif p_value <= .05:
        significance = "significant_95"
        conclusion = "Evidence of wheel bias (p < 0.05)"
    else:
        significance = "not_significant"
        conclusion = "No statistically significant bias detected"
    
    observed_probs = {i: counter.get(i, 0) / n for i in range(37)}
    biased_numbers = [
        num for num, prob in observed_probs.items()
        if prob >= 0.03  # 3% threshold
    ]
    
    return {
        "method": method,
        "chi_square": chi_sq,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "critical_95": critical_95,
        "critical_99": critical_99,
        "critical_999": critical_999,
        "significance": significance,
        "conclusion": conclusion,
        "total_spins": n,
        "biased_numbers_3pct": biased_numbers,
        "most_frequent": counter.most_common(5),
        "least_frequent": counter.most_common()[:-6:-1] if len(counter) >= 5 else []
    }


def compute_probability_with_ci(
    spin_data: List[int],
    number: int,
    confidence: float = 0.95
) -> Dict:
    """
    Compute exact fixed-sample intervals allocated across all 37 numbers.
    """
    from .statistics import compute_probability_with_confidence, BREAK_EVEN_PROBABILITY

    estimate = compute_probability_with_confidence(spin_data, number, confidence)
    n = len(spin_data)
    successes = sum(1 for x in spin_data if x == number)
    p_hat = estimate.observed_probability
    ci_lower, ci_upper = estimate.confidence_interval
    fair_prob = 1 / 37
    
    return {
        "number": number,
        "count": successes,
        "total": n,
        "probability": p_hat,
        "fair_probability": fair_prob,
        "confidence_interval": (ci_lower, ci_upper),
        "z_score": estimate.z_score,
        "advantage_pct": (p_hat - fair_prob) * 100,
        "passes_3pct_threshold": estimate.passes_threshold,
        "statistically_significant": ci_lower > fair_prob,
        "lower_bound_above_break_even": ci_lower > BREAK_EVEN_PROBABILITY,
        "break_even_probability": BREAK_EVEN_PROBABILITY,
        "inference": estimate.inference,
        "family_size": estimate.family_size,
        "confidence_level": confidence
    }


def analyze_all_numbers(spin_data: List[int]) -> List[Dict]:
    """
    Analyze all 37 numbers and return sorted by probability.
    """
    results = []
    for num in range(37):
        result = compute_probability_with_ci(spin_data, num)
        results.append(result)
    
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def format_formal_bias_report(spin_data: List[int]) -> str:
    """
    Generate formatted bias report following academic paper methodology.
    """
    chi_sq = formal_chi_square_test(spin_data)
    
    if "error" in chi_sq:
        return f"❌ {chi_sq['error']}"
    
    lines = [
        "═" * 60,
        "📊 FORMAL WHEEL BIAS ANALYSIS",
        "   Fixed-sample diagnostics inspired by Salirrosas (2016)",
        "   Pocket intervals: exact binomial, Bonferroni family of 37; IID required",
        "   Frequency differences and non-rejection do not establish profitability or fairness",
        "═" * 60,
        "",
        f"Total Spins: {chi_sq['total_spins']}",
        "",
        "─" * 60,
        "CHI-SQUARE GOODNESS-OF-FIT TEST",
        "─" * 60,
        f"Chi-Square Statistic: {chi_sq['chi_square']:.3f}",
        f"Degrees of Freedom:   {chi_sq['degrees_of_freedom']}",
        f"P-Value:              {chi_sq['p_value']:.6f}",
        "",
        f"Critical Values:",
        f"  95% confidence: {chi_sq['critical_95']:.3f}",
        f"  99% confidence: {chi_sq['critical_99']:.3f}",
        f"  99.9% confidence: {chi_sq['critical_999']:.3f}",
        "",
        f"Result: {chi_sq['conclusion']}",
        "",
    ]
    
    if chi_sq['biased_numbers_3pct']:
        lines.append("─" * 60)
        lines.append("NUMBERS PASSING 3% THRESHOLD")
        lines.append("─" * 60)
        for num in chi_sq['biased_numbers_3pct']:
            ci_data = compute_probability_with_ci(spin_data, num)
            sig = "✅" if ci_data['statistically_significant'] else "⚠️"
            lines.append(
                f"  {sig} #{num:2d}: {ci_data['probability']*100:.2f}% "
                f"[CI: {ci_data['confidence_interval'][0]*100:.2f}%-{ci_data['confidence_interval'][1]*100:.2f}%]"
            )
    
    lines.append("")
    lines.append("─" * 60)
    lines.append("MOST/LEAST FREQUENT NUMBERS")
    lines.append("─" * 60)
    
    lines.append("Most frequent:")
    for num, count in chi_sq['most_frequent']:
        pct = count / chi_sq['total_spins'] * 100
        lines.append(f"  #{num:2d}: {count} ({pct:.2f}%)")
    
    lines.append("")
    lines.append("Least frequent:")
    for num, count in chi_sq['least_frequent']:
        pct = count / chi_sq['total_spins'] * 100
        lines.append(f"  #{num:2d}: {count} ({pct:.2f}%)")
    
    lines.append("═" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Wheel Bias Detection Test ===\n")
    
    # Generate test data
    np.random.seed(42)
    
    # 1. Test with uniform data (no bias)
    print("--- Test 1: Uniform Distribution (No Bias) ---")
    uniform_data = list(np.random.randint(0, 37, 1000))
    report = generate_bias_report(uniform_data)
    print_bias_report(report)
    
    # 2. Test with biased data (certain sector favored)
    print("\n--- Test 2: Biased Data (Sector 0 favored) ---")
    biased_data = list(np.random.randint(0, 37, 700))  # 70% uniform
    # Add extra spins in sector 0 (numbers 0, 32, 15, 19, 4)
    sector_0_numbers = [0, 32, 15, 19, 4]
    biased_data.extend(np.random.choice(sector_0_numbers, 300))  # 30% from sector 0
    np.random.shuffle(biased_data)
    
    report = generate_bias_report(biased_data)
    print_bias_report(report)
    
    # 3. Test with single hot number
    print("\n--- Test 3: Single Hot Number (17 favored) ---")
    hot_number_data = list(np.random.randint(0, 37, 800))
    hot_number_data.extend([17] * 200)  # 17 appears 200 extra times
    np.random.shuffle(hot_number_data)
    
    report = generate_bias_report(hot_number_data)
    print_bias_report(report)
    
    # 4. Test the real-time analyzer
    print("\n--- Test 4: Real-time Analyzer ---")
    analyzer = WheelBiasAnalyzer(min_spins_for_analysis=100)
    
    # Simulate spinning
    for i in range(150):
        num = np.random.randint(0, 37)
        report = analyzer.add_spin(num)
    
    print(f"Spins collected: {len(analyzer.spin_history)}")
    print(f"Hot numbers: {analyzer.get_hot_numbers()}")
    print(f"Recommended bets: {analyzer.get_recommended_bets()}")
    
    if analyzer.get_report():
        print(f"Overall bias: {analyzer.get_report().overall_bias.value}")
