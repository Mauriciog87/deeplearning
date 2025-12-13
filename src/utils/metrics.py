from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import Counter
from scipy import stats as scipy_stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, entropy


FAIR_DISTRIBUTION = np.array([1/37] * 37)


@dataclass
class DistributionComparison:
    jsd: float  # Jensen-Shannon Divergence
    wasserstein: float  # Earth Mover's Distance
    kl_divergence: float  # Kullback-Leibler Divergence
    total_variation: float  # Total Variation Distance
    chi_square_distance: float  # Chi-square distance
    hellinger: float  # Hellinger distance


@dataclass
class DensityRatioResult:
    number: int
    observed_prob: float
    expected_prob: float
    density_ratio: float  # observed / expected
    log_density_ratio: float  # log(observed / expected)
    is_overrepresented: bool
    deviation_sigma: float  # Standard deviations from expected


def compute_empirical_distribution(numbers: List[int]) -> np.ndarray:
    """
    Compute empirical probability distribution from observed numbers.
    
    Returns array of length 37 with probabilities for each number.
    """
    if not numbers:
        return FAIR_DISTRIBUTION.copy()
    
    counter = Counter(numbers)
    n = len(numbers)
    
    distribution = np.array([counter.get(i, 0) / n for i in range(37)])
    
    return distribution


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Compute Jensen-Shannon Divergence between two distributions.
    
    JSD is a symmetric and smoothed version of KL divergence.
    JSD(P||Q) = (KL(P||M) + KL(Q||M)) / 2
    where M = (P + Q) / 2
    
    From Merchie (2018) thesis on anomaly detection.
    Range: [0, 1] where 0 = identical distributions.
    """
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    
    p = p / p.sum()
    q = q / q.sum()
    
    return float(jensenshannon(p, q) ** 2)


def compute_wasserstein(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Compute Wasserstein Distance (Earth Mover's Distance).
    
    Measures the minimum "work" to transform one distribution into another.
    More robust than KL divergence for distributions with different supports.
    
    From Merchie (2018) thesis.
    """
    support = np.arange(37)
    
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    p = p / p.sum()
    q = q / q.sum()
    
    return float(wasserstein_distance(support, support, p, q))


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Compute Kullback-Leibler Divergence KL(P||Q).
    
    Note: KL is asymmetric and can be infinite if Q has zeros where P has mass.
    We add small epsilon to avoid this.
    """
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    
    p = p / p.sum()
    q = q / q.sum()
    
    return float(entropy(p, q))


def total_variation_distance(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Compute Total Variation Distance.
    
    TV(P,Q) = (1/2) * sum(|P(x) - Q(x)|)
    
    Range: [0, 1]
    """
    p = p / p.sum()
    q = q / q.sum()
    
    return float(0.5 * np.sum(np.abs(p - q)))


def chi_square_distance(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Compute Chi-Square Distance between distributions.
    
    χ²(P,Q) = sum((P(x) - Q(x))² / Q(x))
    """
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    
    p = p / p.sum()
    q = q / q.sum()
    
    return float(np.sum((p - q) ** 2 / q))


def hellinger_distance(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Compute Hellinger Distance.
    
    H(P,Q) = (1/√2) * sqrt(sum((√P(x) - √Q(x))²))
    
    Range: [0, 1]
    """
    p = p / p.sum()
    q = q / q.sum()
    
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def compare_to_fair(numbers: List[int]) -> DistributionComparison:
    """
    Compare observed distribution to fair (uniform) distribution.
    
    Returns all distance metrics from Merchie (2018) thesis.
    """
    observed = compute_empirical_distribution(numbers)
    
    return DistributionComparison(
        jsd=jensen_shannon_divergence(observed, FAIR_DISTRIBUTION),
        wasserstein=compute_wasserstein(observed, FAIR_DISTRIBUTION),
        kl_divergence=kl_divergence(observed, FAIR_DISTRIBUTION),
        total_variation=total_variation_distance(observed, FAIR_DISTRIBUTION),
        chi_square_distance=chi_square_distance(observed, FAIR_DISTRIBUTION),
        hellinger=hellinger_distance(observed, FAIR_DISTRIBUTION)
    )


def compare_distributions(
    numbers1: List[int],
    numbers2: List[int]
) -> DistributionComparison:
    """
    Compare two observed distributions.
    
    Useful for comparing different sessions or time periods.
    """
    dist1 = compute_empirical_distribution(numbers1)
    dist2 = compute_empirical_distribution(numbers2)
    
    return DistributionComparison(
        jsd=jensen_shannon_divergence(dist1, dist2),
        wasserstein=compute_wasserstein(dist1, dist2),
        kl_divergence=kl_divergence(dist1, dist2),
        total_variation=total_variation_distance(dist1, dist2),
        chi_square_distance=chi_square_distance(dist1, dist2),
        hellinger=hellinger_distance(dist1, dist2)
    )


def compute_density_ratios(
    numbers: List[int],
    baseline: Optional[np.ndarray] = None
) -> List[DensityRatioResult]:
    """
    Compute density ratios for each number.
    
    Density ratio = P_observed(x) / P_expected(x)
    
    > 1: Number appears more than expected (overrepresented)
    < 1: Number appears less than expected (underrepresented)
    = 1: Number appears as expected
    
    From Merchie (2018) for anomaly detection.
    """
    if baseline is None:
        baseline = FAIR_DISTRIBUTION
    
    observed = compute_empirical_distribution(numbers)
    n = len(numbers)
    
    results = []
    for i in range(37):
        obs_p = observed[i]
        exp_p = baseline[i]
        
        ratio = obs_p / exp_p if exp_p > 0 else float('inf')
        
        log_ratio = np.log(ratio) if ratio > 0 else float('-inf')
        
        std_error = np.sqrt(exp_p * (1 - exp_p) / n) if n > 0 else 0
        deviation = (obs_p - exp_p) / std_error if std_error > 0 else 0
        
        results.append(DensityRatioResult(
            number=i,
            observed_prob=obs_p,
            expected_prob=exp_p,
            density_ratio=ratio,
            log_density_ratio=log_ratio,
            is_overrepresented=ratio > 1,
            deviation_sigma=deviation
        ))
    
    return results


def find_anomalous_numbers(
    numbers: List[int],
    sigma_threshold: float = 2.0
) -> Dict[str, List[DensityRatioResult]]:
    """
    Find numbers that deviate significantly from expected.
    
    Returns numbers that are more than sigma_threshold standard deviations
    away from expected probability.
    
    From Merchie (2018) anomaly detection methodology.
    """
    ratios = compute_density_ratios(numbers)
    
    overrepresented = [
        r for r in ratios 
        if r.deviation_sigma > sigma_threshold
    ]
    
    underrepresented = [
        r for r in ratios 
        if r.deviation_sigma < -sigma_threshold
    ]
    
    overrepresented.sort(key=lambda x: x.deviation_sigma, reverse=True)
    underrepresented.sort(key=lambda x: x.deviation_sigma)
    
    return {
        "overrepresented": overrepresented,
        "underrepresented": underrepresented,
        "threshold_sigma": sigma_threshold,
        "total_anomalies": len(overrepresented) + len(underrepresented)
    }


def rolling_jsd(
    numbers: List[int],
    window_size: int = 100,
    step: int = 10
) -> Dict[str, Any]:
    """
    Compute rolling JSD to detect distribution changes over time.
    
    Useful for detecting when a wheel's bias changes.
    """
    if len(numbers) < window_size * 2:
        return {"error": f"Need at least {window_size * 2} numbers"}
    
    positions = []
    jsd_values = []
    
    for i in range(window_size, len(numbers) - window_size + 1, step):
        window1 = numbers[i - window_size:i]
        window2 = numbers[i:i + window_size]
        
        dist1 = compute_empirical_distribution(window1)
        dist2 = compute_empirical_distribution(window2)
        
        jsd = jensen_shannon_divergence(dist1, dist2)
        
        positions.append(i)
        jsd_values.append(jsd)
    
    return {
        "positions": positions,
        "jsd_values": jsd_values,
        "mean_jsd": np.mean(jsd_values),
        "max_jsd": np.max(jsd_values),
        "max_jsd_position": positions[np.argmax(jsd_values)] if jsd_values else None,
        "window_size": window_size
    }


def compute_stability_score(
    numbers: List[int],
    window_size: int = 100,
    num_windows: int = 5
) -> Dict[str, Any]:
    """
    Compute stability score - how consistent is the distribution over time?
    
    Lower score = more stable/consistent
    Higher score = distribution is changing over time
    """
    if len(numbers) < window_size * num_windows:
        num_windows = len(numbers) // window_size
    
    if num_windows < 2:
        return {"error": "Not enough data for stability analysis"}
    
    window_dists = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = numbers[start:end]
        window_dists.append(compute_empirical_distribution(window))
    
    jsd_values = []
    for i in range(len(window_dists)):
        for j in range(i + 1, len(window_dists)):
            jsd = jensen_shannon_divergence(window_dists[i], window_dists[j])
            jsd_values.append(jsd)
    
    return {
        "stability_score": np.mean(jsd_values),
        "stability_std": np.std(jsd_values),
        "min_jsd": np.min(jsd_values),
        "max_jsd": np.max(jsd_values),
        "num_windows": num_windows,
        "window_size": window_size,
        "interpretation": "stable" if np.mean(jsd_values) < 0.01 else "variable"
    }


def format_distribution_report(comparison: DistributionComparison) -> str:
    """Format distribution comparison for display."""
    lines = [
        "═" * 50,
        "📊 DISTRIBUTION COMPARISON",
        "═" * 50,
        "",
        f"Jensen-Shannon Divergence:  {comparison.jsd:.6f}",
        f"Wasserstein Distance:       {comparison.wasserstein:.6f}",
        f"KL Divergence:              {comparison.kl_divergence:.6f}",
        f"Total Variation:            {comparison.total_variation:.6f}",
        f"Chi-Square Distance:        {comparison.chi_square_distance:.6f}",
        f"Hellinger Distance:         {comparison.hellinger:.6f}",
        "",
        "─" * 50,
        "📖 INTERPRETATION",
        "─" * 50,
    ]
    
    if comparison.jsd < 0.001:
        lines.append("✅ Distribution is very close to fair")
    elif comparison.jsd < 0.01:
        lines.append("⚠️  Small deviation from fair distribution")
    elif comparison.jsd < 0.05:
        lines.append("⚠️  Moderate deviation - possible bias")
    else:
        lines.append("🚨 Large deviation - significant bias detected")
    
    if comparison.wasserstein > 0.5:
        lines.append("🎯 Wasserstein suggests concentrated bias on specific numbers")
    
    lines.append("═" * 50)
    
    return "\n".join(lines)


def format_anomaly_report(anomalies: Dict[str, Any]) -> str:
    """Format anomaly detection results for display."""
    lines = [
        "═" * 50,
        "🔍 ANOMALY DETECTION REPORT",
        "═" * 50,
        f"Threshold: {anomalies['threshold_sigma']}σ",
        f"Total anomalies found: {anomalies['total_anomalies']}",
        "",
    ]
    
    if anomalies["overrepresented"]:
        lines.append("🔥 OVERREPRESENTED (hot numbers):")
        lines.append("-" * 40)
        for r in anomalies["overrepresented"][:10]:
            lines.append(
                f"  #{r.number:2d}: {r.observed_prob*100:.2f}% "
                f"(expected {r.expected_prob*100:.2f}%) "
                f"[{r.deviation_sigma:+.2f}σ]"
            )
    
    if anomalies["underrepresented"]:
        lines.append("")
        lines.append("❄️  UNDERREPRESENTED (cold numbers):")
        lines.append("-" * 40)
        for r in anomalies["underrepresented"][:10]:
            lines.append(
                f"  #{r.number:2d}: {r.observed_prob*100:.2f}% "
                f"(expected {r.expected_prob*100:.2f}%) "
                f"[{r.deviation_sigma:+.2f}σ]"
            )
    
    lines.append("═" * 50)
    
    return "\n".join(lines)
