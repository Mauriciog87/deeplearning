from dataclasses import dataclass, field
from math import erfc, log, sqrt
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from src.utils.multiple_testing import false_discovery_control
from src.utils.randomization import uniformity_test, permutation_pvalue, transition_statistic
from src.settlement import validate_number


ROULETTE_NUMBERS = 37
FAIR_PROBABILITY = 1.0 / ROULETTE_NUMBERS
RED_NUMBERS = {
    1, 3, 5, 7, 9, 12, 14, 16, 18,
    19, 21, 23, 25, 27, 30, 32, 34, 36,
}


@dataclass
class RandomnessTestResult:
    name: str
    statistic: float
    p_value: Optional[float]
    status: str
    q_value: Optional[float] = None
    fdr_significant: bool = False
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class RandomnessReport:
    total_spins: int
    valid_spins: int
    ignored_spins: int
    alpha: float
    tests: List[RandomnessTestResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def analyze_randomness(
    numbers: List[int],
    window_size: int = 100,
    step_size: int = 50,
    alpha: float = 0.01,
    markov_max_lag: int = 3,
    resamples: int = 9999,
    seed: int = 42,
    fdr_method: str = 'by',
) -> RandomnessReport:
    valid_numbers = [validate_number(number) for number in numbers]
    if resamples < 1 or window_size < 1 or step_size < 1 or markov_max_lag < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid randomness test configuration')
    report = RandomnessReport(
        total_spins=len(numbers),
        valid_spins=len(valid_numbers),
        ignored_spins=len(numbers) - len(valid_numbers),
        alpha=alpha,
    )

    if report.ignored_spins:
        report.warnings.append("Invalid roulette numbers were ignored.")
    if len(valid_numbers) < 50:
        report.warnings.append("At least 50 valid spins are recommended for randomness tests.")

    report.tests.append(_uniformity_test(valid_numbers, alpha, resamples, seed))
    report.tests.extend(_runs_tests(valid_numbers, alpha))
    report.tests.append(_serial_correlation_test(valid_numbers, alpha, resamples, seed))
    report.tests.append(_transition_test(valid_numbers, alpha, resamples, seed))
    report.tests.append(_markov_lag_scan(valid_numbers, markov_max_lag, alpha, resamples, seed))
    report.tests.append(_categorical_change_point_test(valid_numbers, window_size, step_size, alpha, resamples, seed))
    report.tests.append(_entropy_drift_test(valid_numbers, window_size, step_size, alpha, resamples, seed))
    _apply_fdr(report.tests, alpha, fdr_method)
    report.warnings.append(f'Resampling seed={seed}, resamples={resamples}; FDR={fdr_method.upper()}. Permutations assume exchangeability (IID outcomes under the null).')
    dependence = any(result.fdr_significant for result in report.tests
                     if result.name in ('serial_correlation', 'transition_chi_square', 'markov_lag_scan'))
    if dependence:
        report.warnings.append('Dependence detected: drift is exploratory; IID permutations do not establish significance under general stationary dependence.')
        for result in report.tests:
            if result.name in ('categorical_change_point', 'entropy_drift') and result.p_value is not None:
                result.details['iid_permutation_p'] = result.p_value
                result.p_value = result.q_value = None
                result.fdr_significant = False
                result.status = 'exploratory'

    return report


def format_randomness_report(report: RandomnessReport) -> str:
    lines = [
        "=" * 86,
        "ROULETTE RANDOMNESS TEST REPORT",
        "=" * 86,
        f"Spins: total={report.total_spins}, valid={report.valid_spins}, ignored={report.ignored_spins}",
        f"Alpha: {report.alpha:.4f}",
        "",
        f"{'Test':<30} {'Statistic':>12} {'p-value':>12} {'q-value':>12} {'Status':>10}",
        "-" * 86,
    ]

    for result in report.tests:
        p_value = "n/a" if result.p_value is None else f"{result.p_value:.6f}"
        q_value = "n/a" if result.q_value is None else f"{result.q_value:.6f}"
        lines.append(
            f"{result.name:<30} "
            f"{result.statistic:>12.4f} "
            f"{p_value:>12} "
            f"{q_value:>12} "
            f"{result.status:>10}"
        )

    detail_lines = _format_details(report.tests)
    if detail_lines:
        lines.extend(["", "Details:"])
        lines.extend(detail_lines)

    if report.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in report.warnings)

    lines.extend([
        "",
        "Interpretation: low p-values flag evidence against fair independent roulette outcomes.",
        "Passing these tests does not prove randomness; it only means these tests did not reject it.",
    ])
    return "\n".join(lines)


def _uniformity_test(numbers, alpha, resamples=9999, seed=42):
    if not numbers:
        return _skipped('uniformity_chi_square')
    statistic, p_value, method = uniformity_test(numbers, resamples, seed)
    return _result('uniformity_chi_square', statistic, p_value, alpha,
                   {'method': method, 'resamples': resamples if len(numbers) < 185 else 0, 'seed': seed})


def _runs_tests(numbers: List[int], alpha: float) -> List[RandomnessTestResult]:
    definitions = [
        ("runs_red_black", _red_black_labels(numbers)),
        ("runs_odd_even", _odd_even_labels(numbers)),
        ("runs_low_high", _low_high_labels(numbers)),
    ]
    return [_runs_test(name, labels, alpha) for name, labels in definitions]


def _runs_test(name: str, labels: List[int], alpha: float) -> RandomnessTestResult:
    if len(labels) < 20:
        return _skipped(name)
    zeros = labels.count(0)
    ones = labels.count(1)
    if zeros == 0 or ones == 0:
        return _skipped(name)

    runs = 1 + sum(1 for index in range(1, len(labels)) if labels[index] != labels[index - 1])
    total = zeros + ones
    expected = 1.0 + (2.0 * zeros * ones) / total
    variance = (
        2.0 * zeros * ones * (2.0 * zeros * ones - total)
    ) / ((total ** 2) * (total - 1.0))
    if variance <= 0.0:
        return _skipped(name)
    z_score = (runs - expected) / sqrt(variance)
    p_value = erfc(abs(z_score) / sqrt(2.0))
    return _result(
        name,
        float(z_score),
        float(p_value),
        alpha,
        {
            "runs": float(runs),
            "expected_runs": float(expected),
            "zeros": float(zeros),
            "ones": float(ones),
        },
    )


def _serial_correlation_test(numbers, alpha, resamples=9999, seed=42):
    if len(numbers) < 4 or np.std(numbers[:-1]) == 0 or np.std(numbers[1:]) == 0:
        return _skipped('serial_correlation')
    def statistic(values):
        left, right = values[:-1], values[1:]
        if np.std(left) == 0 or np.std(right) == 0:
            return 0.0
        return abs(float(np.corrcoef(left, right)[0, 1]))
    measured, p_value = permutation_pvalue(numbers, statistic, resamples, seed)
    return _result('serial_correlation', measured, p_value, alpha, {'pairs': len(numbers) - 1, 'resamples': resamples, 'seed': seed})


def _transition_test(numbers, alpha, resamples=9999, seed=42):
    if len(numbers) < 100 or len(set(numbers)) < 2:
        return _skipped('transition_chi_square')
    measured, p_value = permutation_pvalue(numbers, transition_statistic, resamples, seed)
    return _result('transition_chi_square', measured, p_value, alpha,
                   {'transitions': len(numbers) - 1, 'resamples': resamples, 'seed': seed})


def _markov_lag_scan(numbers, max_lag, alpha, resamples=9999, seed=42):
    if len(numbers) < 500 or max_lag <= 0 or len(set(numbers)) < 2:
        return _skipped('markov_lag_scan')
    lags = range(1, min(max_lag + 1, len(numbers)))
    def statistic(values):
        return max(transition_statistic(values, lag) for lag in lags)
    measured, p_value = permutation_pvalue(numbers, statistic, resamples, seed)
    best_lag = max(lags, key=lambda lag: transition_statistic(numbers, lag))
    return _result('markov_lag_scan', measured, p_value, alpha,
                   {'best_lag': best_lag, 'lags_tested': len(lags), 'resamples': resamples, 'seed': seed})


def _change_statistics(numbers, window_size, step_size):
    candidates = []
    for split in range(window_size, len(numbers) - window_size + 1, step_size):
        left = np.bincount(numbers[split - window_size:split], minlength=37).astype(float)
        right = np.bincount(numbers[split:split + window_size], minlength=37).astype(float)
        total = left + right
        active = total > 0
        candidates.append((split, float(np.sum((left[active] - right[active]) ** 2 / total[active]))))
    return candidates


def _categorical_change_point_test(numbers, window_size, step_size, alpha, resamples=9999, seed=42):
    if window_size <= 0 or step_size <= 0 or len(numbers) < 2 * window_size:
        return _skipped('categorical_change_point')
    candidates = _change_statistics(numbers, window_size, step_size)
    best_split, _ = max(candidates, key=lambda item: item[1])
    def statistic(values):
        return max(value for _, value in _change_statistics(values, window_size, step_size))
    measured, p_value = permutation_pvalue(numbers, statistic, resamples, seed)
    return _result('categorical_change_point', measured, p_value, alpha,
                   {'best_split': best_split, 'splits_tested': len(candidates), 'resamples': resamples, 'seed': seed})


def _entropy_drift_test(numbers, window_size, step_size, alpha, resamples=9999, seed=42):
    windows = _rolling_windows(numbers, window_size, step_size)
    if len(windows) < 2:
        return _skipped('entropy_drift')
    entropies = [_normalized_entropy(window) for window in windows]
    def statistic(values):
        entropy = [_normalized_entropy(window) for window in _rolling_windows(values, window_size, step_size)]
        return max(entropy) - min(entropy)
    measured, p_value = permutation_pvalue(numbers, statistic, resamples, seed)
    return _result('entropy_drift', measured, p_value, alpha,
                   {'windows': len(windows), 'min_entropy': min(entropies), 'max_entropy': max(entropies),
                    'resamples': resamples, 'seed': seed})


def _rolling_windows(numbers: List[int], window_size: int, step_size: int) -> List[List[int]]:
    if window_size <= 0 or step_size <= 0 or len(numbers) < window_size:
        return []
    return [
        numbers[start:start + window_size]
        for start in range(0, len(numbers) - window_size + 1, step_size)
    ]


def _normalized_entropy(numbers: List[int]) -> float:
    counts = np.bincount(numbers, minlength=ROULETTE_NUMBERS).astype(float)
    probabilities = counts[counts > 0.0] / len(numbers)
    return float(-np.sum(probabilities * np.log(probabilities)) / log(ROULETTE_NUMBERS))


def _red_black_labels(numbers: List[int]) -> List[int]:
    return [1 if number in RED_NUMBERS else 0 for number in numbers if number != 0]


def _odd_even_labels(numbers: List[int]) -> List[int]:
    return [number % 2 for number in numbers if number != 0]


def _low_high_labels(numbers: List[int]) -> List[int]:
    return [0 if number <= 18 else 1 for number in numbers if number != 0]


def _result(
    name: str,
    statistic: float,
    p_value: Optional[float],
    alpha: float,
    details: Optional[Dict[str, float]] = None,
) -> RandomnessTestResult:
    status = "skip" if p_value is None else ("reject" if p_value < alpha else "pass")
    return RandomnessTestResult(
        name=name,
        statistic=statistic,
        p_value=p_value,
        status=status,
        details=details or {},
    )


def _skipped(name: str) -> RandomnessTestResult:
    return RandomnessTestResult(name=name, statistic=0.0, p_value=None, status="skip")


def _apply_fdr(results: List[RandomnessTestResult], alpha: float, method='by') -> None:
    q_values, rejected = false_discovery_control(
        [result.p_value for result in results],
        alpha, method=method,
    )
    for result, q_value, is_rejected in zip(results, q_values, rejected):
        result.q_value = q_value
        result.fdr_significant = is_rejected
        if result.p_value is None:
            result.status = "skip"
        elif is_rejected:
            result.status = "reject"
        else:
            result.status = "pass"


def _format_details(results: List[RandomnessTestResult]) -> List[str]:
    lines = []
    for result in results:
        if not result.details:
            continue
        values = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, (int, float)) else f"{key}={value}"
            for key, value in sorted(result.details.items())
        )
        lines.append(f"- {result.name}: {values}")
    return lines
