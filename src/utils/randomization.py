import numpy as np
from scipy import stats


def categorical_goodness_of_fit(counts, probabilities, resamples=9999, seed=42):
    counts = np.asarray(counts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (counts.ndim != 1 or counts.shape != probabilities.shape or len(counts) < 2
            or not np.all(np.isfinite(counts)) or np.any(counts < 0)
            or np.any(counts != np.floor(counts)) or counts.sum() <= 0
            or not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0)
            or not np.isclose(probabilities.sum(), 1) or resamples < 1):
        raise ValueError('Goodness of fit requires counts, a positive probability vector and resamples')
    probabilities = probabilities / probabilities.sum()
    expected = counts.sum() * probabilities
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    if np.all(expected >= 5):
        return statistic, float(stats.chi2.sf(statistic, len(counts) - 1)), 'chi-square asymptotic'
    rng = np.random.default_rng(seed)
    simulated = rng.multinomial(int(counts.sum()), probabilities, size=resamples)
    values = np.sum((simulated - expected) ** 2 / expected, axis=1)
    return statistic, float((1 + np.sum(values >= statistic - 1e-12)) / (resamples + 1)), 'multinomial Monte Carlo'


def uniformity_test(numbers, resamples=9999, seed=42):
    from src.settlement import validate_number
    counts = np.bincount([validate_number(number) for number in numbers], minlength=37)
    return categorical_goodness_of_fit(counts, np.full(37, 1 / 37), resamples, seed)


def permutation_pvalue(numbers, statistic, resamples=9999, seed=42):
    if resamples < 1:
        raise ValueError('At least one resample is required')
    array = np.asarray(numbers, dtype=np.int64)
    observed = float(statistic(array))
    if observed == 0:
        return observed, 1.0
    rng = np.random.default_rng(seed)
    greater = sum(float(statistic(rng.permutation(array))) >= observed - 1e-12 for _ in range(resamples))
    return observed, (greater + 1) / (resamples + 1)


def transition_statistic(numbers, lag=1):
    numbers = np.asarray(numbers, dtype=np.int64)
    matrix = np.bincount(numbers[:-lag] * 37 + numbers[lag:], minlength=37 * 37).reshape(37, 37)
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / matrix.sum()
    active = expected > 0
    return float(np.sum((matrix[active] - expected[active]) ** 2 / expected[active]))
