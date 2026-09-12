from dataclasses import dataclass
from math import log, sqrt
from numbers import Integral
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CalibrationBound:
    point: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    allocated_alpha: Optional[float] = None
    observations: int = 0
    runs: int = 0
    bins: int = 0
    occupied_bins_average: float = 0.0
    status: str = 'unavailable'
    reason: str = ''
    method: str = 'hilbert_azuma_dyadic_v1'
    target: str = 'Mean across runs and channels of fixed-bin absolute cumulative conditional residuals'


def hilbert_mean_radius(count, alpha, increment_bound=1.0):
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError('Observation count must be a positive integer')
    if not 0 < alpha < 1 or not np.isfinite(increment_bound) or increment_bound < 0:
        raise ValueError('Alpha must be in (0, 1) and the increment bound must be finite and nonnegative')
    epoch = (int(count) - 1).bit_length()
    horizon = 2 ** epoch
    log_threshold = log(2) + log(epoch + 1) + log(epoch + 2) - log(alpha)
    return increment_bound * sqrt(2 * horizon * log_threshold) / count


def fixed_bin_calibration_bound(probabilities, outcomes, *, bins=10, alpha=.05, categorical=False):
    predicted = np.asarray(probabilities, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if predicted.ndim != 3 or predicted.shape != observed.shape or any(size < 1 for size in predicted.shape):
        raise ValueError('Calibration arrays must have matching nonempty (run, observation, channel) shapes')
    if isinstance(bins, bool) or not isinstance(bins, Integral) or not 1 <= bins <= 10:
        raise ValueError('Calibration bin count must be an integer between 1 and 10')
    if not np.isfinite(predicted).all() or np.any(predicted < -1e-12) or np.any(predicted > 1 + 1e-12):
        raise ValueError('Predicted probabilities must be finite and between zero and one')
    if not np.isin(observed, (0, 1)).all():
        raise ValueError('Calibration outcomes must be binary indicators')
    predicted = np.clip(predicted, 0, 1)
    runs, count, channels = predicted.shape
    if categorical and (channels < 2 or not np.allclose(predicted.sum(axis=2), 1, atol=1e-12, rtol=0)
                        or not np.all(observed.sum(axis=2) == 1)
                        or not np.all(observed == observed[0])):
        raise ValueError('Categorical channels must form distributions and share one-hot outcomes across runs')
    increment_bound = sqrt(2 / channels) if categorical else 1.0
    maximum = 2 / channels if categorical else 1.0
    radius = hilbert_mean_radius(count, alpha, increment_bound)
    indices = np.minimum((predicted * bins).astype(int), bins - 1)
    errors, occupied = [], []
    for run in range(runs):
        for channel in range(channels):
            selected = indices[run, :, channel]
            residuals = observed[run, :, channel] - predicted[run, :, channel]
            totals = np.bincount(selected, weights=residuals, minlength=bins)
            errors.append(float(np.abs(totals).sum() / count))
            occupied.append(len(np.unique(selected)))
    point = float(np.mean(errors))
    occupied_average = float(np.mean(occupied))
    width = sqrt(occupied_average) * radius
    return CalibrationBound(
        point=point, lower=max(0.0, point - width), upper=min(maximum, point + width),
        allocated_alpha=alpha, observations=count, runs=runs, bins=bins,
        occupied_bins_average=occupied_average, status='available',
    )
