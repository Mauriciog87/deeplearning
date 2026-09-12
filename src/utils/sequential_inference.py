from dataclasses import dataclass
import json
from math import exp, isfinite, log, log1p
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, xlogy

from src.checkpoints import atomic_save
from src.probabilities import validate_probabilities
from src.settlement import validate_number


@dataclass(frozen=True)
class MonitoringBudget:
    alpha: float
    streams: tuple

    def __post_init__(self):
        if not isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError('The monitoring error budget must be in (0, 1)')
        if not self.streams or any(not isinstance(stream, str) or not stream for stream in self.streams):
            raise ValueError('Monitoring streams must have nonempty string identifiers')
        if len(set(self.streams)) != len(self.streams):
            raise ValueError('Monitoring stream identifiers must be unique')
        object.__setattr__(self, 'streams', tuple(self.streams))

    def allocation(self, stream, restart=0):
        if stream not in self.streams:
            raise ValueError('The stream is not part of the declared monitoring family')
        if isinstance(restart, bool) or not isinstance(restart, Integral) or restart < 0:
            raise ValueError('Restart index must be a nonnegative integer')
        return self.alpha / len(self.streams) / (restart + 1) / (restart + 2)


class MultinomialMonitor:
    def __init__(self, stream_id, *, alpha=.05, prior_strengths=(18.5, 370.0, 3700.0), null_probabilities=None):
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError('A nonempty stream identifier is required')
        if not isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('The monitor alpha must be in (0, 1)')
        if not prior_strengths or any(isinstance(value, bool) or not isfinite(value) or value <= 0 for value in prior_strengths):
            raise ValueError('Dirichlet prior strengths must be positive and finite')
        null = validate_probabilities(np.full(37, 1 / 37) if null_probabilities is None else null_probabilities)
        if np.any(null <= 0):
            raise ValueError('The declared multinomial null must assign positive probability to every number')
        self.stream_id = stream_id
        self.alpha = float(alpha)
        self.prior_strengths = tuple(float(value) for value in prior_strengths)
        self.null_probabilities = null.copy()
        self.counts = np.zeros(37, dtype=np.int64)
        self._prior = np.outer(self.prior_strengths, null)
        self._log_marginals = np.zeros(len(self.prior_strengths))
        self._null_log_likelihood = 0.0
        self._events = {}
        self._max_log_evidence = 0.0
        self._alarm_at = None

    @property
    def observations(self):
        return len(self._events)

    @property
    def log_marginal(self):
        return float(logsumexp(self._log_marginals) - log(len(self.prior_strengths)))

    @property
    def log_evidence(self):
        return self.log_marginal - self._null_log_likelihood

    def update(self, number, *, event_id):
        number = validate_number(number)
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('A nonempty event ID is required for sequential inference')
        if event_id in self._events:
            if self._events[event_id] != number:
                raise ValueError('The event ID was already recorded with a different outcome')
            return False
        self._log_marginals += np.log(self.counts[number] + self._prior[:, number]) - np.log(
            self.observations + np.asarray(self.prior_strengths))
        self._null_log_likelihood += log(self.null_probabilities[number])
        self.counts[number] += 1
        self._events[event_id] = number
        self._max_log_evidence = max(self._max_log_evidence, self.log_evidence)
        if self._alarm_at is None and self._max_log_evidence >= -log(self.alpha):
            self._alarm_at = self.observations
        return True

    def probability_bounds(self, numbers):
        selected = sorted({validate_number(number) for number in numbers})
        if not selected:
            return 0.0, 0.0
        if len(selected) == 37:
            return 1.0, 1.0
        count = self.observations
        if not count:
            return 0.0, 1.0
        successes = int(self.counts[selected].sum())
        failures = count - successes
        offset = float(np.sum(xlogy(self.counts, self.counts))
                       - xlogy(successes, successes) - xlogy(failures, failures))
        target = self.log_marginal + log(self.alpha) - offset
        if successes == 0:
            return 0.0, -float(np.expm1(target / failures))
        if failures == 0:
            return exp(target / successes), 1.0
        mode = successes / count

        def log_likelihood(value):
            return successes * log(value) + failures * log1p(-value)

        left, right = 0.0, mode
        for _ in range(60):
            middle = (left + right) / 2
            if log_likelihood(middle) < target:
                left = middle
            else:
                right = middle
        lower = left
        left, right = mode, 1.0
        for _ in range(60):
            middle = (left + right) / 2
            if middle == 1.0 or log_likelihood(middle) < target:
                right = middle
            else:
                left = middle
        return lower, right

    def snapshot(self):
        frequency = self.counts / self.observations if self.observations else self.null_probabilities
        return {
            'method': 'dirichlet_mixture_multinomial_v1', 'stream_id': self.stream_id,
            'observations': self.observations, 'allocated_alpha': self.alpha,
            'log_evidence': self.log_evidence, 'max_log_evidence': self._max_log_evidence,
            'anytime_p_value': exp(-self._max_log_evidence), 'rejected': self._alarm_at is not None,
            'alarm_at': self._alarm_at, 'observed_total_variation': float(np.abs(frequency - self.null_probabilities).sum() / 2),
            'null': 'The conditional outcome probabilities equal the declared null at every observation',
            'confidence_target': 'A fixed conditional probability vector; projections are simultaneous over coordinates, subsets and time',
            'interpretation': 'Rejection is evidence against the declared null, not proof of profitable or persistent bias. Non-rejection does not establish fairness.',
        }

    def to_state(self):
        return {
            'schema_version': 1, 'method': 'dirichlet_mixture_multinomial_v1',
            'stream_id': self.stream_id, 'alpha': self.alpha, 'prior_strengths': list(self.prior_strengths),
            'null_probabilities': self.null_probabilities.tolist(), 'events': list(self._events.items()),
        }

    @classmethod
    def from_state(cls, state, *, stream_id):
        if state.get('schema_version') != 1 or state.get('method') != 'dirichlet_mixture_multinomial_v1':
            raise ValueError('Unsupported sequential monitor state')
        if state.get('stream_id') != stream_id:
            raise ValueError('The saved monitor belongs to a different stream')
        monitor = cls(stream_id, alpha=state['alpha'], prior_strengths=state['prior_strengths'],
                      null_probabilities=state['null_probabilities'])
        saved_null = np.asarray(state['null_probabilities'], dtype=float)
        if not np.isclose(saved_null.sum(), 1, rtol=0, atol=1e-12):
            raise ValueError('The saved null distribution is not normalized')
        monitor.null_probabilities = saved_null.copy()
        monitor._prior = np.outer(monitor.prior_strengths, saved_null)
        for event_id, number in state['events']:
            if not monitor.update(number, event_id=event_id):
                raise ValueError('The saved monitor contains a duplicate event')
        return monitor

    def save(self, path):
        atomic_save(path, self.to_state(), neural=False)

    @classmethod
    def load(cls, path, *, stream_id):
        return cls.from_state(json.loads(Path(path).read_text(encoding='utf-8')), stream_id=stream_id)
