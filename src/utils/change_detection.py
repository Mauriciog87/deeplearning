from bisect import bisect_right, insort
from math import exp, isfinite, log, sqrt
from numbers import Integral

import numpy as np
from scipy.special import gammaln, logsumexp

from src.probabilities import validate_probabilities
from src.settlement import validate_number


def _new_event(events, event_id, value):
    if not isinstance(event_id, str) or not event_id:
        raise ValueError('A nonempty event ID is required')
    if event_id not in events:
        return True
    if events[event_id] != value:
        raise ValueError('The event ID already has a different observation')
    return False


def categorical_pit(probabilities, actual, randomization, *, order=tuple(range(37))):
    vector = validate_probabilities(probabilities)
    actual = validate_number(actual)
    if len(order) != 37 or {validate_number(number) for number in order} != set(range(37)):
        raise ValueError('PIT ordering must contain every pocket exactly once')
    if not isfinite(randomization) or not 0 <= randomization < 1:
        raise ValueError('PIT randomization must lie within [0,1)')
    position = list(order).index(actual)
    value = float(vector[list(order[:position])].sum() + randomization * vector[actual])
    return min(1.0, max(0.0, value))


class CategoricalEDetector:
    def __init__(self, *, average_run_length=1000.0, variant='sr', alternative_weight=.1, null_probabilities=None):
        if not isfinite(average_run_length) or average_run_length <= 1 or variant not in ('sr', 'cusum'):
            raise ValueError('Expected ARL greater than one and variant sr or cusum')
        if not isfinite(alternative_weight) or not 0 < alternative_weight < 1:
            raise ValueError('Alternative weight must lie within (0,1)')
        null = np.full(37, 1 / 37) if null_probabilities is None else np.asarray(null_probabilities, dtype=float)
        validate_probabilities(null)
        if np.any(null <= 0):
            raise ValueError('Every null probability must be positive')
        self.null = null.copy() if np.isclose(null.sum(), 1, rtol=0, atol=1e-12) else validate_probabilities(null)
        self.average_run_length, self.variant, self.alternative_weight = float(average_run_length), variant, float(alternative_weight)
        self._log_factors = np.log(((1 - alternative_weight) * self.null[None, :] + alternative_weight * np.eye(37)) / self.null[None, :])
        self._statistics = np.full(37, -np.inf)
        self._events = {}
        self._alarm_at = None

    def update(self, actual, *, event_id):
        actual = validate_number(actual)
        if not _new_event(self._events, event_id, actual):
            return False
        initial = np.logaddexp(self._statistics, 0) if self.variant == 'sr' else np.maximum(self._statistics, 0)
        self._statistics = initial + self._log_factors[:, actual]
        self._events[event_id] = actual
        if self._alarm_at is None and self.log_statistic >= log(self.average_run_length):
            self._alarm_at = len(self._events)
        return True

    @property
    def log_statistic(self):
        return float(logsumexp(self._statistics) - log(37))

    def snapshot(self):
        return {'method': f'categorical_mixture_e_{self.variant}_v1', 'observations': len(self._events),
                'log_statistic': self.log_statistic if self._events else None, 'alarm_at': self._alarm_at,
                'rejected': self._alarm_at is not None, 'average_run_length_lower_bound': self.average_run_length,
                'null': 'Each outcome has the declared conditional probability vector given the past',
                'interpretation': 'The null guarantee is average run length, not probability of ever alarming. Alternatives boost one pocket each; mixture weights are fixed before monitoring.',
                'source': 'https://arxiv.org/abs/2203.03532v4'}

    def to_state(self):
        return {'schema_version': 1, 'average_run_length': self.average_run_length, 'variant': self.variant,
                'alternative_weight': self.alternative_weight, 'null_probabilities': self.null.tolist(), 'events': list(self._events.items())}

    @classmethod
    def from_state(cls, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported e-detector state')
        detector = cls(**{key: state[key] for key in ('average_run_length', 'variant', 'alternative_weight', 'null_probabilities')})
        for event_id, actual in state['events']:
            if not detector.update(actual, event_id=event_id):
                raise ValueError('Saved events must be unique')
        return detector


class ReferenceConditionalMonitor:
    def __init__(self, reference, *, alpha=.05, reference_delta=None, seed=42, score_id='fixed_score'):
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
            raise ValueError('Reference scores must form a nonempty finite vector')
        delta = alpha / 2 if reference_delta is None else reference_delta
        if not 0 < delta < alpha < 1 or not isinstance(score_id, str) or not score_id:
            raise ValueError('Require 0 < reference delta < alpha < 1 and a score identity')
        self.reference = np.sort(reference)
        self.reference.setflags(write=False)
        self.alpha, self.reference_delta, self.seed, self.score_id = float(alpha), float(delta), seed, score_id
        self.epsilon = min(1.0, sqrt(log(2 / delta) / (2 * len(reference))))
        limit = min(1.0, .95 / (.5 + self.epsilon))
        self._bets = limit * np.asarray([-.999, -.5, -.25, 0, .25, .5, .999])
        self._wealth = np.zeros(len(self._bets))
        self._maximum = 0.0
        self._events = {}
        self._alarm_at = None
        self._rng = np.random.default_rng(seed)

    def update(self, score, *, event_id):
        if not isfinite(score):
            raise ValueError('Scores must be finite')
        score = float(score)
        if not _new_event(self._events, event_id, score):
            return False
        left, right = np.searchsorted(self.reference, score, side='left'), np.searchsorted(self.reference, score, side='right')
        empirical_pit = (left + self._rng.random() * (right - left)) / len(self.reference)
        factors = 1 + self._bets * (empirical_pit - .5) - np.abs(self._bets) * self.epsilon
        self._wealth += np.log(factors)
        self._events[event_id] = score
        self._maximum = max(self._maximum, float(logsumexp(self._wealth) - log(len(self._bets))))
        if self._alarm_at is None and self._maximum >= -log(self.alpha - self.reference_delta):
            self._alarm_at = len(self._events)
        return True

    def snapshot(self):
        return {'method': 'reference_conditional_linear_mixture_v1', 'observations': len(self._events),
                'reference_size': len(self.reference), 'score_id': self.score_id, 'dkw_epsilon': self.epsilon,
                'reference_failure_budget': self.reference_delta, 'conditional_monitoring_alpha': self.alpha - self.reference_delta,
                'total_error_bound': self.alpha, 'conditional_anytime_p_value': exp(-self._maximum),
                'rejected': self._alarm_at is not None, 'alarm_at': self._alarm_at,
                'null': 'A fixed score function applied to independent identically distributed reference and monitoring observations',
                'interpretation': 'On the DKW reference event, the monitoring error is conditionally controlled. The total bound includes reference failure. Randomized ECDF ties handle discrete scores. Power depends on a mean ECDF shift exceeding reference uncertainty.',
                'adaptation': 'Equation 7 with a fixed mixture of bounded bets and randomized ties; not the smoothed ONS update from the paper',
                'source': 'https://arxiv.org/abs/2602.13848v2'}

    def to_state(self):
        return {'schema_version': 1, 'reference': self.reference.tolist(), 'alpha': self.alpha, 'reference_delta': self.reference_delta,
                'seed': self.seed, 'score_id': self.score_id, 'events': list(self._events.items())}

    @classmethod
    def from_state(cls, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported reference monitor state')
        monitor = cls(**{key: state[key] for key in ('reference', 'alpha', 'reference_delta', 'seed', 'score_id')})
        for event_id, score in state['events']:
            if not monitor.update(score, event_id=event_id):
                raise ValueError('Saved events must be unique')
        return monitor


class PITMonitor:
    def __init__(self, *, alpha=.05, bins=10, seed=42):
        if not 0 < alpha < 1 or isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 2:
            raise ValueError('Alpha must lie in (0,1) and bins must be an integer of at least two')
        self.alpha, self.bins, self.seed = float(alpha), int(bins), seed
        self._rng = np.random.default_rng(seed)
        self._ordered = []
        self._counts = np.ones(self.bins)
        self._log_active = -np.inf
        self._maximum = 0.0
        self._events = {}
        self._p_values = []
        self._alarm_at = None
        self._location = None

    def update(self, pit, *, event_id):
        if not isfinite(pit) or not 0 <= pit <= 1:
            raise ValueError('PIT values must be finite and lie within [0,1]')
        pit = float(pit)
        if not _new_event(self._events, event_id, pit):
            return False
        time = len(self._events) + 1
        key = (pit, float(self._rng.random()))
        rank = bisect_right(self._ordered, key) + 1
        p_value = (rank - 1 + self._rng.random()) / time
        cell = min(self.bins - 1, int(p_value * self.bins))
        log_factor = log(self.bins * self._counts[cell] / self._counts.sum())
        log_weight = -log(time) - log(time + 1)
        self._log_active = log_factor + float(np.logaddexp(self._log_active, log_weight))
        self._maximum = max(self._maximum, self.log_evidence_at(time))
        self._counts[cell] += 1
        insort(self._ordered, key)
        self._p_values.append(float(p_value))
        self._events[event_id] = pit
        if self._alarm_at is None and self._maximum >= -log(self.alpha):
            self._alarm_at = time
            self._location = self._estimate_location()
        return True

    def log_evidence_at(self, time):
        return float(np.logaddexp(self._log_active, -log(time + 1)))

    def _estimate_location(self):
        counts = np.zeros(self.bins)
        best_score, best_split = -np.inf, None
        for split in range(len(self._p_values) - 1, 0, -1):
            counts[min(self.bins - 1, int(self._p_values[split] * self.bins))] += 1
            size = counts.sum()
            score = float(gammaln(self.bins / 2) - gammaln(size + self.bins / 2)
                          + np.sum(gammaln(counts + .5) - gammaln(.5)) + size * log(self.bins))
            if score >= best_score:
                best_score, best_split = score, split
        return {'estimated_boundary_after': best_split, 'segment_log_bayes_factor': best_score if best_split is not None else None,
                'interpretation': 'Post-alarm segment estimate against uniform sequential ranks with Jeffreys Dirichlet prior; not an alarm time or confidence interval'}

    def snapshot(self):
        count = len(self._events)
        return {'method': 'pit_rank_histogram_mixture_v1', 'observations': count, 'bins': self.bins,
                'log_evidence': self.log_evidence_at(count), 'max_log_evidence': self._maximum,
                'unstarted_weight': 1 / (count + 1), 'anytime_p_value': exp(-self._maximum),
                'allocated_alpha': self.alpha, 'alarm_at': self._alarm_at, 'rejected': self._alarm_at is not None,
                'changepoint': self._location,
                'null': 'The PIT sequence is IID; monitoring decisions use the sequential-rank filtration and independent randomization',
                'interpretation': 'Tests PIT distribution stability, not uniformity or deterioration alone. Calibration improvements can trigger alarms. Categorical PIT ordering must be fixed. Independent tie keys handle atoms.',
                'adaptation': 'Histogram bets and start weights 1/(t(t+1)); reports the full mixture including unstarted mass used in the paper proof',
                'source': 'https://arxiv.org/abs/2603.13156v1'}

    def to_state(self):
        return {'schema_version': 1, 'alpha': self.alpha, 'bins': self.bins, 'seed': self.seed, 'events': list(self._events.items())}

    @classmethod
    def from_state(cls, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported PIT monitor state')
        monitor = cls(alpha=state['alpha'], bins=state['bins'], seed=state['seed'])
        for event_id, pit in state['events']:
            if not monitor.update(pit, event_id=event_id):
                raise ValueError('Saved events must be unique')
        return monitor
