from math import exp, log
from numbers import Integral
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from src.probabilities import validate_probabilities
from src.settlement import validate_number


class JointCalibrationMonitor:
    def __init__(self, *, runs=1, resolution=10, prior_strength=10.0, alpha=.05):
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in (runs, resolution)):
            raise ValueError('Runs and resolution must be positive integers')
        if not np.isfinite(prior_strength) or prior_strength <= 0 or not 0 < alpha < 1:
            raise ValueError('Prior strength must be positive and alpha must lie within (0,1)')
        self.runs, self.resolution = int(runs), int(resolution)
        self.prior_strength, self.alpha = float(prior_strength), float(alpha)
        self._cells = [{} for _ in range(self.runs)]
        self._log_evidence = np.zeros(self.runs)
        self._max_log_evidence = 0.0
        self._support_violation = False
        self._alarm_at = None
        self._events = {}

    @property
    def observations(self):
        return len(self._events)

    @property
    def log_evidence(self):
        return float(logsumexp(self._log_evidence) - log(self.runs))

    def update(self, probabilities, actual, *, event_id):
        actual = validate_number(actual)
        predicted = np.asarray(probabilities, dtype=float)
        if predicted.shape != (self.runs, 37):
            raise ValueError('Expected one 37-class vector per configured run')
        normalized = []
        for vector in predicted:
            validated = validate_probabilities(vector)
            normalized.append(vector.copy() if np.isclose(vector.sum(), 1, rtol=0, atol=1e-12) else validated)
        predicted = np.asarray(normalized)
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('A nonempty event ID is required')
        record = (actual, predicted.tolist())
        if event_id in self._events:
            if self._events[event_id] != record:
                raise ValueError('An event was already recorded with a different forecast or outcome')
            return False
        for run, vector in enumerate(predicted):
            cell = tuple(np.floor(vector * self.resolution).astype(int))
            counts, forecast_sum = self._cells[run].setdefault(cell, (np.zeros(37), np.zeros(37)))
            if vector[actual] == 0:
                self._support_violation = True
            elif not self._support_violation:
                log_ratio = log(counts[actual]) - log(vector[actual]) if counts[actual] else -np.inf
                self._log_evidence[run] += float(np.logaddexp(log_ratio, log(self.prior_strength))) - log(counts.sum() + self.prior_strength)
            counts[actual] += 1
            forecast_sum += vector
        self._events[event_id] = record
        if not self._support_violation:
            self._max_log_evidence = max(self._max_log_evidence, self.log_evidence)
        if self._alarm_at is None and (self._support_violation or self._max_log_evidence >= -log(self.alpha)):
            self._alarm_at = self.observations
        return True

    def snapshot(self):
        error = sum(np.abs(counts - forecast_sum).sum() for cells in self._cells for counts, forecast_sum in cells.values())
        return {
            'status': 'ready', 'method': 'predictable_joint_cell_likelihood_v1',
            'observations': self.observations, 'runs': self.runs, 'allocated_alpha': self.alpha,
            'resolution': self.resolution, 'prior_strength': self.prior_strength,
            'joint_cell_total_variation': float(error / (2 * self.runs * self.observations)) if self.observations else None,
            'log_evidence': None if self._support_violation else self.log_evidence,
            'max_log_evidence': None if self._support_violation else self._max_log_evidence,
            'anytime_p_value': 0.0 if self._support_violation else exp(-self._max_log_evidence),
            'support_violation': self._support_violation, 'rejected': self._alarm_at is not None, 'alarm_at': self._alarm_at,
            'null': 'For every configured run, the announced full vector equals the outcome distribution conditional on the monitored past and current forecasts',
            'aggregation': 'Arithmetic mean of run likelihood processes on shared outcomes, never their product',
            'interpretation': 'The null is stronger than marginal or classwise calibration and stronger than E[Y|forecast]=forecast without conditioning on the past. Joint-cell error is descriptive. Non-rejection does not certify calibration; power depends on the fixed cells and alternative.',
        }

    def to_state(self):
        return {'schema_version': 1, 'method': 'predictable_joint_cell_likelihood_v1', 'runs': self.runs,
                'resolution': self.resolution, 'prior_strength': self.prior_strength, 'alpha': self.alpha,
                'events': [(event_id, actual, deepcopy(probabilities)) for event_id, (actual, probabilities) in self._events.items()]}

    @classmethod
    def from_state(cls, state):
        if state.get('schema_version') != 1 or state.get('method') != 'predictable_joint_cell_likelihood_v1':
            raise ValueError('Unsupported joint calibration state')
        monitor = cls(runs=state['runs'], resolution=state['resolution'], prior_strength=state['prior_strength'], alpha=state['alpha'])
        for event_id, actual, probabilities in state['events']:
            if not monitor.update(probabilities, actual, event_id=event_id):
                raise ValueError('Saved events must be unique')
        return monitor
