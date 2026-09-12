from copy import deepcopy
from math import isfinite, sqrt
from numbers import Integral

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from src.probabilities import validate_probabilities
from src.settlement import validate_number


class OnlineRecalibrator:
    def __init__(self, *, bandwidth=.35, max_iterations=100):
        if not isfinite(bandwidth) or not 1e-6 <= bandwidth <= 1e6:
            raise ValueError('Kernel bandwidth must lie within [1e-6,1e6]')
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
            raise ValueError('Iteration limit must be a positive integer')
        self.bandwidth, self.max_iterations = float(bandwidth), int(max_iterations)
        self.anchors = np.vstack((np.full(37, 1 / 37), np.eye(37)))
        self._calibration_sum = np.zeros((38, 37))
        self._regret_sum = 0.0
        self._weighted_residual = 0.0
        self._events = {}
        self._pending = None
        self._last_forecast = np.full(37, 1 / 37)

    def _features(self, probability):
        weights = softmax(-np.sum((self.anchors - probability) ** 2, axis=1) / (2 * self.bandwidth ** 2))
        derivative = weights[:, None] * (self.anchors - weights @ self.anchors) / self.bandwidth ** 2
        return weights, derivative

    def _halfspace(self, probability, baseline):
        count = len(self._events)
        calibration = self._calibration_sum / max(1, count)
        regret = max(0.0, self._regret_sum / max(1, count))
        features, feature_gradient = self._features(probability)
        weighted = features @ calibration
        derivative = calibration.T @ feature_gradient
        values = weighted - np.dot(weighted, probability)
        gradients = derivative - (derivative.T @ probability + weighted)[None, :]
        values += regret * (.5 * (np.dot(probability, probability) - np.dot(baseline, baseline)) - probability + baseline)
        gradients += regret * (probability[None, :] - np.eye(37))
        return values, gradients

    def forecast(self, baseline, *, event_id):
        vector = np.asarray(baseline, dtype=float)
        checked = validate_probabilities(vector)
        vector = vector.copy() if np.isclose(vector.sum(), 1, rtol=0, atol=1e-12) else checked
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('A nonempty event ID is required')
        if event_id in self._events:
            previous = self._events[event_id]['decision']
            if previous['baseline'] != vector.tolist():
                raise ValueError('The event already has a different baseline forecast')
            return deepcopy(previous)
        if self._pending is not None:
            if self._pending['event_id'] != event_id or self._pending['baseline'] != vector.tolist():
                raise ValueError('Record the pending outcome before forecasting another event')
            return deepcopy(self._pending)
        starts = [vector, np.full(37, 1 / 37), self._last_forecast]
        best = vector
        upper = float(np.max(self._halfspace(best, vector)[0]))
        iterations, solver_success = 0, None
        if self._events:
            def constraints(parameters):
                values, gradients = self._halfspace(parameters[:37], vector)
                return parameters[-1] - values, np.column_stack((-gradients, np.ones(37)))

            for initial in starts:
                initial_upper = float(np.max(self._halfspace(initial, vector)[0]))
                if initial_upper < upper:
                    best, upper = initial.copy(), initial_upper
                result = minimize(lambda parameters: parameters[-1], np.append(initial, initial_upper),
                                  jac=lambda parameters: np.append(np.zeros(37), 1), method='SLSQP',
                                  bounds=[(0, 1)] * 37 + [(None, None)],
                                  constraints=[{'type': 'eq', 'fun': lambda parameters: parameters[:37].sum() - 1,
                                                'jac': lambda parameters: np.append(np.ones(37), 0)},
                                               {'type': 'ineq', 'fun': lambda parameters: constraints(parameters)[0],
                                                'jac': lambda parameters: constraints(parameters)[1]}],
                                  options={'maxiter': self.max_iterations, 'ftol': 1e-10})
                iterations += int(result.nit)
                candidate = np.clip(result.x[:37], 0, 1)
                if not np.isfinite(candidate).all() or candidate.sum() <= 0:
                    continue
                candidate /= candidate.sum()
                candidate_upper = float(np.max(self._halfspace(candidate, vector)[0]))
                if candidate_upper < upper:
                    best, upper, solver_success = candidate, candidate_upper, bool(result.success)
        self._pending = {'event_id': event_id, 'baseline': vector.tolist(), 'probabilities': best.tolist(),
                         'oracle_upper_bound': upper, 'oracle_nonpositive': upper <= 0,
                         'solver_success': solver_success, 'iterations': iterations,
                         'previous_observations': len(self._events)}
        return deepcopy(self._pending)

    def update(self, actual, *, event_id):
        actual = validate_number(actual)
        if event_id in self._events:
            if self._events[event_id]['actual'] != actual:
                raise ValueError('The event already has a different outcome')
            return False
        if self._pending is None or self._pending['event_id'] != event_id:
            raise ValueError('Forecast this event before recording its outcome')
        decision = self._pending
        probability, baseline = np.asarray(decision['probabilities']), np.asarray(decision['baseline'])
        outcome = np.eye(37)[actual]
        self._calibration_sum += self._features(probability)[0][:, None] * (outcome - probability)[None, :]
        self._regret_sum += float((np.sum((probability - outcome) ** 2) - np.sum((baseline - outcome) ** 2)) / 2)
        self._weighted_residual += len(self._events) * max(0.0, decision['oracle_upper_bound'])
        self._events[event_id] = {'actual': actual, 'decision': deepcopy(decision)}
        self._last_forecast = probability
        self._pending = None
        return True

    def snapshot(self):
        count = len(self._events)
        calibration = float(np.linalg.norm(self._calibration_sum) / count) if count else 0.0
        regret = self._regret_sum / count if count else 0.0
        return {'method': 'finite_basis_blackwell_recalibration_v1', 'observations': count,
                'bandwidth': self.bandwidth, 'features': len(self.anchors), 'calibration_residual_norm': calibration,
                'mean_brier_regret': regret, 'cone_distance': sqrt(calibration ** 2 + max(regret, 0) ** 2),
                'residual_bound': sqrt(3 / count + 2 * self._weighted_residual / count ** 2) if count else None,
                'nonpositive_oracle_steps': sum(record['decision']['oracle_nonpositive'] for record in self._events.values()),
                'maximum_positive_oracle_residual': max((max(0, record['decision']['oracle_upper_bound']) for record in self._events.values()), default=0),
                'pending_event_id': self._pending['event_id'] if self._pending else None,
                'score': 'Multiclass Brier divided by two, bounded in [0,1]',
                'interpretation': 'Experimental finite RBF calibration basis and bounded-score comparison with one baseline. All 37 adversarial outcomes are enumerated before observing the label. The numerical optimizer can miss the halfspace condition; its achieved residual enters the reported bound. No unconditional full-calibration or regret guarantee is claimed.',
                'source': 'https://arxiv.org/abs/2409.19157'}

    def to_state(self):
        return {'schema_version': 1, 'method': 'finite_basis_blackwell_recalibration_v1', 'bandwidth': self.bandwidth,
                'max_iterations': self.max_iterations, 'events': deepcopy(list(self._events.items())), 'pending': deepcopy(self._pending)}

    @classmethod
    def from_state(cls, state):
        if state.get('schema_version') != 1 or state.get('method') != 'finite_basis_blackwell_recalibration_v1':
            raise ValueError('Unsupported online recalibrator state')
        recalibrator = cls(bandwidth=state['bandwidth'], max_iterations=state['max_iterations'])
        for event_id, record in state['events']:
            recalibrator._restore_decision(record['decision'], event_id)
            if not recalibrator.update(record['actual'], event_id=event_id):
                raise ValueError('Saved events must be unique')
        if state['pending'] is not None:
            recalibrator._restore_decision(state['pending'], state['pending']['event_id'])
        return recalibrator

    def _restore_decision(self, decision, event_id):
        if not isinstance(event_id, str) or not event_id or event_id in self._events or decision['event_id'] != event_id:
            raise ValueError('Invalid saved event identity')
        if decision['previous_observations'] != len(self._events):
            raise ValueError('Saved decision has inconsistent chronological provenance')
        baseline, probability = np.asarray(decision['baseline']), np.asarray(decision['probabilities'])
        validate_probabilities(baseline)
        validate_probabilities(probability)
        bound = float(np.max(self._halfspace(probability, baseline)[0]))
        if not isfinite(decision['oracle_upper_bound']) or abs(bound - decision['oracle_upper_bound']) > 1e-10 or decision['oracle_nonpositive'] != (decision['oracle_upper_bound'] <= 0):
            raise ValueError('Saved oracle residual does not match its forecast and past')
        self._pending = deepcopy(decision)
