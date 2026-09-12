from copy import deepcopy
from math import fsum, isfinite
from numbers import Integral

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize, minimize_scalar
from scipy.special import logsumexp, softmax

from src.probabilities import validate_probabilities
from src.settlement import validate_number


CALIBRATORS = ('temperature', 'mcllo', 'normalized_isotonic')


def _probability_matrix(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 37 or not len(values):
        raise ValueError('Expected a nonempty matrix of 37-class probability vectors')
    normalized = []
    for row in values:
        validate_probabilities(row)
        total = fsum(row)
        normalized.append(row.copy() if abs(total - 1) <= 1e-12 else row / total)
    return np.asarray(normalized)


def normalized_isotonic_objective(log_values, assignments, actual):
    logits = np.asarray(log_values)[assignments]
    predicted = softmax(logits, axis=1)
    count = len(actual)
    objective = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(count), actual]))
    gradient = np.bincount(assignments.ravel(), weights=predicted.ravel(), minlength=len(log_values))
    gradient -= np.bincount(assignments[np.arange(count), actual], minlength=len(log_values))
    return objective, gradient / count


def monotone_log_gap(log_values, gradient, log_range=30.0):
    values, derivative = np.asarray(log_values), np.asarray(gradient)
    if values.ndim != 1 or not len(values) or values.shape != derivative.shape or not np.isfinite(values).all() or not np.isfinite(derivative).all():
        raise ValueError('Values and gradient must be matching finite vectors')
    if not isfinite(log_range) or log_range <= 0 or np.any(np.diff(values) < -1e-10) or values[0] < -log_range - 1e-10 or abs(values[-1]) > 1e-10:
        raise ValueError('Expected monotone log values within [-log_range,0] with the last value fixed to zero')
    best_prefix = max(0.0, float(np.max(np.cumsum(derivative)[:-1]))) if len(values) > 1 else 0.0
    return max(0.0, float(np.dot(derivative, values) + log_range * best_prefix))


class ProbabilityCalibrator:
    def __init__(self, method='temperature', *, probability_floor=1e-12, regularization=1e-3,
                 reference_class=36, max_iterations=500):
        if method not in CALIBRATORS:
            raise ValueError('Unknown probability calibrator')
        if not isfinite(probability_floor) or not 0 < probability_floor < 1 / 37:
            raise ValueError('Probability floor must lie within (0,1/37)')
        if not isfinite(regularization) or regularization < 0:
            raise ValueError('Regularization must be finite and nonnegative')
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
            raise ValueError('Iteration limit must be a positive integer')
        self.method = method
        self.probability_floor = probability_floor
        self.regularization = regularization
        self.reference_class = validate_number(reference_class)
        self.max_iterations = int(max_iterations)
        self.parameters = {}
        self.report = {}
        self.is_fitted = False

    def fit(self, probabilities, actual):
        matrix = _probability_matrix(probabilities)
        outcomes = np.asarray([validate_number(number) for number in actual])
        if outcomes.shape != (len(matrix),):
            raise ValueError('Each calibration vector needs exactly one outcome')
        logs = np.log(np.maximum(matrix, self.probability_floor))
        report = {'method': self.method, 'samples': len(matrix), 'probability_floor': self.probability_floor,
                  'reference_class': self.reference_class, 'experimental': True,
                  'interpretation': 'Fitted only on the declared calibration partition. Training loss or optimizer convergence does not establish held-out calibration or predictive improvement.'}
        if self.method == 'temperature':
            def objective(inverse_temperature):
                logits = inverse_temperature * logs
                return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(outcomes)), outcomes]))

            result = minimize_scalar(objective, bounds=(.02, 50), method='bounded', options={'maxiter': self.max_iterations, 'xatol': 1e-10})
            inverse = float(result.x) if result.success and result.fun <= objective(1) else 1.0
            self.parameters = {'inverse_temperature': inverse}
            report.update(optimization_success=bool(result.success), iterations=int(result.nfev),
                          initial_objective=objective(1), fitted_objective=objective(inverse),
                          parameter_bounds={'inverse_temperature': [.02, 50]}, parameter_count=1)
        elif self.method == 'mcllo':
            other = [number for number in range(37) if number != self.reference_class]
            odds = logs[:, other] - logs[:, self.reference_class, None]
            identity = np.concatenate((np.zeros(36), np.ones(36)))

            def objective(parameters):
                logits = np.zeros_like(matrix)
                logits[:, other] = parameters[:36] + odds * parameters[36:]
                predicted = softmax(logits, axis=1)
                loss = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(outcomes)), outcomes])
                residual = predicted - np.eye(37)[outcomes]
                gradient = np.concatenate((residual[:, other].mean(axis=0), (residual[:, other] * odds).mean(axis=0)))
                penalty = parameters - identity
                return float(loss + self.regularization * np.dot(penalty, penalty) / 2), gradient + self.regularization * penalty

            result = minimize(objective, identity, jac=True, method='L-BFGS-B',
                              bounds=[(-20, 20)] * 36 + [(-10, 10)] * 36,
                              options={'maxiter': self.max_iterations, 'ftol': 1e-12, 'gtol': 1e-7})
            parameters = result.x if np.isfinite(result.fun) and result.fun <= objective(identity)[0] else identity
            self.parameters = {'offsets': parameters[:36].tolist(), 'slopes': parameters[36:].tolist()}
            report.update(optimization_success=bool(result.success), iterations=int(result.nit),
                          initial_objective=objective(identity)[0], fitted_objective=objective(parameters)[0],
                          regularization=self.regularization, parameter_count=72,
                          parameter_bounds={'offsets': [-20, 20], 'slopes': [-10, 10]},
                          source='https://arxiv.org/abs/2602.18573',
                          adaptation='MCLLO log-odds map with a fixed reference class, bounded parameters and ridge penalty toward identity; not an unpenalized MLE replication')
        else:
            from sklearn.isotonic import IsotonicRegression

            initial = IsotonicRegression(out_of_bounds='clip').fit(matrix.ravel(), np.eye(37)[outcomes].ravel())
            starts = np.concatenate(([True], np.diff(initial.y_thresholds_) > 0))
            thresholds = initial.X_thresholds_[starts]
            initial_values = np.log(np.maximum(initial.y_thresholds_[starts], np.exp(-30)))
            initial_values = np.clip(initial_values - initial_values[-1], -30, 0)
            assignments = np.clip(np.searchsorted(thresholds, matrix, side='right') - 1, 0, len(thresholds) - 1)
            objective = lambda values: normalized_isotonic_objective(values, assignments, outcomes)
            success, iterations = True, 0
            fitted = initial_values
            if len(thresholds) > 1:
                difference = np.diff(np.eye(len(thresholds)), axis=0)
                lower = np.full(len(thresholds), -30.0)
                lower[-1] = 0
                result = minimize(objective, initial_values, method='SLSQP', jac=True,
                                  bounds=Bounds(lower, np.zeros(len(thresholds))),
                                  constraints=[LinearConstraint(difference, 0, np.inf)],
                                  options={'maxiter': self.max_iterations, 'ftol': 1e-11})
                candidate = np.clip(np.maximum.accumulate(result.x), -30, 0)
                candidate[-1] = 0
                if np.isfinite(result.fun) and objective(candidate)[0] <= objective(initial_values)[0]:
                    fitted = candidate
                success, iterations = bool(result.success), int(result.nit)
            self.parameters = {'thresholds': thresholds.tolist(), 'log_values': fitted.tolist(), 'log_range': 30.0}
            loss, gradient = objective(fitted)
            gap = monotone_log_gap(fitted, gradient)
            report.update(optimization_success=success, iterations=iterations,
                          initial_objective=objective(initial_values)[0], fitted_objective=loss,
                          objective_suboptimality_upper_bound=gap, optimization_certified=gap <= 1e-6,
                          parameter_count=max(0, len(thresholds) - 1), source='https://arxiv.org/abs/2512.09054',
                          adaptation='Equation 4 on fixed PAVA blocks, reparameterized by monotone log values in [-30,0]. Convex finite problem with a first-order optimality-gap bound; not the paper MCMC algorithm or unrestricted functional optimum')
        self.is_fitted = True
        logits = self._logits(matrix)
        report['uncalibrated_nll'] = float(-np.log(np.maximum(matrix[np.arange(len(outcomes)), outcomes], self.probability_floor)).mean())
        report['calibrated_nll'] = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(outcomes)), outcomes]))
        self.report = report
        return deepcopy(report)

    def transform(self, probabilities):
        if not self.is_fitted:
            raise RuntimeError('Calibrator has not been fitted')
        matrix = _probability_matrix(probabilities)
        return softmax(self._logits(matrix), axis=1)

    def _logits(self, matrix):
        logs = np.log(np.maximum(matrix, self.probability_floor))
        if self.method == 'temperature':
            logits = self.parameters['inverse_temperature'] * logs
        elif self.method == 'mcllo':
            other = [number for number in range(37) if number != self.reference_class]
            logits = np.zeros_like(matrix)
            logits[:, other] = np.asarray(self.parameters['offsets']) + np.asarray(self.parameters['slopes']) * (logs[:, other] - logs[:, self.reference_class, None])
        else:
            thresholds = np.asarray(self.parameters['thresholds'])
            indices = np.clip(np.searchsorted(thresholds, matrix, side='right') - 1, 0, len(thresholds) - 1)
            logits = np.asarray(self.parameters['log_values'])[indices]
        return logits

    def to_state(self):
        if not self.is_fitted:
            raise RuntimeError('Calibrator has not been fitted')
        return {'schema_version': 1, 'method': self.method, 'probability_floor': self.probability_floor,
                'regularization': self.regularization, 'reference_class': self.reference_class,
                'max_iterations': self.max_iterations, 'parameters': deepcopy(self.parameters), 'report': deepcopy(self.report)}

    @classmethod
    def from_state(cls, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported calibrator state')
        calibrator = cls(state['method'], probability_floor=state['probability_floor'], regularization=state['regularization'],
                         reference_class=state['reference_class'], max_iterations=state['max_iterations'])
        parameters = deepcopy(state['parameters'])
        if calibrator.method == 'temperature':
            value = parameters['inverse_temperature']
            if not isfinite(value) or not .02 <= value <= 50:
                raise ValueError('Invalid saved temperature')
        elif calibrator.method == 'mcllo':
            for name, limit in (('offsets', 20), ('slopes', 10)):
                values = np.asarray(parameters[name])
                if values.shape != (36,) or not np.isfinite(values).all() or np.any(np.abs(values) > limit):
                    raise ValueError('Invalid saved MCLLO parameters')
        else:
            thresholds, values = np.asarray(parameters['thresholds']), np.asarray(parameters['log_values'])
            if thresholds.ndim != 1 or not len(thresholds) or values.shape != thresholds.shape or not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) <= 0) or thresholds[0] < 0 or thresholds[-1] > 1 or parameters['log_range'] != 30:
                raise ValueError('Invalid saved isotonic blocks')
            monotone_log_gap(values, np.zeros_like(values))
        calibrator.parameters = parameters
        calibrator.report = deepcopy(state['report'])
        calibrator.is_fitted = True
        return calibrator
