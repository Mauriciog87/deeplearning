from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import permutations
from math import isfinite, log
from numbers import Integral
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.special import softmax
from scipy.stats import binomtest, sem, t

from src.expected_value import choose_expected_value_action, counterfactual_profits, expected_action_values
from src.settlement import settle_action
from src.utils.calibration import fixed_bin_calibration_bound
from src.utils.change_detection import CategoricalEDetector, PITMonitor, ReferenceConditionalMonitor, categorical_pit
from src.utils.joint_calibration import JointCalibrationMonitor
from src.utils.recalibration import CALIBRATORS, ProbabilityCalibrator
from src.utils.sequential_inference import MultinomialMonitor
from src.utils.temporal_statistics import calibration_error


SCENARIOS = ('uniform', 'fixed_bias', 'abrupt_bias', 'gradual_bias', 'dependence',
             'overconfidence', 'joint_counterexample', 'calibration_improvement')
LEARNED_MODELS = ('extra_trees', 'lstm_one_hot', 'lstm_ordinal')


@dataclass(frozen=True)
class ResearchConfig:
    seed: int = 42
    trials: int = 20
    observations: int = 1000
    reference_size: int = 300
    calibration_size: int = 300
    alpha: float = .05
    scenarios: tuple = SCENARIOS
    recalibrators: tuple = ()
    learned_models: tuple = ()
    include_online: bool = False
    online_iterations: int = 30
    neural_epochs: int = 3
    neural_hidden_size: int = 32
    device: str = 'cpu'
    bankroll: float = 1000.0

    def __post_init__(self):
        for name in ('seed', 'trials', 'observations', 'reference_size', 'calibration_size', 'online_iterations', 'neural_epochs', 'neural_hidden_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'{name} must be a valid integer')
        if self.observations < 20 or min(self.reference_size, self.calibration_size) < 20:
            raise ValueError('Research partitions require at least 20 observations each')
        for name, allowed in (('scenarios', SCENARIOS), ('recalibrators', CALIBRATORS), ('learned_models', LEARNED_MODELS)):
            selected = getattr(self, name)
            if set(selected) - set(allowed) or len(set(selected)) != len(selected):
                raise ValueError(f'Invalid or duplicated {name}')
        if not self.scenarios or not 0 < self.alpha < 1 or not isfinite(self.bankroll) or self.bankroll <= 0:
            raise ValueError('Scenarios, alpha and bankroll must be valid')
        if self.device not in ('cpu', 'cuda', 'auto'):
            raise ValueError('Invalid device')


def generate_scenario(config, scenario, trial):
    sequence = np.random.SeedSequence([config.seed, SCENARIOS.index(scenario), trial])
    words = sequence.generate_state(4).tolist()
    rng = np.random.default_rng(sequence)
    prefix = config.reference_size + config.calibration_size
    count = prefix + config.observations
    change = config.observations // 2 if scenario in ('abrupt_bias', 'gradual_bias', 'calibration_improvement') else None
    truth = np.full((count, 37), 1 / 37)
    forecasts = truth.copy()
    outcomes = np.zeros(count, dtype=int)
    biased = np.full(37, .85 / 36)
    biased[7] = .15
    if scenario == 'fixed_bias':
        truth[:] = biased
    elif scenario in ('abrupt_bias', 'gradual_bias'):
        for index in range(prefix + change, count):
            fraction = 1 if scenario == 'abrupt_bias' else (index - prefix - change + 1) / (config.observations - change)
            truth[index] = (1 - fraction) / 37 + fraction * biased
    elif scenario == 'overconfidence':
        truth = rng.dirichlet(np.full(37, .3), size=count)
        forecasts = softmax(3 * np.log(truth), axis=1)
    elif scenario == 'joint_counterexample':
        orderings = list(permutations(range(3)))
        for index, pattern in enumerate(rng.integers(0, 6, count)):
            order = orderings[pattern]
            inversions = sum(order[i] > order[j] for i in range(3) for j in range(i + 1, 3))
            forecasts[index] = 0
            forecasts[index, :3] = np.array([.5, .3, .2])[list(order)]
            truth[index] = forecasts[index]
            truth[index, :3] += (-1 if inversions % 2 else 1) * np.array([.1, -.1, 0])
    elif scenario == 'calibration_improvement':
        forecasts[:prefix + change] = .6 / 36
        forecasts[:prefix + change, 7] = .4
    for index in range(count):
        if scenario == 'dependence' and index:
            truth[index] = .7 / 37
            truth[index, outcomes[index - 1]] += .3
        outcomes[index] = rng.choice(37, p=truth[index])
    fingerprint = sha256(outcomes.astype('<i8').tobytes() + truth.astype('<f8').tobytes() + forecasts.astype('<f8').tobytes()).hexdigest()
    return {'outcomes': outcomes, 'truth': truth, 'forecasts': forecasts, 'seed_words': words,
            'sha256': fingerprint, 'change_after_test_observations': change,
            'fixed_probability': scenario in ('uniform', 'fixed_bias', 'calibration_improvement')}


def _null_status(scenario, detector):
    if detector in ('reference', 'pit'):
        if scenario == 'dependence':
            return 'assumption_violated'
        return 'alternative' if scenario in ('abrupt_bias', 'gradual_bias', 'calibration_improvement') else 'null'
    if detector == 'joint':
        return 'null' if scenario == 'uniform' else 'alternative'
    return 'null' if scenario in ('uniform', 'overconfidence', 'calibration_improvement') else 'alternative'


def _forecast_metrics(probabilities, outcomes):
    ranking = np.argsort(-probabilities, axis=1, kind='stable')
    rows = np.arange(len(outcomes))
    hits = ranking[:, 0] == outcomes
    confidence = probabilities[rows, ranking[:, 0]]
    return {'log_loss': float(-np.log(np.maximum(probabilities[rows, outcomes], 1e-12)).mean()),
            'brier': float(np.sum((probabilities - np.eye(37)[outcomes]) ** 2, axis=1).mean()),
            'confidence_ece': calibration_error(confidence, hits, max_bins=10),
            'classwise_ece': float(np.mean([calibration_error(probabilities[:, number], outcomes == number, max_bins=10) for number in range(37)])),
            'top_1': float(hits.mean()), 'top_5': float(np.any(ranking[:, :5] == outcomes[:, None], axis=1).mean())}


def _calibration_coverage(probabilities, outcomes, truth, alpha):
    ranking = np.argsort(-probabilities, axis=1, kind='stable')
    top = ranking[:, 0]
    top5 = ranking[:, :5]
    positions = np.arange(len(outcomes))
    channels = {
        'confidence': (probabilities[positions, top, None], (outcomes == top)[:, None], truth[positions, top, None], False),
        'classwise': (probabilities, np.eye(37)[outcomes], truth, True),
        'top_5': (np.take_along_axis(probabilities, top5, axis=1).sum(axis=1)[:, None],
                  np.any(top5 == outcomes[:, None], axis=1)[:, None], np.take_along_axis(truth, top5, axis=1).sum(axis=1)[:, None], False),
    }
    results = {}
    for name, (predicted, actual, conditional, categorical) in channels.items():
        checkpoints = []
        for count in sorted(set(np.linspace(min(20, len(outcomes)), len(outcomes), 4, dtype=int))):
            bound = fixed_bin_calibration_bound(predicted[None, :count], actual[None, :count], alpha=alpha / 3, categorical=categorical)
            targets = []
            for channel in range(predicted.shape[1]):
                bins = np.clip((predicted[:count, channel] * 10).astype(int), 0, 9)
                residual = conditional[:count, channel] - predicted[:count, channel]
                targets.append(np.abs(np.bincount(bins, weights=residual, minlength=10)).sum() / count)
            target = float(np.mean(targets))
            checkpoints.append({'count': int(count), 'target': target, 'lower': bound.lower, 'upper': bound.upper,
                                'covered': bool(bound.lower - 1e-12 <= target <= bound.upper + 1e-12)})
        results[name] = {'covered_at_all_checked_times': all(point['covered'] for point in checkpoints), 'checkpoints': checkpoints}
    return results


def _fit_learned_forecasts(config, prefix, outcomes, seed):
    forecasts, reports = {}, {}
    for name in config.learned_models:
        try:
            if name == 'extra_trees':
                from src.utils.predictor import ExtraTreesPredictor
                model = ExtraTreesPredictor(seed=seed)
                fit = model.fit(outcomes[:prefix].tolist())
                if not model.is_fitted:
                    raise RuntimeError(fit.get('message', 'Model did not fit'))
            else:
                import torch
                from src.checkpoints import seed_everything
                from src.utils.lstm_predictor import LSTMPredictor
                seed_everything(seed, config.device)
                model = LSTMPredictor(hidden_size=config.neural_hidden_size, device=config.device,
                                      representation='one_hot' if name == 'lstm_one_hot' else 'ordinal')
                previous_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                try:
                    fit = model.fit(outcomes[:prefix].tolist(), epochs=config.neural_epochs)
                    if not model.is_trained:
                        raise RuntimeError(fit.get('message', 'Model did not fit'))
                    forecasts[name] = np.asarray([model.predict_proba(outcomes[:index].tolist()) for index in range(prefix, len(outcomes))])
                finally:
                    torch.set_num_threads(previous_threads)
            if name == 'extra_trees':
                forecasts[name] = np.asarray([model.predict_proba(outcomes[:index].tolist()) for index in range(prefix, len(outcomes))])
            reports[name] = {'status': 'ready', 'fit': fit, 'training_indices': [0, prefix], 'test_indices': [prefix, len(outcomes)]}
        except (ImportError, ValueError, RuntimeError) as error:
            reports[name] = {'status': 'unavailable', 'reason': str(error)}
    return forecasts, reports


def run_research_trial(config, scenario, trial):
    data = generate_scenario(config, scenario, trial)
    prefix = config.reference_size + config.calibration_size
    outcomes = data['outcomes'][prefix:]
    truth, base = data['truth'][prefix:], data['forecasts'][prefix:]
    forecasts = {'base': base, 'fair': np.full_like(base, 1 / 37)}
    fitted = {}
    for method in config.recalibrators:
        try:
            calibrator = ProbabilityCalibrator(method)
            fit = calibrator.fit(data['forecasts'][config.reference_size:prefix], data['outcomes'][config.reference_size:prefix])
            forecasts[method] = calibrator.transform(base)
            fitted[method] = {'status': 'ready', 'fit': fit, 'state': calibrator.to_state()}
        except (ImportError, ValueError, RuntimeError) as error:
            fitted[method] = {'status': 'unavailable', 'reason': str(error)}
    learned, learned_reports = _fit_learned_forecasts(config, prefix, data['outcomes'], data['seed_words'][0])
    forecasts.update(learned)
    online = None
    if config.include_online:
        from src.utils.online_recalibration import OnlineRecalibrator
        online = OnlineRecalibrator(max_iterations=config.online_iterations)
        forecasts['online'] = np.zeros_like(base)
    forecasts['frequency'] = np.zeros_like(base)
    pit_rng = np.random.default_rng(data['seed_words'][3])
    pit_values = np.asarray([categorical_pit(probabilities, int(actual), float(pit_rng.random()))
                             for probabilities, actual in zip(data['forecasts'], data['outcomes'])])
    reference_scores = pit_values[:config.reference_size]
    monitors = {'multinomial': MultinomialMonitor('trial', alpha=config.alpha),
                'kt': MultinomialMonitor('trial-kt', alpha=config.alpha, prior_strengths=(18.5,)),
                'e_sr': CategoricalEDetector(average_run_length=config.observations / config.alpha),
                'e_cusum': CategoricalEDetector(average_run_length=config.observations / config.alpha, variant='cusum'),
                'reference': ReferenceConditionalMonitor(reference_scores, alpha=config.alpha, seed=data['seed_words'][1], score_id='randomized_base_pit_order_0_36'),
                'pit': PITMonitor(alpha=config.alpha, seed=data['seed_words'][2]),
                'joint': JointCalibrationMonitor(alpha=config.alpha)}
    frequencies = np.bincount(data['outcomes'][:prefix], minlength=37)
    policies = {name: {'balance': config.bankroll, 'profit': 0.0, 'stake': 0.0, 'expected_profit': 0.0,
                       'peak': config.bankroll, 'drawdown': 0.0, 'action_counts': np.zeros(47, dtype=int)}
                for name in (*forecasts, 'confidence', 'oracle', 'pass')}
    confidence_checks = {name: [] for name in ('multinomial', 'kt')}
    region_covered = {name: True for name in confidence_checks}
    checkpoints = set(np.linspace(min(20, len(outcomes)), len(outcomes), 4, dtype=int))
    counterfactual = np.zeros(47)
    for index, actual in enumerate(outcomes):
        actual = int(actual)
        event_id = str(index)
        forecasts['frequency'][index] = (frequencies + 1) / (frequencies.sum() + 37)
        if online is not None:
            forecasts['online'][index] = online.forecast(base[index], event_id=event_id)['probabilities']
        true_values = expected_action_values(truth[index])
        for name, policy in policies.items():
            if name == 'pass':
                action = 46
            else:
                probability = truth[index] if name == 'oracle' else forecasts['frequency'][index] if name == 'confidence' else forecasts[name][index]
                choice = choose_expected_value_action(probability, policy['balance'], probability_bounds=monitors['multinomial'].probability_bounds if name == 'confidence' else None)
                action = choice.action
            result = settle_action(actual, action, 1, policy['balance'])
            policy['balance'] = result.balance_after
            policy['profit'] += result.net_profit
            policy['stake'] += result.total_stake
            policy['expected_profit'] += true_values[action]
            policy['peak'] = max(policy['peak'], result.balance_after)
            policy['drawdown'] = max(policy['drawdown'], policy['peak'] - result.balance_after)
            policy['action_counts'][action] += 1
        counterfactual += counterfactual_profits(actual)
        for name in ('multinomial', 'kt', 'e_sr', 'e_cusum'):
            monitors[name].update(actual, event_id=event_id)
        monitors['reference'].update(pit_values[prefix + index], event_id=event_id)
        monitors['pit'].update(pit_values[prefix + index], event_id=event_id)
        monitors['joint'].update([base[index]], actual, event_id=event_id)
        if online is not None:
            online.update(actual, event_id=event_id)
        frequencies[actual] += 1
        if data['fixed_probability']:
            for name in confidence_checks:
                monitor = monitors[name]
                log_ratio_at_truth = monitor.log_marginal - float(np.dot(monitor.counts, np.log(truth[0])))
                region_covered[name] = region_covered[name] and log_ratio_at_truth <= -log(config.alpha) + 1e-12
        if data['fixed_probability'] and index + 1 in checkpoints:
            for name in confidence_checks:
                bounds = np.asarray([monitors[name].probability_bounds([number]) for number in range(37)])
                confidence_checks[name].append({'count': index + 1,
                    'covered_all_pockets': bool(np.all((bounds[:, 0] <= truth[0] + 1e-12) & (bounds[:, 1] >= truth[0] - 1e-12))),
                    'mean_width': float(np.mean(bounds[:, 1] - bounds[:, 0]))})
    detector_results = {}
    for name, monitor in monitors.items():
        result = monitor.snapshot()
        detector_results[name] = {'null_status': _null_status(scenario, name), 'result': result}
    for policy in policies.values():
        policy['roi'] = policy['profit'] / policy['stake'] if policy['stake'] else None
        policy['pass_rate'] = float(policy['action_counts'][46] / len(outcomes))
        policy['action_counts'] = policy['action_counts'].tolist()
        del policy['peak']
    return {'scenario': scenario, 'trial': trial, 'seed_words': data['seed_words'], 'dataset_sha256': data['sha256'],
            'partitions': {'reference': [0, config.reference_size], 'calibration': [config.reference_size, prefix], 'test': [prefix, len(data['outcomes'])]},
            'change_after_test_observations': data['change_after_test_observations'], 'detectors': detector_results,
            'confidence_sequences': {'status': 'applicable' if data['fixed_probability'] else 'not_applicable_fixed_probability_assumption', 'methods': confidence_checks,
                                     'joint_region_covered_at_all_times': region_covered if data['fixed_probability'] else None},
            'forecasts': {name: _forecast_metrics(values, outcomes) for name, values in forecasts.items()},
            'calibration_coverage': _calibration_coverage(base, outcomes, truth, config.alpha),
            'policies': policies, 'counterfactual_unit_profit_totals': counterfactual.tolist(),
            'recalibrator_fits': fitted, 'learned_model_fits': learned_reports,
            'online': online.snapshot() if online is not None else None}


def _binomial_interval(successes, trials):
    if not trials:
        return {'rate': None, 'lower': None, 'upper': None, 'successes': 0, 'trials': 0}
    interval = binomtest(successes, trials).proportion_ci(confidence_level=.95, method='exact')
    return {'rate': successes / trials, 'lower': interval.low, 'upper': interval.high, 'successes': successes, 'trials': trials}


def _mean_interval(values):
    values = [float(value) for value in values if value is not None and isfinite(value)]
    if not values:
        return {'mean': None, 'lower': None, 'upper': None, 'trials': 0}
    mean = float(np.mean(values))
    width = float(t.ppf(.975, len(values) - 1) * sem(values)) if len(values) > 1 else None
    return {'mean': mean, 'lower': mean - width if width is not None else None, 'upper': mean + width if width is not None else None, 'trials': len(values)}


def summarize_research_trials(records, config):
    result = {}
    for scenario in config.scenarios:
        selected = [record for record in records if record['scenario'] == scenario]
        summary = {'trials': len(selected), 'detectors': {}, 'forecasts': {}, 'policies': {}, 'confidence_sequences': {}, 'calibration_coverage': {}}
        for name in selected[0]['detectors']:
            alarms = [record['detectors'][name]['result']['alarm_at'] for record in selected]
            status = selected[0]['detectors'][name]['null_status']
            delays, capped, early = [], [], 0
            for record, alarm in zip(selected, alarms):
                change = record['change_after_test_observations']
                if name == 'joint' and scenario == 'calibration_improvement':
                    change = 0
                if change is None and status == 'alternative':
                    change = 0
                if change is not None and status == 'alternative':
                    if alarm is not None and alarm <= change:
                        early += 1
                    else:
                        capped.append((alarm if alarm is not None else config.observations) - change)
                        if alarm is not None:
                            delays.append(alarm - change)
            summary['detectors'][name] = {'null_status': status, 'alarm_rate': _binomial_interval(sum(alarm is not None for alarm in alarms), len(alarms)),
                'pre_change_alarms': early, 'detected_delay': _mean_interval(delays), 'restricted_delay': _mean_interval(capped),
                'post_change_detection_rate': _binomial_interval(len(delays), len(alarms)) if status == 'alternative' else None,
                'detection_rate_given_no_early_alarm': _binomial_interval(len(delays), len(alarms) - early) if status == 'alternative' else None,
                'delay_interpretation': 'Detected delay excludes nondetections; restricted delay censors at the fixed horizon. Both exclude pre-change alarms.'}
        for name in sorted({name for record in selected for name in record['forecasts']}):
            matched = [record for record in selected if name in record['forecasts']]
            summary['forecasts'][name] = {metric: _mean_interval([record['forecasts'][name][metric] for record in matched]) for metric in matched[0]['forecasts'][name]}
            summary['forecasts'][name]['paired_log_loss_delta_vs_base'] = _mean_interval([record['forecasts'][name]['log_loss'] - record['forecasts']['base']['log_loss'] for record in matched])
        for name in sorted({name for record in selected for name in record['policies']}):
            summary['policies'][name] = {metric: _mean_interval([record['policies'][name][metric] for record in selected if name in record['policies']])
                                         for metric in ('profit', 'expected_profit', 'stake', 'roi', 'drawdown', 'pass_rate')}
        for name in ('multinomial', 'kt'):
            checks = [record['confidence_sequences']['methods'][name] for record in selected if record['confidence_sequences']['status'] == 'applicable']
            summary['confidence_sequences'][name] = {'coverage': _binomial_interval(sum(all(point['covered_all_pockets'] for point in trial) for trial in checks), len(checks)),
                'joint_region_coverage_at_all_times': _binomial_interval(sum(record['confidence_sequences']['joint_region_covered_at_all_times'][name] for record in selected if record['confidence_sequences']['status'] == 'applicable'), len(checks)),
                'coverage_scope': 'Joint region checked after every observation; pocket projections checked at four predetermined counts',
                'final_mean_width': _mean_interval([trial[-1]['mean_width'] for trial in checks])}
        for name in ('confidence', 'classwise', 'top_5'):
            summary['calibration_coverage'][name] = _binomial_interval(sum(record['calibration_coverage'][name]['covered_at_all_checked_times'] for record in selected), len(selected))
        result[scenario] = summary
    return result


def run_research_benchmark(config=None, *, progress=None):
    config = config or ResearchConfig()
    started = time.perf_counter()
    records = []
    for scenario in config.scenarios:
        for trial in range(config.trials):
            records.append(run_research_trial(config, scenario, trial))
            if progress is not None:
                progress(scenario, trial + 1, config.trials)
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), *(root / 'src' / 'utils' / name for name in ('calibration.py', 'change_detection.py', 'joint_calibration.py', 'sequential_inference.py', 'recalibration.py', 'online_recalibration.py', 'lstm_predictor.py', 'predictor.py', 'temporal_statistics.py')),
             *(root / 'src' / name for name in ('expected_value.py', 'settlement.py', 'probabilities.py', 'checkpoints.py'))]
    packages = {}
    for package in ('numpy', 'scipy', 'scikit-learn', 'torch', 'gymnasium'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=False)
    manifest = {'schema_version': 1, 'generator': 'roulette_research_scenarios_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': time.perf_counter() - started, 'python': platform.python_version(), 'packages': packages,
                'git_head': git.stdout.strip() if git.returncode == 0 else None,
                'source_sha256': {str(path.relative_to(root)).replace('\\', '/'): sha256(path.read_bytes()).hexdigest() for path in files},
                'experiment_unit': 'One independently generated sequence per scenario/trial; methods share its outcomes. Alpha is per method, not family-wise across compared methods or trial replications. There are no detector restarts within a trial.',
                'uncertainty': 'Exact 95% binomial Monte Carlo intervals for rates; approximate Student-t intervals across independent trials for means and paired differences. Not simultaneous across comparisons.',
                'inference': 'IID-null false alarms are distinguished from alternatives and violated assumptions. CS coverage is measured only under fixed conditional probabilities. Calibration bounds target known conditional residuals.',
                'score_floor': 1e-12, 'pit_order': list(range(37)),
                'data_access': 'Synthetic generation only; no database is read or modified. Oracle policy uses known generator probabilities and is not a deployable predictor.',
                'limitations': 'Synthetic evidence does not establish real-wheel predictive superiority. No automatic selection or promotion of methods.'}
    return {'config': asdict(config), 'manifest': manifest, 'summary': summarize_research_trials(records, config), 'trials': records}


def format_research_report(result):
    lines = [f"RESEARCH BENCHMARK: {result['config']['trials']} trials per scenario; {result['config']['observations']} test observations",
             'Alpha is per method. Rates describe the declared null, alternative or violated assumption.']
    for scenario, summary in result['summary'].items():
        lines.append(scenario)
        for name, detector in summary['detectors'].items():
            rate = detector['alarm_rate']
            lines.append(f"  {name}: {detector['null_status']}; alarms {rate['successes']}/{rate['trials']} [{rate['lower']:.3f}, {rate['upper']:.3f}]")
    lines.append('Full results include paired scores, exposure, delay/censoring, coverage, fitted parameters and source hashes. Synthetic results do not establish a real-world edge.')
    return '\n'.join(lines)
