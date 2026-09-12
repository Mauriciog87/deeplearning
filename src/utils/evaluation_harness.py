from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from math import isfinite, log
from random import Random
from typing import Dict, List, Optional, Tuple

from src.engine.prediction_engine import FullPrediction, PredictionEngine, PredictorType
from src.probabilities import FAIR_PROBABILITY, validate_probabilities, Availability
from src.settlement import settle_bet, settle_action
from src.datasets import as_sessions, dataset_manifest
from src.utils.temporal_statistics import adaptive_bins, calibration_error, block_bootstrap_indices
from src.utils.calibration import CalibrationBound, fixed_bin_calibration_bound
import numpy as np
import platform
from importlib.metadata import version, PackageNotFoundError


EPSILON = 1e-12
MODEL_ORDER = [
    "lstm",
    "dqn",
    "extra_trees",
    "bias",
    "consensus",
    "fair",
    "rolling_frequency",
    "last_n",
    "hot",
    "random",
    "pass",
]


@dataclass
class EvaluationConfig:
    training_window: int = 500
    testing_window: int = 100
    step_size: int = 100
    bet_top_n: int = 5
    top_k: Tuple[int, ...] = (1, 3, 5, 10)
    ece_bins: int = 10
    seed: int = 42
    include_baselines: bool = True
    model_path: Optional[str] = None
    bootstrap_resamples: int = 2000
    confidence_level: float = 0.95
    comparison_baseline: str = "fair"
    compute_intervals: bool = True
    initial_bankroll: float = 1000.0
    unit_stake: float = 1.0
    lstm_epochs: int = 30
    runs: int = 5
    device: str = 'auto'
    models: Tuple[str, ...] = ('lstm', 'extra_trees', 'bias')
    block_length: Optional[int] = None

    def __post_init__(self):
        if self.training_window < 1 or self.testing_window < 1 or self.step_size < self.testing_window:
            raise ValueError('Windows must be positive and test folds must not overlap (step_size >= testing_window)')
        if not 1 <= self.bet_top_n <= 37 or not self.top_k or any(not 1 <= k <= 37 for k in self.top_k):
            raise ValueError('Bet top N and top K must be between 1 and 37')
        if not 1 <= self.ece_bins <= 10 or self.runs < 1 or self.lstm_epochs < 1:
            raise ValueError('Bins must be between 1 and 10; runs and epochs must be positive')
        if self.bootstrap_resamples < 0 or not 0 < self.confidence_level < 1:
            raise ValueError('Invalid bootstrap configuration')
        if self.block_length is not None and self.block_length < 1:
            raise ValueError('Block length must be positive')
        if not isfinite(self.initial_bankroll) or self.initial_bankroll <= 0 or not isfinite(self.unit_stake) or self.unit_stake <= 0:
            raise ValueError('Bankroll and unit stake must be positive and finite')
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Device must be auto, cpu or cuda')
        if set(self.models) - {'lstm', 'extra_trees', 'bias', 'dqn'}:
            raise ValueError('Unknown model requested')


@dataclass
class PredictionEvaluationRow:
    fold: int
    index: int
    model: str
    actual: int
    predicted: Optional[int]
    confidence: Optional[float]
    number_probs: Dict[int, float]
    top_numbers: List[Tuple[int, float]]
    profit: float
    session_id: str = 'provided'
    spin_id: str = ''
    run_id: int = 0
    total_stake: Optional[float] = None
    balance_before: Optional[float] = None
    balance_after: Optional[float] = None
    kind: str = 'forecast'
    action: Optional[int] = None


@dataclass
class LiveModelPrediction:
    model: str
    predicted: int
    confidence: float
    top_numbers: List[Tuple[int, float]]
    number_probs: Dict[int, float]


@dataclass
class MetricInterval:
    mean: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass
class CalibrationBin:
    lower: float = 0.0
    upper: float = 0.0
    count: int = 0
    avg_confidence: float = 0.0
    accuracy: float = 0.0
    run_id: int = 0


@dataclass
class ModelComparisonSummary:
    model: str = ""
    baseline_model: str = ""
    metric: str = ""
    delta_mean: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    positive_bootstrap_fraction: float = 0.0


@dataclass
class ModelEvaluationSummary:
    spins: int
    exact_accuracy: float
    top_k_hit_rates: Dict[int, float]
    log_loss: float
    brier: float
    ece: float
    roi: float
    max_drawdown: float
    max_consecutive_losses: int
    activation_rate: float
    exact_accuracy_interval: MetricInterval = field(default_factory=MetricInterval)
    top_k_hit_rate_intervals: Dict[int, MetricInterval] = field(default_factory=dict)
    log_loss_interval: MetricInterval = field(default_factory=MetricInterval)
    brier_interval: MetricInterval = field(default_factory=MetricInterval)
    roi_interval: MetricInterval = field(default_factory=MetricInterval)
    classwise_ece: float = 0.0
    top_k_ece: Dict[int, float] = field(default_factory=dict)
    calibration_bounds: Dict[str, CalibrationBound] = field(default_factory=dict)
    calibration_bins: List[CalibrationBin] = field(default_factory=list)
    comparisons: Dict[str, ModelComparisonSummary] = field(default_factory=dict)
    status: str = 'ready'
    reason: str = ''
    net_profit: Optional[float] = None
    total_stake: Optional[float] = None
    training_runs: int = 0
    evaluated_rows: int = 0


@dataclass
class OpeReadinessReport:
    supported_level: str = "paired_walk_forward_backtest"
    evaluated_rows: int = 0
    logged_actions_available: bool = False
    propensities_available: bool = False
    missing_fields: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


@dataclass
class WalkForwardEvaluationResult:
    config: EvaluationConfig
    folds: int
    per_model_summaries: Dict[str, ModelEvaluationSummary]
    warnings: List[str] = field(default_factory=list)
    rows: List[PredictionEvaluationRow] = field(default_factory=list)
    ope_readiness: OpeReadinessReport = field(default_factory=OpeReadinessReport)
    manifest: Dict = field(default_factory=dict)
    model_statuses: List[Dict] = field(default_factory=list)


def evaluate_walk_forward(numbers, config: Optional[EvaluationConfig] = None, cancel_event=None) -> WalkForwardEvaluationResult:
    config = config or EvaluationConfig()
    sessions = as_sessions(numbers)
    manifest = {'schema_version': 3, 'dataset': dataset_manifest(sessions),
                'config': asdict(config), 'seeds': [config.seed + run for run in range(config.runs)],
                'python': platform.python_version(), 'packages': {}, 'partitions': [],
                'log_loss_probability_floor': EPSILON,
                'interval_assumptions': 'Conditional on this dataset; approximately stationary circular temporal blocks within sessions, shared across resampled training seeds. Not evidence of future profit.'}
    manifest['calibration_inference'] = {
        'method': 'hilbert_azuma_dyadic_v1',
        'target': CalibrationBound().target,
        'bin_edges': np.linspace(0, 1, config.ece_bins + 1).tolist(),
        'model_family_size': len(MODEL_ORDER),
        'metric_family_size': 2 + len(set(config.top_k) - {1}),
        'assumptions': 'All configured runs share the same outcomes. Forecasts, rankings and observation inclusion are chosen before each outcome. Bins and configuration are fixed before evaluation.',
        'coverage': 'Simultaneous over time, the declared model family, and confidence/classwise/top-set summaries; conditional on the configured training runs. No guarantee across data-selected configurations.',
        'interpretation': 'Conservative bounds for predictable fixed-bin residuals, not population l2-ECE intervals or proof of joint calibration. Adaptive ECE remains descriptive.',
    }
    for package in ('numpy', 'scipy', 'scikit-learn', 'torch', 'gymnasium'):
        try:
            manifest['packages'][package] = version(package)
        except PackageNotFoundError:
            manifest['packages'][package] = None
    warnings, rows, statuses = [], [], []
    required = config.training_window + config.testing_window
    fold = 0
    evaluated = 0
    for session in sessions:
        if len(session.numbers) < required:
            warnings.append(f'Session {session.session_id}: need {required} observations, received {len(session.numbers)}.')
            continue
        balances = {(run, model): config.initial_bankroll for run in range(config.runs) for model in MODEL_ORDER}
        for start in range(0, len(session.numbers) - required + 1, config.step_size):
            test_start = start + config.training_window
            test_end = test_start + config.testing_window
            train_data = list(session.numbers[start:test_start])
            manifest['partitions'].append({'fold': fold, 'session_id': session.session_id,
                'train_spin_ids': list(session.spin_ids[start:test_start]),
                'test_spin_ids': list(session.spin_ids[test_start:test_end])})
            for run in range(config.runs):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError('Evaluation cancelled')
                engine = PredictionEngine(model_path=config.model_path, device=config.device, seed=config.seed + run)
                engine.load_history(train_data, session_id=session.session_id)
                requested_fit = tuple(model for model in config.models if model in ('lstm', 'extra_trees'))
                if requested_fit:
                    engine.train_sync(epochs=config.lstm_epochs, models=requested_fit, cancel_event=cancel_event)
                policy_reason = None
                if 'dqn' in config.models:
                    policy_reason = _check_policy_partition(engine, session, test_start, test_end, config)
                    if policy_reason:
                        warnings.append(f'DQN fold {fold}, run {run}: {policy_reason}')
                rng = Random(config.seed + run + fold * 1009)
                for index in range(test_start, test_end):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError('Evaluation cancelled')
                    actual = session.numbers[index]
                    emitted = _engine_rows(engine, fold, index, actual, config)
                    if 'dqn' in config.models and not policy_reason:
                        decision = engine.get_policy_decision(balances[(run, 'dqn')], config.initial_bankroll, config.unit_stake)
                        engine.statuses[PredictorType.DQN] = decision.status
                        if decision.action is not None:
                            emitted.append(PredictionEvaluationRow(fold, index, 'dqn', actual, None, None, {}, [], 0,
                                                                  kind='policy', action=decision.action))
                    if config.include_baselines:
                        emitted.extend(_baseline_rows(engine.history, fold, index, actual, config, rng))
                        emitted.append(PredictionEvaluationRow(fold, index, 'pass', actual, None, None, {}, [], 0,
                                                              kind='policy', action=46))
                    for row in emitted:
                        row.session_id, row.spin_id, row.run_id = session.session_id, session.spin_ids[index], run
                        key = (run, row.model)
                        balance = balances[key]
                        selected = [n for n, _ in row.top_numbers[:config.bet_top_n]] if row.kind == 'forecast' else []
                        if balance < config.unit_stake * len(selected):
                            selected = []
                        settlement = (settle_action(actual, row.action, config.unit_stake, balance)
                                      if row.kind == 'policy' else settle_bet(actual, selected, config.unit_stake, balance))
                        row.profit, row.total_stake = settlement.net_profit, settlement.total_stake
                        row.balance_before, row.balance_after = settlement.balance_before, settlement.balance_after
                        balances[key] = settlement.balance_after
                        rows.append(row)
                    engine.add_number(actual)
                statuses.append({'fold': fold, 'run_id': run, 'session_id': session.session_id, **engine.get_status()})
            evaluated += config.testing_window * config.runs
            fold += 1
    keys = [(row.run_id, row.session_id, row.spin_id, row.model) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('An observation was evaluated twice for the same model and run')
    summaries = _summarize_rows(rows, config, evaluated, cancel_event)
    for model, summary in summaries.items():
        if not summary.spins:
            reasons = [record.get('models', {}).get(model, {}).get('reason', '') for record in statuses]
            summary.reason = '; '.join(sorted(set(reason for reason in reasons if reason))) or 'No evaluable predictions'
    return WalkForwardEvaluationResult(config, fold, summaries, warnings, rows,
                                       _assess_ope_readiness(rows), manifest, statuses)


def _check_policy_partition(engine, session, test_start, test_end, config):
    from src.probabilities import ModelStatus
    decision = engine.get_policy_decision(config.initial_bankroll, config.initial_bankroll, config.unit_stake)
    reason = decision.status.reason if decision.action is None else None
    metadata = engine.policy_metadata.get('training_config', {})
    if not reason and metadata.get('data_source') == 'real':
        partition = metadata.get('partition') or {}
        records = partition.get('dataset', {}).get('sessions', [])
        original = next((item for item in records if item['session_id'] == session.session_id), None)
        split = next((item for item in partition.get('partitions', []) if item['session_id'] == session.session_id), None)
        if original is None or split is None or original['source'] != session.source:
            reason = 'Checkpoint dataset provenance does not cover this session'
        else:
            original_numbers = dict(zip(original['spin_ids'], original['numbers']))
            if any(original_numbers.get(spin_id) != number for spin_id, number in zip(session.spin_ids, session.numbers)):
                reason = 'Checkpoint and evaluation observations do not match'
            elif set(split['train_spin_ids']) & set(session.spin_ids[test_start:test_end]):
                reason = 'Checkpoint training observations overlap this test fold'
    elif not reason and metadata.get('data_source') != 'simulated':
        reason = 'Checkpoint has no verifiable training provenance'
    engine.statuses[PredictorType.DQN] = ModelStatus(Availability.UNAVAILABLE, reason) if reason else decision.status
    return reason


def format_evaluation_report(result: WalkForwardEvaluationResult) -> str:
    def value(number, percent=False):
        if number is None:
            return 'N/A'
        return f'{number * 100:.2f}%' if percent else f'{number:.4f}'
    lines = ['WALK-FORWARD ENGINE EVALUATION', f'Folds: {result.folds}; training runs: {result.config.runs}',
             f'Windows: train={result.config.training_window}, test={result.config.testing_window}, step={result.config.step_size}',
             f'Unit stake per number: {result.config.unit_stake}; bet top N: {result.config.bet_top_n}',
             'Model             Status             Acc@1    Top5     Top10    LogLoss  Brier    ROI']
    lines.extend(result.warnings)
    for model, summary in result.per_model_summaries.items():
        lines.append(f'{model:<17} {summary.status:<18} {value(summary.exact_accuracy, True):>8} '
                     f'{value(summary.top_k_hit_rates.get(5), True):>8} {value(summary.top_k_hit_rates.get(10), True):>8} '
                     f'{value(summary.log_loss):>8} {value(summary.brier):>8} {value(summary.roi, True):>8}')
        if summary.reason:
            lines.append(f'  {summary.reason}')
        if summary.net_profit is not None:
            lines.append(f'  Profit/run: {summary.net_profit:.2f}; exposure/run: {summary.total_stake:.2f}; max drawdown: {summary.max_drawdown:.2f}')
    if result.config.compute_intervals:
        lines.append(f'BOOTSTRAP INTERVALS ({result.config.confidence_level:.0%} CI)')
        lines.append(result.manifest.get('interval_assumptions', 'Temporal blocks within sessions; conditional on this dataset.'))
        for model, summary in result.per_model_summaries.items():
            if summary.spins and summary.log_loss is not None:
                lines.append(f'{model}: log loss {_format_interval(summary.log_loss_interval)}; '
                             f'Brier {_format_interval(summary.brier_interval)}; ROI {_format_interval(summary.roi_interval)}')
    lines.append('CALIBRATION RELIABILITY')
    lines.append('ECE is computed within each run, then averaged across runs. Classwise ECE averages class errors; it does not establish joint calibration.')
    lines.append('Adaptive bins keep tied probabilities together (target: 20 observations/bin/run). Top-k calibration concerns the selected set probability.')
    for model, summary in result.per_model_summaries.items():
        if summary.ece is not None:
            lines.append(f'{model}: confidence ECE {value(summary.ece)}; classwise ECE {value(summary.classwise_ece)}')
    if result.config.compute_intervals:
        lines.append('CALIBRATION BOUNDS: fixed-bin conditional residuals, simultaneous over time and the declared model/metric family.')
        lines.append('These conservative bounds do not cover adaptive ECE or establish joint calibration. All runs must share outcomes; forecasts and inclusion must precede results.')
        for model, summary in result.per_model_summaries.items():
            for name, bound in summary.calibration_bounds.items():
                if bound.status == 'available':
                    lines.append(f'{model} {name}: {value(bound.point)} [{value(bound.lower)}, {value(bound.upper)}]; unique outcomes={bound.observations}')
                else:
                    lines.append(f'{model} {name}: unavailable ({bound.reason})')
    lines.append(f'PAIRED IMPROVEMENT VS {result.config.comparison_baseline}')
    for model, summary in result.per_model_summaries.items():
        for metric, comparison in summary.comparisons.items():
            lines.append(f'{model} {metric}: delta {value(comparison.delta_mean)} '
                         f'[{value(comparison.lower)}, {value(comparison.upper)}]')
    lines.extend(_format_ope_readiness(result.ope_readiness))
    return '\n'.join(lines)


def _assess_ope_readiness(rows: List[PredictionEvaluationRow]) -> OpeReadinessReport:
    missing_fields = [
        "behavior_policy_id",
        "logged_action_numbers",
        "action_probability",
        "stake",
        "payout",
        "bankroll_before",
        "logged_reward",
    ]
    next_steps = [
        'Passive roulette replay observes every outcome: action-independent settlement needs no propensity weighting.',
        "Log the policy name and selection probability for every deployed betting action.",
        "Store action numbers, stake, payout and bankroll before settlement.",
        "Use IPS/DR only after propensities are available and overlap is checked.",
    ]
    return OpeReadinessReport(
        evaluated_rows=len(rows),
        logged_actions_available=False,
        propensities_available=False,
        missing_fields=missing_fields,
        next_steps=next_steps,
    )


def _format_ope_readiness(report: OpeReadinessReport) -> List[str]:
    lines = [
        "",
        "OPE READINESS",
        f"Supported now: {report.supported_level}",
        f"Evaluated prediction rows: {report.evaluated_rows}",
        (
            "Logged actions: yes"
            if report.logged_actions_available
            else "Logged actions: no"
        ),
        (
            "Propensities: yes"
            if report.propensities_available
            else "Propensities: no"
        ),
    ]
    if report.missing_fields:
        lines.append("Missing for IPS/DR/SWITCH:")
        lines.extend(f"- {field}" for field in report.missing_fields)
    if report.next_steps:
        lines.append("Next logging steps:")
        lines.extend(f"- {step}" for step in report.next_steps)
    return lines


def make_fair_prediction(
    history: List[int],
    bet_top_n: int = 5,
) -> LiveModelPrediction:
    return _live_prediction_from_probs("fair", _fair_probs(), bet_top_n)


def make_rolling_frequency_prediction(
    history: List[int],
    bet_top_n: int = 5,
) -> LiveModelPrediction:
    return _live_prediction_from_probs(
        "rolling_frequency",
        _rolling_frequency_probs(history),
        bet_top_n,
    )


def make_last_n_prediction(
    history: List[int],
    last_n: int = 18,
    bet_top_n: int = 5,
) -> LiveModelPrediction:
    selected_numbers = _last_n_numbers(history, last_n, bet_top_n)
    return _live_prediction_from_probs(
        "last_n",
        _selected_mixture_probs(selected_numbers),
        bet_top_n,
    )


def _engine_rows(
    engine: PredictionEngine,
    fold: int,
    index: int,
    actual: int,
    config: EvaluationConfig,
) -> List[PredictionEvaluationRow]:
    rows = []
    predictions = engine.predict_all()

    for predictor_type, prediction in predictions.items():
        if prediction is None or predictor_type.value not in config.models:
            continue
        rows.append(_row_from_prediction(
            fold,
            index,
            predictor_type.value,
            actual,
            prediction,
            config,
        ))

    consensus = engine.get_consensus_prediction({kind: prediction for kind, prediction in predictions.items() if kind.value in config.models})
    if consensus is not None and consensus.predictor == PredictorType.CONSENSUS:
        rows.append(_row_from_prediction(
            fold,
            index,
            "consensus",
            actual,
            consensus,
            config,
        ))

    return rows


def _row_from_prediction(
    fold: int,
    index: int,
    model: str,
    actual: int,
    prediction: FullPrediction,
    config: EvaluationConfig,
) -> PredictionEvaluationRow:
    number_probs = _normalize_probs(prediction.number.all_probabilities)
    top_numbers = _top_numbers(number_probs, 37)
    confidence = max(number_probs.values()) if number_probs else 0.0

    return PredictionEvaluationRow(
        fold=fold,
        index=index,
        model=model,
        actual=actual,
        predicted=top_numbers[0][0],
        confidence=confidence,
        number_probs=number_probs,
        top_numbers=top_numbers,
        profit=_flat_profit(actual, top_numbers[:config.bet_top_n]),
    )


def _live_prediction_from_probs(
    model: str,
    probs: Dict[int, float],
    bet_top_n: int,
) -> LiveModelPrediction:
    number_probs = _normalize_probs(probs)
    top_numbers = _top_numbers(number_probs, bet_top_n)
    predicted = top_numbers[0][0] if top_numbers else 0
    confidence = top_numbers[0][1] if top_numbers else 0.0

    return LiveModelPrediction(
        model=model,
        predicted=predicted,
        confidence=confidence,
        top_numbers=top_numbers,
        number_probs=number_probs,
    )


def _baseline_rows(
    history: List[int],
    fold: int,
    index: int,
    actual: int,
    config: EvaluationConfig,
    rng: Random,
) -> List[PredictionEvaluationRow]:
    return [
        _row_from_probs(fold, index, "fair", actual, _fair_probs(), config),
        _row_from_probs(
            fold,
            index,
            "rolling_frequency",
            actual,
            _rolling_frequency_probs(history),
            config,
        ),
        _row_from_probs(
            fold,
            index,
            "last_n",
            actual,
            _selected_mixture_probs(_last_n_numbers(history, 18, config.bet_top_n)),
            config,
        ),
        _row_from_probs(
            fold,
            index,
            "hot",
            actual,
            _selected_mixture_probs(_hot_numbers(history, config.bet_top_n)),
            config,
        ),
        _row_from_probs(
            fold,
            index,
            "random",
            actual,
            _selected_mixture_probs(rng.sample(range(37), config.bet_top_n)),
            config,
        ),
    ]


def _row_from_probs(
    fold: int,
    index: int,
    model: str,
    actual: int,
    probs: Dict[int, float],
    config: EvaluationConfig,
) -> PredictionEvaluationRow:
    number_probs = _normalize_probs(probs)
    top_numbers = _top_numbers(number_probs, 37)
    predicted = top_numbers[0][0] if top_numbers else 0
    confidence = top_numbers[0][1] if top_numbers else 0.0

    return PredictionEvaluationRow(
        fold=fold,
        index=index,
        model=model,
        actual=actual,
        predicted=predicted,
        confidence=confidence,
        number_probs=number_probs,
        top_numbers=top_numbers,
        profit=_flat_profit(actual, top_numbers[:config.bet_top_n]),
    )


def _metric_values(rows, config, calibration=True):
    forecasts = [row for row in rows if row.kind == 'forecast']
    values = {'roi': _roi(rows, config.bet_top_n), 'profit_per_spin': float(np.mean([row.profit for row in rows])) if rows else None}
    values.update({name: None for name in ('exact_accuracy', 'log_loss', 'brier', 'ece', 'classwise_ece')})
    values.update({f'top_{k}_hit': None for k in config.top_k})
    values.update({f'top_{k}_ece': None for k in config.top_k})
    if forecasts:
        probabilities = np.asarray([[row.number_probs[number] for number in range(37)] for row in forecasts])
        outcomes = np.asarray([row.actual for row in forecasts])
        positions = np.arange(len(forecasts))
        hits = np.asarray([_row_exact_hit(row) for row in forecasts])
        truth = np.eye(37)[outcomes]
        values.update(exact_accuracy=float(hits.mean()),
                      log_loss=float(-np.log(np.maximum(EPSILON, probabilities[positions, outcomes])).mean()),
                      brier=float(np.sum((probabilities - truth) ** 2, axis=1).mean()))
        if calibration:
            values.update(_run_calibration_values(forecasts, config))
        for k in config.top_k:
            top_hits = np.asarray([_row_top_hit(row, k) for row in forecasts])
            values[f'top_{k}_hit'] = float(top_hits.mean())
    return values


def _run_calibration_values(rows, config):
    runs = defaultdict(list)
    for row in rows:
        runs[row.run_id].append(row)
    estimates = defaultdict(list)
    for run_rows in runs.values():
        estimates['ece'].append(_ece(run_rows, config.ece_bins))
        estimates['classwise_ece'].append(_classwise_ece(run_rows, config.ece_bins))
        for k in config.top_k:
            estimates[f'top_{k}_ece'].append(_top_k_ece(run_rows, k, config.ece_bins))
    return {metric: float(np.mean(values)) for metric, values in estimates.items()}


def _interval(point, values, config):
    if point is None:
        return MetricInterval()
    finite = [value for value in values if value is not None and isfinite(value)]
    if not finite:
        return MetricInterval(point, None, None)
    lower, upper = _confidence_interval(finite, config.confidence_level)
    return MetricInterval(point, lower, upper)


def _calibration_bounds(rows, config):
    if not rows or not config.compute_intervals:
        return {}
    names = ['confidence', 'classwise'] + [f'top_{k}' for k in sorted(set(config.top_k) - {1})]
    runs = defaultdict(dict)
    for row in rows:
        identity = (row.session_id, row.index, row.spin_id)
        if identity in runs[row.run_id]:
            raise ValueError('Calibration inference cannot count an outcome twice in a run')
        runs[row.run_id][identity] = row
    keys = sorted(next(iter(runs.values())))
    if set(runs) != set(range(config.runs)) or any(set(run) != set(keys) for run in runs.values()):
        return {name: CalibrationBound(reason='All configured runs must cover the same observations') for name in names}
    aligned = [[runs[run][key] for key in keys] for run in range(config.runs)]
    actual = np.asarray([[row.actual for row in run] for run in aligned])
    if not np.all(actual == actual[0]):
        raise ValueError('Training runs disagree about an observed outcome')
    probabilities = np.asarray([[[row.number_probs[n] for n in range(37)] for row in run] for run in aligned])
    alpha = (1 - config.confidence_level) / (len(MODEL_ORDER) * len(names))
    confidence = np.asarray([[row.confidence for row in run] for run in aligned])[:, :, None]
    hits = np.asarray([[_row_exact_hit(row) for row in run] for run in aligned])[:, :, None]
    result = {'confidence': fixed_bin_calibration_bound(confidence, hits, bins=config.ece_bins, alpha=alpha),
              'classwise': fixed_bin_calibration_bound(probabilities, np.eye(37)[actual], bins=config.ece_bins,
                                                       alpha=alpha, categorical=True)}
    for k in sorted(set(config.top_k) - {1}):
        mass = np.asarray([[sum(p for _, p in row.top_numbers[:k]) for row in run] for run in aligned])[:, :, None]
        found = np.asarray([[_row_top_hit(row, k) for row in run] for run in aligned])[:, :, None]
        result[f'top_{k}'] = fixed_bin_calibration_bound(mass, found, bins=config.ece_bins, alpha=alpha)
    return result


def _summarize_rows(rows, config, total_evaluation_spins, cancel_event=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.model].append(row)
    summaries = {}
    for model, model_rows in grouped.items():
        points = _metric_values(model_rows, config)
        samples = defaultdict(list)
        if config.compute_intervals and config.bootstrap_resamples:
            for indices in block_bootstrap_indices(model_rows, config.bootstrap_resamples, config.seed, config.block_length):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError('Evaluation cancelled')
                measured = _metric_values([model_rows[index] for index in indices], config, calibration=False)
                for key, value in measured.items():
                    samples[key].append(value)
        intervals = {key: _interval(value, samples[key], config) for key, value in points.items() if 'ece' not in key}
        trajectories = defaultdict(list)
        for row in model_rows:
            trajectories[(row.run_id, row.session_id)].append(row.profit)
        runs = len({row.run_id for row in model_rows})
        forecasts = [row for row in model_rows if row.kind == 'forecast']
        summaries[model] = ModelEvaluationSummary(
            spins=len({(row.session_id, row.spin_id or row.index) for row in model_rows}),
            exact_accuracy=points['exact_accuracy'],
            top_k_hit_rates={k: points[f'top_{k}_hit'] for k in config.top_k},
            log_loss=points['log_loss'], brier=points['brier'], ece=points['ece'], roi=points['roi'],
            max_drawdown=max(_max_drawdown(profits) for profits in trajectories.values()),
            max_consecutive_losses=max(_max_consecutive_losses(profits) for profits in trajectories.values()),
            activation_rate=len(model_rows) / total_evaluation_spins if total_evaluation_spins else 0.0,
            exact_accuracy_interval=intervals['exact_accuracy'],
            top_k_hit_rate_intervals={k: intervals[f'top_{k}_hit'] for k in config.top_k},
            log_loss_interval=intervals['log_loss'], brier_interval=intervals['brier'],
            roi_interval=intervals['roi'], classwise_ece=points['classwise_ece'],
            top_k_ece={k: points[f'top_{k}_ece'] for k in config.top_k},
            calibration_bounds=_calibration_bounds(forecasts, config),
            calibration_bins=_calibration_bins(forecasts, config.ece_bins),
            net_profit=sum(row.profit for row in model_rows) / runs,
            total_stake=sum(_row_stake(row, config.bet_top_n) for row in model_rows) / runs,
            training_runs=(runs if model in ('lstm', 'extra_trees') or
                           (model == 'consensus' and set(config.models) & {'lstm', 'extra_trees'})
                           else 1 if model == 'dqn' else 0), evaluated_rows=len(model_rows),
            reason='Policy only; forecast scores are not applicable.' if not forecasts else '',
        )
    _attach_model_comparisons(summaries, grouped, config, cancel_event)
    if total_evaluation_spins:
        for model in MODEL_ORDER:
            if model not in summaries:
                summaries[model] = ModelEvaluationSummary(
                    spins=0, exact_accuracy=None, top_k_hit_rates={k: None for k in config.top_k},
                    log_loss=None, brier=None, ece=None, roi=None, max_drawdown=None,
                    max_consecutive_losses=None, activation_rate=0.0, classwise_ece=None,
                    status='unavailable', reason='No evaluable predictions')
    return summaries


def _format_interval(interval: MetricInterval, percent: bool = False) -> str:
    if interval.mean is None:
        return 'N/A'
    scale = 100 if percent else 1
    if interval.lower is None or interval.upper is None:
        return f'{interval.mean * scale:.4f} (interval not computed)'
    return f'{interval.mean * scale:.4f} [{interval.lower * scale:.4f}, {interval.upper * scale:.4f}]'


def _attach_model_comparisons(summaries, grouped, config, cancel_event=None):
    def identity(row):
        return row.run_id, row.session_id, row.spin_id or row.index
    baseline = {identity(row): row for row in grouped.get(config.comparison_baseline, [])}
    if not baseline:
        return
    for model, rows in grouped.items():
        if model == config.comparison_baseline:
            continue
        paired = [row for row in rows if identity(row) in baseline]
        if not paired:
            continue
        references = [baseline[identity(row)] for row in paired]
        metric_names = ['exact_accuracy', 'log_loss', 'brier', 'roi', 'profit_per_spin', f'top_{config.bet_top_n}_hit']
        def deltas(model_rows, baseline_rows):
            model_values = _metric_values(model_rows, config, calibration=False)
            baseline_values = _metric_values(baseline_rows, config, calibration=False)
            return {metric: (baseline_values[metric] - model_values[metric] if metric in ('log_loss', 'brier')
                             else model_values[metric] - baseline_values[metric])
                    for metric in metric_names
                    if model_values.get(metric) is not None and baseline_values.get(metric) is not None}
        points = deltas(paired, references)
        sampled = defaultdict(list)
        if config.compute_intervals and config.bootstrap_resamples:
            for indices in block_bootstrap_indices(paired, config.bootstrap_resamples, config.seed, config.block_length):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError('Evaluation cancelled')
                for metric, value in deltas([paired[index] for index in indices], [references[index] for index in indices]).items():
                    sampled[metric].append(value)
        for metric, point in points.items():
            interval = _interval(point, sampled[metric], config)
            fraction = sum(value > 0 for value in sampled[metric]) / len(sampled[metric]) if sampled[metric] else None
            summaries[model].comparisons[metric] = ModelComparisonSummary(
                model, config.comparison_baseline, metric, point, interval.lower, interval.upper, fraction)


def _confidence_interval(values: List[float], confidence_level: float) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    confidence = max(0.0, min(1.0, confidence_level))
    lower_percentile = (1.0 - confidence) / 2.0
    upper_percentile = 1.0 - lower_percentile
    sorted_values = sorted(values)
    return (
        _percentile(sorted_values, lower_percentile),
        _percentile(sorted_values, upper_percentile),
    )


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, percentile)) * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def _row_exact_hit(row: PredictionEvaluationRow) -> float:
    return 1.0 if row.predicted == row.actual else 0.0


def _row_top_hit(row: PredictionEvaluationRow, limit: int) -> float:
    return 1.0 if row.actual in [number for number, _ in row.top_numbers[:limit]] else 0.0


def _row_log_loss(row: PredictionEvaluationRow) -> float:
    return -log(max(EPSILON, row.number_probs.get(row.actual, 0.0)))


def _row_stake(row, bet_top_n):
    return row.total_stake if row.total_stake is not None else min(bet_top_n, len(row.top_numbers))


def _fair_probs() -> Dict[int, float]:
    return {number: FAIR_PROBABILITY for number in range(37)}


def _rolling_frequency_probs(history: List[int], alpha: float = 1.0) -> Dict[int, float]:
    counts = Counter(history)
    total = len(history) + alpha * 37
    return {
        number: (counts.get(number, 0) + alpha) / total
        for number in range(37)
    }


def _selected_mixture_probs(selected_numbers: List[int]) -> Dict[int, float]:
    if not selected_numbers:
        return _fair_probs()

    probs = {number: FAIR_PROBABILITY * 0.30 for number in range(37)}
    selected_mass = 0.70 / len(selected_numbers)

    for number in selected_numbers:
        probs[number] += selected_mass

    return _normalize_probs(probs)


def _last_n_numbers(history: List[int], last_n: int, max_numbers: int) -> List[int]:
    selected = []
    for number in reversed(history[-last_n:]):
        if number not in selected:
            selected.append(number)
        if len(selected) >= max_numbers:
            break
    return list(reversed(selected))


def _hot_numbers(history: List[int], max_numbers: int) -> List[int]:
    return [number for number, _ in Counter(history).most_common(max_numbers)]


def _normalize_probs(probs: Dict[int, float]) -> Dict[int, float]:
    vector = validate_probabilities(probs)
    return {number: float(vector[number]) for number in range(37)}


def _top_numbers(probs: Dict[int, float], limit: int) -> List[Tuple[int, float]]:
    return sorted(probs.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _flat_profit(actual: int, top_numbers: List[Tuple[int, float]]) -> float:
    return settle_bet(actual, [number for number, _ in top_numbers], 1.0, float(len(top_numbers))).net_profit


def _brier(probs: Dict[int, float], actual: int) -> float:
    return sum(
        (probs.get(number, 0.0) - (1.0 if number == actual else 0.0)) ** 2
        for number in range(37)
    )


def _ece(rows: List[PredictionEvaluationRow], bins: int) -> float:
    return _binned_calibration_error(
        [
            (row.confidence, _row_exact_hit(row))
            for row in rows
        ],
        bins,
    )


def _top_k_ece(rows: List[PredictionEvaluationRow], top_k: int, bins: int) -> float:
    if top_k <= 1:
        return _ece(rows, bins)
    return _binned_calibration_error(
        [
            (
                sum(probability for _, probability in row.top_numbers[:top_k]),
                _row_top_hit(row, top_k),
            )
            for row in rows
        ],
        bins,
    )


def _classwise_ece(rows: List[PredictionEvaluationRow], bins: int) -> float:
    if not rows:
        return None
    return sum(calibration_error([row.number_probs[number] for row in rows],
                                 [float(row.actual == number) for row in rows], bins)
               for number in range(37)) / 37


def _binned_calibration_error(pairs: List[Tuple[float, float]], bins: int) -> float:
    return calibration_error([pair[0] for pair in pairs], [pair[1] for pair in pairs], bins)


def _calibration_bins(rows: List[PredictionEvaluationRow], bins: int) -> List[CalibrationBin]:
    runs = defaultdict(list)
    for row in rows:
        runs[row.run_id].append(row)
    return [CalibrationBin(*values, run_id=run_id)
            for run_id, run_rows in sorted(runs.items())
            for values in adaptive_bins([row.confidence for row in run_rows],
                                        [_row_exact_hit(row) for row in run_rows], bins)]


def _roi(rows: List[PredictionEvaluationRow], bet_top_n: int):
    stake = sum(_row_stake(row, bet_top_n) for row in rows)
    return sum(row.profit for row in rows) / stake if stake > 0 else None


def _max_drawdown(profits: List[float]) -> float:
    balance = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for profit in profits:
        balance += profit
        peak = max(peak, balance)
        max_drawdown = max(max_drawdown, peak - balance)

    return max_drawdown


def _max_consecutive_losses(profits: List[float]) -> int:
    current = 0
    maximum = 0

    for profit in profits:
        if profit < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0

    return maximum
