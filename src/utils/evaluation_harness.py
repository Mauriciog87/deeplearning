from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import isfinite, log
from random import Random
from typing import Dict, List, Optional, Tuple

from src.engine.prediction_engine import FullPrediction, PredictionEngine
from src.utils.predictor import FAIR_PROBABILITY


EPSILON = 1e-12
STRAIGHT_UP_PAYOUT = 35.0
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


@dataclass
class PredictionEvaluationRow:
    fold: int
    index: int
    model: str
    actual: int
    predicted: int
    confidence: float
    number_probs: Dict[int, float]
    top_numbers: List[Tuple[int, float]]
    profit: float


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


@dataclass
class WalkForwardEvaluationResult:
    config: EvaluationConfig
    folds: int
    per_model_summaries: Dict[str, ModelEvaluationSummary]
    warnings: List[str] = field(default_factory=list)
    rows: List[PredictionEvaluationRow] = field(default_factory=list)


def evaluate_walk_forward(
    numbers: List[int],
    config: Optional[EvaluationConfig] = None,
) -> WalkForwardEvaluationResult:
    config = config or EvaluationConfig()
    clean_numbers = [number for number in numbers if 0 <= number <= 36]
    warnings = []
    rows = []

    if len(clean_numbers) != len(numbers):
        warnings.append("Invalid roulette numbers were ignored.")

    required = config.training_window + config.testing_window
    if len(clean_numbers) < required:
        warnings.append(
            f"Need at least {required} spins for evaluation "
            f"(received {len(clean_numbers)})."
        )
        return WalkForwardEvaluationResult(
            config=config,
            folds=0,
            per_model_summaries={},
            warnings=warnings,
            rows=[],
        )

    fold = 0
    start = 0
    total_evaluation_spins = 0

    while start + required <= len(clean_numbers):
        train_data = clean_numbers[start:start + config.training_window]
        test_start = start + config.training_window
        test_data = clean_numbers[test_start:test_start + config.testing_window]
        engine = PredictionEngine(model_path=config.model_path)

        for number in train_data:
            engine.add_number(number)

        rng = Random(config.seed + fold)

        for offset, actual in enumerate(test_data):
            index = test_start + offset
            rows.extend(_engine_rows(engine, fold, index, actual, config))

            if config.include_baselines:
                rows.extend(
                    _baseline_rows(
                        engine.history,
                        fold,
                        index,
                        actual,
                        config,
                        rng,
                    )
                )

            engine.add_number(actual)

        total_evaluation_spins += len(test_data)
        fold += 1
        start += config.step_size

    summaries = _summarize_rows(rows, config, total_evaluation_spins)

    return WalkForwardEvaluationResult(
        config=config,
        folds=fold,
        per_model_summaries=summaries,
        warnings=warnings,
        rows=rows,
    )


def format_evaluation_report(result: WalkForwardEvaluationResult) -> str:
    lines = [
        "=" * 96,
        "WALK-FORWARD ENGINE EVALUATION",
        "=" * 96,
        f"Folds: {result.folds}",
        (
            f"Windows: train={result.config.training_window}, "
            f"test={result.config.testing_window}, step={result.config.step_size}"
        ),
        f"Bet top N: {result.config.bet_top_n}",
    ]

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"- {warning}")

    if not result.per_model_summaries:
        return "\n".join(lines)

    lines.extend([
        "",
        (
            f"{'Model':<18} {'Act%':>7} {'Acc@1':>7} {'Top3':>7} "
            f"{'Top5':>7} {'LogLoss':>9} {'Brier':>8} {'ECE':>7} "
            f"{'ROI':>8} {'MaxDD':>9} {'MaxL':>6}"
        ),
        "-" * 96,
    ])

    for model in MODEL_ORDER:
        summary = result.per_model_summaries.get(model)
        if summary is None:
            continue
        lines.append(
            f"{model:<18} "
            f"{summary.activation_rate * 100:>6.1f}% "
            f"{summary.exact_accuracy * 100:>6.1f}% "
            f"{summary.top_k_hit_rates.get(3, 0.0) * 100:>6.1f}% "
            f"{summary.top_k_hit_rates.get(5, 0.0) * 100:>6.1f}% "
            f"{summary.log_loss:>9.4f} "
            f"{summary.brier:>8.4f} "
            f"{summary.ece:>7.4f} "
            f"{summary.roi * 100:>7.1f}% "
            f"{summary.max_drawdown:>9.2f} "
            f"{summary.max_consecutive_losses:>6}"
        )

    return "\n".join(lines)


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
        if prediction is None:
            continue
        rows.append(_row_from_prediction(
            fold,
            index,
            predictor_type.value,
            actual,
            prediction,
            config,
        ))

    consensus = engine.get_consensus_prediction()
    if consensus is not None:
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
    top_numbers = prediction.top_numbers or _top_numbers(number_probs, config.bet_top_n)
    confidence = max(number_probs.values()) if number_probs else 0.0

    return PredictionEvaluationRow(
        fold=fold,
        index=index,
        model=model,
        actual=actual,
        predicted=int(prediction.number.value),
        confidence=confidence,
        number_probs=number_probs,
        top_numbers=top_numbers,
        profit=_flat_profit(actual, top_numbers[:config.bet_top_n]),
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
    top_numbers = _top_numbers(number_probs, config.bet_top_n)
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


def _summarize_rows(
    rows: List[PredictionEvaluationRow],
    config: EvaluationConfig,
    total_evaluation_spins: int,
) -> Dict[str, ModelEvaluationSummary]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.model].append(row)

    summaries = {}

    for model, model_rows in grouped.items():
        count = len(model_rows)
        if count == 0:
            continue

        exact_hits = [1.0 if row.predicted == row.actual else 0.0 for row in model_rows]
        top_k_hit_rates = {
            k: _mean([
                1.0 if row.actual in [number for number, _ in row.top_numbers[:k]] else 0.0
                for row in model_rows
            ])
            for k in config.top_k
        }
        log_losses = [
            -log(max(EPSILON, row.number_probs.get(row.actual, 0.0)))
            for row in model_rows
        ]
        briers = [_brier(row.number_probs, row.actual) for row in model_rows]
        profits = [row.profit for row in model_rows]

        summaries[model] = ModelEvaluationSummary(
            spins=count,
            exact_accuracy=_mean(exact_hits),
            top_k_hit_rates=top_k_hit_rates,
            log_loss=_safe_mean(log_losses),
            brier=_safe_mean(briers),
            ece=_ece(model_rows, config.ece_bins),
            roi=_roi(model_rows, config.bet_top_n),
            max_drawdown=_max_drawdown(profits),
            max_consecutive_losses=_max_consecutive_losses(profits),
            activation_rate=count / total_evaluation_spins if total_evaluation_spins else 0.0,
        )

    if total_evaluation_spins:
        for model in MODEL_ORDER:
            summaries.setdefault(
                model,
                ModelEvaluationSummary(
                    spins=0,
                    exact_accuracy=0.0,
                    top_k_hit_rates={k: 0.0 for k in config.top_k},
                    log_loss=0.0,
                    brier=0.0,
                    ece=0.0,
                    roi=0.0,
                    max_drawdown=0.0,
                    max_consecutive_losses=0,
                    activation_rate=0.0,
                ),
            )

    return summaries


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
    sanitized = {
        number: max(0.0, float(probs.get(number, 0.0)))
        for number in range(37)
    }
    total = sum(sanitized.values())
    if total <= 0.0:
        return _fair_probs()
    return {number: value / total for number, value in sanitized.items()}


def _top_numbers(probs: Dict[int, float], limit: int) -> List[Tuple[int, float]]:
    return sorted(probs.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _flat_profit(actual: int, top_numbers: List[Tuple[int, float]]) -> float:
    selected = [number for number, _ in top_numbers]
    if not selected:
        return 0.0
    if actual in selected:
        return STRAIGHT_UP_PAYOUT - (len(selected) - 1)
    return -float(len(selected))


def _brier(probs: Dict[int, float], actual: int) -> float:
    return sum(
        (probs.get(number, 0.0) - (1.0 if number == actual else 0.0)) ** 2
        for number in range(37)
    )


def _ece(rows: List[PredictionEvaluationRow], bins: int) -> float:
    if not rows or bins <= 0:
        return 0.0

    total = len(rows)
    error = 0.0

    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            bucket = [row for row in rows if lower <= row.confidence <= upper]
        else:
            bucket = [row for row in rows if lower <= row.confidence < upper]

        if not bucket:
            continue

        accuracy = _mean([1.0 if row.predicted == row.actual else 0.0 for row in bucket])
        confidence = _mean([row.confidence for row in bucket])
        error += (len(bucket) / total) * abs(accuracy - confidence)

    return error


def _roi(rows: List[PredictionEvaluationRow], bet_top_n: int) -> float:
    stake = sum(min(bet_top_n, len(row.top_numbers)) for row in rows)
    if stake <= 0:
        return 0.0
    return sum(row.profit for row in rows) / stake


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


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_mean(values: List[float]) -> float:
    finite_values = [value for value in values if isfinite(value)]
    return _mean(finite_values)
