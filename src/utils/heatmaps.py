from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from src.utils.multiple_testing import false_discovery_control
from src.settlement import validate_number


FAIR_PROBABILITY = 1.0 / 37.0
MIN_STATIC_SPINS = 50
WHEEL_ORDER = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13,
    36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20,
    14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26,
]
TABLE_LAYOUT = np.array([
    [-1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
    [-1, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34],
])
RED_NUMBERS = {
    1, 3, 5, 7, 9, 12, 14, 16, 18,
    19, 21, 23, 25, 27, 30, 32, 34, 36,
}
CATEGORY_LABELS = [
    "red",
    "black",
    "green",
    "odd",
    "even",
    "low",
    "high",
    "dozen_1",
    "dozen_2",
    "dozen_3",
    "column_1",
    "column_2",
    "column_3",
]
CATEGORY_FAIR_PROBS = np.array([
    18 / 37,
    18 / 37,
    1 / 37,
    18 / 37,
    18 / 37,
    18 / 37,
    18 / 37,
    12 / 37,
    12 / 37,
    12 / 37,
    12 / 37,
    12 / 37,
    12 / 37,
])


@dataclass
class HeatmapConfig:
    window_size: int = 100
    step_size: int = 25
    sector_count: int = 12
    output_dir: str = "reports/heatmaps"
    show: bool = False
    dpi: int = 150
    z_threshold: float = 2.5
    fdr_alpha: float = 0.05
    fdr_method: str = 'by'
    include_anomalies: bool = True


@dataclass
class HeatmapAnomaly:
    number: Optional[int]
    feature: str
    window_label: str
    observed_rate: float
    expected_rate: float
    residual: float
    z_score: float
    p_value: float
    severity: str
    q_value: float = 1.0
    fdr_significant: bool = False


@dataclass
class SessionDriftResult:
    session_a: str
    session_b: str
    distance: float
    top_differences: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class HeatmapResult:
    total_spins: int
    source_label: str
    output_paths: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    anomalies: List[HeatmapAnomaly] = field(default_factory=list)
    drift_results: List[SessionDriftResult] = field(default_factory=list)


def clean_numbers(numbers: List[int]) -> List[int]:
    return [validate_number(number) for number in numbers]


def build_number_counts(numbers: List[int]) -> np.ndarray:
    valid_numbers = clean_numbers(numbers)
    counts = np.zeros(37, dtype=float)
    for number in valid_numbers:
        counts[number] += 1
    return counts


def build_number_residuals(numbers: List[int]) -> np.ndarray:
    counts = build_number_counts(numbers)
    total = counts.sum()
    if total <= 0:
        return np.zeros(37, dtype=float)
    return counts / total - FAIR_PROBABILITY


def build_table_heatmap_matrix(numbers: List[int]) -> np.ndarray:
    residuals = build_number_residuals(numbers)
    matrix = np.full(TABLE_LAYOUT.shape, np.nan, dtype=float)
    for row_index in range(TABLE_LAYOUT.shape[0]):
        for column_index in range(TABLE_LAYOUT.shape[1]):
            number = int(TABLE_LAYOUT[row_index, column_index])
            if number >= 0:
                matrix[row_index, column_index] = residuals[number]
    return matrix


def build_wheel_sector_residuals(
    numbers: List[int],
    sector_count: int = 12,
) -> Tuple[np.ndarray, List[str]]:
    valid_numbers = clean_numbers(numbers)
    if not valid_numbers or sector_count <= 0:
        return np.zeros(0, dtype=float), []

    counts = build_number_counts(valid_numbers)
    total = counts.sum()
    sectors = np.array_split(np.array(WHEEL_ORDER), sector_count)
    values = []
    labels = []
    for index, sector in enumerate(sectors, 1):
        observed = counts[sector].sum() / total
        expected = len(sector) / 37
        values.append(observed - expected)
        labels.append(f"S{index}")
    return np.array(values, dtype=float), labels


def build_rolling_number_residuals(
    numbers: List[int],
    window_size: int = 100,
    step_size: int = 25,
) -> Tuple[np.ndarray, List[str]]:
    valid_numbers = clean_numbers(numbers)
    starts = _rolling_starts(len(valid_numbers), window_size, step_size)
    matrix = [
        build_number_residuals(valid_numbers[start:start + window_size])
        for start in starts
    ]
    labels = [
        f"{start + 1}-{start + window_size}"
        for start in starts
    ]
    if not matrix:
        return np.zeros((0, 37), dtype=float), []
    return np.array(matrix, dtype=float), labels


def build_rolling_category_residuals(
    numbers: List[int],
    window_size: int = 100,
    step_size: int = 25,
) -> Tuple[np.ndarray, List[str], List[str]]:
    valid_numbers = clean_numbers(numbers)
    starts = _rolling_starts(len(valid_numbers), window_size, step_size)
    matrix = [
        _category_rates(valid_numbers[start:start + window_size]) - CATEGORY_FAIR_PROBS
        for start in starts
    ]
    labels = [
        f"{start + 1}-{start + window_size}"
        for start in starts
    ]
    if not matrix:
        return np.zeros((0, len(CATEGORY_LABELS)), dtype=float), [], CATEGORY_LABELS.copy()
    return np.array(matrix, dtype=float), labels, CATEGORY_LABELS.copy()


def detect_heatmap_anomalies(
    numbers: List[int],
    window_size: int = 100,
    step_size: int = 25,
    z_threshold: float = 2.5,
    sector_count: int = 12,
    fdr_alpha: float = 0.05,
    fdr_method: str = 'by',
) -> List[HeatmapAnomaly]:
    valid_numbers = clean_numbers(numbers)
    if not valid_numbers:
        return []

    candidates = _window_anomalies(valid_numbers, "all", z_threshold, sector_count, include_all=True)
    for start in _rolling_starts(len(valid_numbers), window_size, step_size):
        window = valid_numbers[start:start + window_size]
        label = f"{start + 1}-{start + window_size}"
        candidates.extend(_window_anomalies(
            window,
            label,
            z_threshold,
            sector_count,
            include_all=True,
        ))

    q_values, rejected = false_discovery_control(
        [candidate.p_value for candidate in candidates],
        fdr_alpha, method=fdr_method,
    )
    for candidate, q_value, is_rejected in zip(candidates, q_values, rejected):
        candidate.q_value = q_value if q_value is not None else 1.0
        candidate.fdr_significant = is_rejected

    anomalies = [
        candidate
        for candidate in candidates
        if candidate.fdr_significant and abs(candidate.z_score) >= z_threshold
    ]
    return sorted(anomalies, key=lambda item: abs(item.z_score), reverse=True)


def build_session_drift_matrix(
    session_numbers: Dict[str, List[int]],
    metric: str = "l2",
) -> Tuple[np.ndarray, List[str], List[SessionDriftResult]]:
    labels = []
    vectors = []
    for label, numbers in session_numbers.items():
        valid_numbers = clean_numbers(numbers)
        if len(valid_numbers) < MIN_STATIC_SPINS:
            continue
        labels.append(label)
        vectors.append(_session_feature_vector(valid_numbers))

    if not vectors:
        return np.zeros((0, 0), dtype=float), [], []

    matrix = np.zeros((len(vectors), len(vectors)), dtype=float)
    results = []
    for left_index, left_vector in enumerate(vectors):
        for right_index in range(left_index + 1, len(vectors)):
            right_vector = vectors[right_index]
            distance = _feature_distance(left_vector, right_vector, metric)
            matrix[left_index, right_index] = distance
            matrix[right_index, left_index] = distance
            results.append(SessionDriftResult(
                session_a=labels[left_index],
                session_b=labels[right_index],
                distance=distance,
                top_differences=_top_feature_differences(left_vector, right_vector),
            ))
    return matrix, labels, results


def generate_heatmap_report(
    numbers: List[int],
    config: Optional[HeatmapConfig] = None,
    source_label: str = "global",
    session_numbers: Optional[Dict[str, List[int]]] = None,
) -> HeatmapResult:
    active_config = config or HeatmapConfig()
    valid_numbers = clean_numbers(numbers)
    if len(valid_numbers) < MIN_STATIC_SPINS:
        raise ValueError(
            f"Need at least {MIN_STATIC_SPINS} valid spins for heatmaps "
            f"(received {len(valid_numbers)})"
        )
    if active_config.window_size <= 0 or active_config.step_size <= 0:
        raise ValueError("window_size and step_size must be positive")
    if active_config.sector_count <= 0:
        raise ValueError("sector_count must be positive")

    output_dir = Path(active_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = HeatmapResult(total_spins=len(valid_numbers), source_label=source_label)
    result.warnings.append('Binomial anomaly tests assume independent spins. BY adjusts the tested feature/window family; it does not correct serial dependence or repeated monitoring.')
    title_suffix = f"{source_label} ({len(valid_numbers)} spins)"

    paths = {
        "number_table": output_dir / "number_table_heatmap.png",
        "wheel_sector": output_dir / "wheel_sector_heatmap.png",
    }

    _plot_table_heatmap(
        build_table_heatmap_matrix(valid_numbers),
        f"Table residual heatmap - {title_suffix}",
        paths["number_table"],
        active_config,
    )
    sectors, sector_labels = build_wheel_sector_residuals(
        valid_numbers,
        active_config.sector_count,
    )
    _plot_matrix_heatmap(
        sectors.reshape(1, -1),
        [""],
        sector_labels,
        f"Wheel sector residual heatmap - {title_suffix}",
        paths["wheel_sector"],
        active_config,
    )
    result.output_paths["number_table"] = str(paths["number_table"])
    result.output_paths["wheel_sector"] = str(paths["wheel_sector"])

    if active_config.include_anomalies:
        result.anomalies = detect_heatmap_anomalies(
            valid_numbers,
            active_config.window_size,
            active_config.step_size,
            active_config.z_threshold,
            active_config.sector_count,
            active_config.fdr_alpha,
            active_config.fdr_method,
        )
        anomaly_path = output_dir / "anomaly_summary.txt"
        _write_anomaly_summary(anomaly_path, result.anomalies, active_config.z_threshold)
        result.output_paths["anomaly_summary"] = str(anomaly_path)

    if session_numbers:
        drift_matrix, drift_labels, drift_results = build_session_drift_matrix(session_numbers)
        skipped = [
            label
            for label, values in session_numbers.items()
            if len(clean_numbers(values)) < MIN_STATIC_SPINS
        ]
        if skipped:
            result.warnings.append(
                "Skipped drift sessions with fewer than "
                f"{MIN_STATIC_SPINS} spins: {', '.join(skipped)}"
            )
        if drift_matrix.shape[0] >= 2:
            drift_path = output_dir / "session_drift_heatmap.png"
            _plot_matrix_heatmap(
                drift_matrix,
                drift_labels,
                drift_labels,
                f"Session drift heatmap - {source_label}",
                drift_path,
                active_config,
                colorbar_label="Feature distance",
            )
            result.output_paths["session_drift"] = str(drift_path)
            result.drift_results = drift_results
        elif len(session_numbers) > 0:
            result.warnings.append("Need at least 2 valid sessions for drift heatmap")

    number_matrix, window_labels = build_rolling_number_residuals(
        valid_numbers,
        active_config.window_size,
        active_config.step_size,
    )
    category_matrix, category_window_labels, category_labels = build_rolling_category_residuals(
        valid_numbers,
        active_config.window_size,
        active_config.step_size,
    )
    if number_matrix.size == 0:
        result.warnings.append(
            "Not enough spins for rolling heatmaps with the configured window and step"
        )
        return result

    rolling_paths = {
        "rolling_number": output_dir / "rolling_number_heatmap.png",
        "rolling_category": output_dir / "rolling_category_heatmap.png",
    }
    _plot_matrix_heatmap(
        number_matrix.T,
        [str(number) for number in range(37)],
        window_labels,
        f"Rolling number residual heatmap - {title_suffix}",
        rolling_paths["rolling_number"],
        active_config,
    )
    _plot_matrix_heatmap(
        category_matrix.T,
        category_labels,
        category_window_labels,
        f"Rolling category residual heatmap - {title_suffix}",
        rolling_paths["rolling_category"],
        active_config,
    )
    result.output_paths["rolling_number"] = str(rolling_paths["rolling_number"])
    result.output_paths["rolling_category"] = str(rolling_paths["rolling_category"])
    return result


def _rolling_starts(length: int, window_size: int, step_size: int) -> List[int]:
    if window_size <= 0 or step_size <= 0 or length < window_size + step_size:
        return []
    return list(range(0, length - window_size + 1, step_size))


def _window_anomalies(
    numbers: List[int],
    window_label: str,
    z_threshold: float,
    sector_count: int,
    include_all: bool = False,
) -> List[HeatmapAnomaly]:
    total = len(numbers)
    if total <= 0:
        return []

    anomalies = []
    counts = build_number_counts(numbers)
    for number in range(37):
        anomalies.extend(_maybe_anomaly(
            number=number,
            feature=f"number_{number}",
            window_label=window_label,
            observed_rate=counts[number] / total,
            expected_rate=FAIR_PROBABILITY,
            sample_size=total,
            z_threshold=z_threshold,
            include_all=include_all,
        ))

    category_rates = _category_rates(numbers)
    for index, label in enumerate(CATEGORY_LABELS):
        anomalies.extend(_maybe_anomaly(
            number=None,
            feature=label,
            window_label=window_label,
            observed_rate=float(category_rates[index]),
            expected_rate=float(CATEGORY_FAIR_PROBS[index]),
            sample_size=total,
            z_threshold=z_threshold,
            include_all=include_all,
        ))

    sectors = np.array_split(np.array(WHEEL_ORDER), sector_count)
    for index, sector in enumerate(sectors, 1):
        observed = counts[sector].sum() / total
        expected = len(sector) / 37
        anomalies.extend(_maybe_anomaly(
            number=None,
            feature=f"sector_{index}",
            window_label=window_label,
            observed_rate=float(observed),
            expected_rate=float(expected),
            sample_size=total,
            z_threshold=z_threshold,
            include_all=include_all,
        ))
    return anomalies


def _maybe_anomaly(
    number: Optional[int],
    feature: str,
    window_label: str,
    observed_rate: float,
    expected_rate: float,
    sample_size: int,
    z_threshold: float,
    include_all: bool = False,
) -> List[HeatmapAnomaly]:
    if sample_size <= 0 or expected_rate <= 0.0 or expected_rate >= 1.0:
        return []
    standard_error = float(np.sqrt(expected_rate * (1.0 - expected_rate) / sample_size))
    if standard_error <= 0.0:
        return []
    z_score = (observed_rate - expected_rate) / standard_error
    if not include_all and abs(z_score) < z_threshold:
        return []
    return [HeatmapAnomaly(
        number=number,
        feature=feature,
        window_label=window_label,
        observed_rate=observed_rate,
        expected_rate=expected_rate,
        residual=observed_rate - expected_rate,
        z_score=float(z_score),
        severity=_severity(z_score),
        p_value=float(binomtest(round(observed_rate * sample_size), sample_size, expected_rate).pvalue),
    )]


def _severity(z_score: float) -> str:
    absolute = abs(z_score)
    if absolute >= 4.0:
        return "extreme"
    if absolute >= 3.0:
        return "strong"
    return "watch"


def _session_feature_vector(numbers: List[int]) -> Dict[str, float]:
    counts = build_number_counts(numbers)
    total = counts.sum()
    if total <= 0:
        return {}
    values = {
        f"number_{number}": float(counts[number] / total)
        for number in range(37)
    }
    category_rates = _category_rates(numbers)
    values.update({
        label: float(category_rates[index])
        for index, label in enumerate(CATEGORY_LABELS)
    })
    sectors = np.array_split(np.array(WHEEL_ORDER), 12)
    for index, sector in enumerate(sectors, 1):
        values[f"sector_{index}"] = float(counts[sector].sum() / total)
    return values


def _feature_distance(
    left: Dict[str, float],
    right: Dict[str, float],
    metric: str,
) -> float:
    if metric != "l2":
        raise ValueError("Only l2 drift distance is supported")
    keys = sorted(set(left) | set(right))
    return float(np.sqrt(sum((left.get(key, 0.0) - right.get(key, 0.0)) ** 2 for key in keys)))


def _top_feature_differences(
    left: Dict[str, float],
    right: Dict[str, float],
    limit: int = 5,
) -> List[Tuple[str, float]]:
    keys = sorted(set(left) | set(right))
    differences = [
        (key, abs(left.get(key, 0.0) - right.get(key, 0.0)))
        for key in keys
    ]
    return sorted(differences, key=lambda item: item[1], reverse=True)[:limit]


def _write_anomaly_summary(
    output_path: Path,
    anomalies: List[HeatmapAnomaly],
    z_threshold: float,
) -> None:
    lines = [
        "Heatmap anomaly summary",
        f"z_threshold: {z_threshold}",
        f"anomalies: {len(anomalies)}",
        "",
    ]
    if not anomalies:
        lines.append("No anomalies detected.")
    else:
        lines.append("severity | window | feature | observed | expected | residual | z_score | p_value | q_value | fdr")
        lines.append("-" * 116)
        for anomaly in anomalies:
            lines.append(
                f"{anomaly.severity:<8} | "
                f"{anomaly.window_label:<11} | "
                f"{anomaly.feature:<12} | "
                f"{anomaly.observed_rate:.4f} | "
                f"{anomaly.expected_rate:.4f} | "
                f"{anomaly.residual:+.4f} | "
                f"{anomaly.z_score:+.3f} | "
                f"{anomaly.p_value:.6f} | "
                f"{anomaly.q_value:.6f} | "
                f"{'yes' if anomaly.fdr_significant else 'no'}"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _category_rates(numbers: List[int]) -> np.ndarray:
    if not numbers:
        return np.zeros(len(CATEGORY_LABELS), dtype=float)
    total = len(numbers)
    counts = np.zeros(len(CATEGORY_LABELS), dtype=float)
    for number in numbers:
        is_zero = number == 0
        is_red = number in RED_NUMBERS
        if is_zero:
            counts[2] += 1
        elif is_red:
            counts[0] += 1
        else:
            counts[1] += 1
        if not is_zero and number % 2 == 1:
            counts[3] += 1
        elif not is_zero:
            counts[4] += 1
        if 1 <= number <= 18:
            counts[5] += 1
        elif 19 <= number <= 36:
            counts[6] += 1
        if number > 0:
            counts[7 + (number - 1) // 12] += 1
            counts[10 + (number - 1) % 3] += 1
    return counts / total


def _plot_table_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    config: HeatmapConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    masked = np.ma.masked_invalid(matrix)
    limit = _color_limit(matrix)
    image = ax.imshow(masked, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for row_index in range(TABLE_LAYOUT.shape[0]):
        for column_index in range(TABLE_LAYOUT.shape[1]):
            number = int(TABLE_LAYOUT[row_index, column_index])
            if number >= 0:
                value = matrix[row_index, column_index] * 100
                ax.text(
                    column_index,
                    row_index,
                    f"{number}\n{value:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label="Observed probability minus fair probability")
    _save_figure(fig, output_path, config)


def _plot_matrix_heatmap(
    matrix: np.ndarray,
    y_labels: List[str],
    x_labels: List[str],
    title: str,
    output_path: Path,
    config: HeatmapConfig,
    colorbar_label: str = "Observed probability minus fair probability",
) -> None:
    height = max(3.5, min(12.0, 0.28 * max(1, len(y_labels)) + 2.0))
    width = max(8.0, min(18.0, 0.45 * max(1, len(x_labels)) + 4.0))
    fig, ax = plt.subplots(figsize=(width, height))
    limit = _color_limit(matrix)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(_tick_positions(len(x_labels)))
    ax.set_xticklabels([x_labels[index] for index in _tick_positions(len(x_labels))], rotation=45, ha="right")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    _save_figure(fig, output_path, config)


def _tick_positions(length: int) -> List[int]:
    if length <= 12:
        return list(range(length))
    step = max(1, length // 12)
    positions = list(range(0, length, step))
    if positions[-1] != length - 1:
        positions.append(length - 1)
    return positions


def _color_limit(matrix: np.ndarray) -> float:
    finite_values = np.asarray(matrix, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.01
    return max(float(np.max(np.abs(finite_values))), 0.01)


def _save_figure(fig, output_path: Path, config: HeatmapConfig) -> None:
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    if config.show:
        plt.show()
    plt.close(fig)
