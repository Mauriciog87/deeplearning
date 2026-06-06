from dataclasses import dataclass, field
from typing import Dict, List
from collections import Counter

from src.utils.bias_detection import analyze_all_numbers, formal_chi_square_test
from src.utils.predictor import FAIR_PROBABILITY


@dataclass
class BiasAwareConfig:
    min_spins: int = 200
    min_probability: float = 0.03
    last_n: int = 18
    max_numbers: int = 12
    smoothing_alpha: float = 1.0
    significance_level: float = 0.05
    confidence_cap: float = 0.20


@dataclass
class BiasAwareResult:
    number_probs: Dict[int, float]
    selected_numbers: List[int]
    confidence: float
    is_significant: bool
    reason: str
    sample_size: int
    diagnostics: Dict[str, float] = field(default_factory=dict)


class BiasAwarePredictor:
    def __init__(self, config: BiasAwareConfig | None = None):
        self.config = config or BiasAwareConfig()

    def fit_predict(self, history: List[int]) -> BiasAwareResult:
        valid_history = [n for n in history if 0 <= n <= 36]
        sample_size = len(valid_history)

        if sample_size < self.config.min_spins:
            return BiasAwareResult(
                number_probs=self._uniform_probs(),
                selected_numbers=[],
                confidence=0.0,
                is_significant=False,
                reason=f"Need at least {self.config.min_spins} spins",
                sample_size=sample_size,
            )

        chi_square = formal_chi_square_test(valid_history)
        if "error" in chi_square:
            return BiasAwareResult(
                number_probs=self._uniform_probs(),
                selected_numbers=[],
                confidence=0.0,
                is_significant=False,
                reason=chi_square["error"],
                sample_size=sample_size,
            )

        number_analysis = analyze_all_numbers(valid_history)
        threshold_numbers = [
            item["number"]
            for item in number_analysis
            if item["probability"] >= self.config.min_probability
            and item["confidence_interval"][0] > FAIR_PROBABILITY
        ]
        bunch_numbers = self._last_n_bunching_numbers(valid_history)
        selected_numbers = self._merge_numbers(threshold_numbers, bunch_numbers)

        p_value = float(chi_square["p_value"])
        is_significant = p_value < self.config.significance_level and bool(selected_numbers)

        if not is_significant:
            return BiasAwareResult(
                number_probs=self._uniform_probs(),
                selected_numbers=[],
                confidence=0.0,
                is_significant=False,
                reason="No statistically significant wheel bias",
                sample_size=sample_size,
                diagnostics={
                    "chi_square": float(chi_square["chi_square"]),
                    "p_value": p_value,
                },
            )

        number_probs = self._build_probabilities(valid_history, selected_numbers)
        confidence = self._compute_confidence(number_probs, selected_numbers, p_value)

        return BiasAwareResult(
            number_probs=number_probs,
            selected_numbers=selected_numbers,
            confidence=confidence,
            is_significant=True,
            reason="Significant wheel bias detected",
            sample_size=sample_size,
            diagnostics={
                "chi_square": float(chi_square["chi_square"]),
                "p_value": p_value,
            },
        )

    def _last_n_bunching_numbers(self, history: List[int]) -> List[int]:
        recent = history[-self.config.last_n:]
        counts = Counter(recent)
        return [
            number
            for number, _ in counts.most_common(self.config.max_numbers)
            if number in history
        ]

    def _merge_numbers(self, threshold_numbers: List[int], bunch_numbers: List[int]) -> List[int]:
        merged = []
        for number in threshold_numbers + bunch_numbers:
            if number not in merged:
                merged.append(number)
            if len(merged) >= self.config.max_numbers:
                break
        return merged

    def _build_probabilities(self, history: List[int], selected_numbers: List[int]) -> Dict[int, float]:
        counts = Counter(history)
        total = len(history) + self.config.smoothing_alpha * 37
        probs = {
            number: (counts.get(number, 0) + self.config.smoothing_alpha) / total
            for number in range(37)
        }

        if not selected_numbers:
            return self._normalize(probs)

        selected_mass = sum(probs[number] for number in selected_numbers)
        baseline_mass = FAIR_PROBABILITY * len(selected_numbers)
        lift = max(0.0, min(self.config.confidence_cap, selected_mass - baseline_mass))
        non_selected = [number for number in range(37) if number not in selected_numbers]

        for number in selected_numbers:
            probs[number] += lift / len(selected_numbers)
        for number in non_selected:
            probs[number] = max(0.0, probs[number] - lift / len(non_selected))

        return self._normalize(probs)

    def _compute_confidence(
        self,
        number_probs: Dict[int, float],
        selected_numbers: List[int],
        p_value: float,
    ) -> float:
        if not selected_numbers:
            return 0.0
        selected_mass = sum(number_probs[number] for number in selected_numbers)
        baseline_mass = FAIR_PROBABILITY * len(selected_numbers)
        evidence = min(1.0, max(0.0, 1.0 - p_value / self.config.significance_level))
        lift = max(0.0, selected_mass - baseline_mass)
        return min(self.config.confidence_cap, lift * evidence)

    def _uniform_probs(self) -> Dict[int, float]:
        return {number: FAIR_PROBABILITY for number in range(37)}

    def _normalize(self, probs: Dict[int, float]) -> Dict[int, float]:
        total = sum(probs.values())
        if total <= 0:
            return self._uniform_probs()
        return {number: prob / total for number, prob in probs.items()}
