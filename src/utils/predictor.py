import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from dataclasses import dataclass

from src.database.models import Spin, Prediction
from src.probabilities import validate_probabilities, ranked_numbers, frequency_probabilities


PROBABILITY_THRESHOLD = 0.03  # 3% filter from Salirrosas (2016)
FAIR_PROBABILITY = 1 / 37  # ~2.7%


def __getattr__(name):
    if name in ('LSTMPredictor', 'LSTMNetwork', 'get_optimal_device'):
        from . import lstm_predictor
        return getattr(lstm_predictor, name)
    raise AttributeError(name)


@dataclass
class PredictionWithConfidence:
    value: Any
    confidence: float
    confidence_interval: Optional[Tuple[float, float]] = None
    passes_threshold: bool = False
    method: str = "statistical"


class ExtraTreesPredictor:
    """
    Extra Trees based predictor from Merchie (2018) thesis.
    
    Extra Trees (Extremely Randomized Trees) showed best performance
    for casino game anomaly detection in the thesis.
    """
    
    def __init__(self, seed: int = 42):
        self.model = None
        self.is_fitted = False
        self.seed = seed
        self.window_size = 10
    
    def fit(self, history: List[int], window_size: int = 10):
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            
            self.window_size = window_size
            if len(history) < window_size + 1:
                print(f"[ExtraTrees] Not enough history: {len(history)} < {window_size + 1}")
                return {'status': 'insufficient_data', 'message': 'Not enough history'}
            
            X = []
            y = []
            
            for i in range(window_size, len(history)):
                features = self._extract_features(history[i-window_size:i])
                X.append(features)
                y.append(history[i])
            
            unique_classes = len(set(y))
            if unique_classes < 1:
                print(f"[ExtraTrees] Not enough unique classes: {unique_classes}")
                return
            
            self.model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.seed,
                n_jobs=1
            )
            self.model.fit(X, y)
            self.is_fitted = True
            return {'status': 'success', 'samples': len(X)}
        except ImportError as e:
            print(f"[ExtraTrees] sklearn not installed: {e}")
            self.is_fitted = False
            return {'status': 'unavailable', 'message': str(e)}
        except Exception as e:
            print(f"[ExtraTrees] Error during fit: {e}")
            self.is_fitted = False
            return {'status': 'failed', 'message': str(e)}
    
    def predict_proba(self, history: List[int]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError('Extra Trees is untrained')
        if len(history) < self.window_size:
            raise ValueError('Insufficient history for Extra Trees inference')
        features = self._extract_features(history[-self.window_size:])
        probabilities = np.zeros(37, dtype=float)
        probabilities[np.asarray(self.model.classes_, dtype=int)] = self.model.predict_proba([features])[0]
        return validate_probabilities(probabilities)

    def predict(self, history: List[int], window_size: int = 10) -> Tuple[int, float]:
        return ranked_numbers(self.predict_proba(history))[0]

    def _extract_features(self, window: List[int]) -> List[float]:
        features = []
        
        features.extend([float(n) / 36 for n in window])
        
        red_count = sum(1 for n in window if n in [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36])
        features.append(red_count / len(window))
        
        odd_count = sum(1 for n in window if n != 0 and n % 2 == 1)
        features.append(odd_count / len(window))
        
        high_count = sum(1 for n in window if n >= 19)
        features.append(high_count / len(window))
        
        for d in range(1, 4):
            dozen_count = sum(1 for n in window if (n-1)//12 + 1 == d)
            features.append(dozen_count / len(window))
        
        counter = Counter(window)
        features.append(len(counter) / 37)
        
        return features


class RoulettePredictor:
    
    def __init__(self, model_path: Optional[str] = None, use_extra_trees: bool = False):
        self.model = None
        self.device = "cpu"
        self.history_size = 20
        
        self.use_extra_trees = use_extra_trees
        self.extra_trees = ExtraTreesPredictor() if use_extra_trees else None
        
        self.consecutive_losses = 0
        self.loss_history: List[bool] = []
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        raise ValueError('DQN checkpoints provide betting policies. Use PredictionEngine.get_policy_decision().')

    def train_extra_trees(self, history: List[int]):
        """Train Extra Trees model on historical data."""
        if self.extra_trees is None:
            self.extra_trees = ExtraTreesPredictor()
        self.extra_trees.fit(history)
        self.use_extra_trees = self.extra_trees.is_fitted
    
    def record_result(self, won: bool):
        """Track prediction results for consecutive loss analysis."""
        self.loss_history.append(won)
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get statistics about prediction losses."""
        if not self.loss_history:
            return {"total": 0, "wins": 0, "losses": 0, "current_streak": 0}
        
        wins = sum(self.loss_history)
        losses = len(self.loss_history) - wins
        
        max_loss_streak = 0
        current_streak = 0
        loss_streaks = []
        
        for won in self.loss_history:
            if not won:
                current_streak += 1
            else:
                if current_streak > 0:
                    loss_streaks.append(current_streak)
                    max_loss_streak = max(max_loss_streak, current_streak)
                current_streak = 0
        
        if current_streak > 0:
            loss_streaks.append(current_streak)
            max_loss_streak = max(max_loss_streak, current_streak)
        
        return {
            "total": len(self.loss_history),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self.loss_history) if self.loss_history else 0,
            "current_loss_streak": self.consecutive_losses,
            "max_loss_streak": max_loss_streak,
            "loss_streak_histogram": Counter(loss_streaks)
        }
    
    def predict_proba(self, history):
        if self.use_extra_trees and self.extra_trees is not None:
            return self.extra_trees.predict_proba(history)
        return frequency_probabilities(history)

    def predict_from_history(self, history: List[int]) -> Dict:
        from src.engine.prediction_engine import PredictionEngine, PredictorType
        prediction = PredictionEngine().prediction_from_probabilities(
            PredictorType.EXTRA_TREES if self.use_extra_trees else PredictorType.BIAS,
            self.predict_proba(history))
        return {name: (getattr(prediction, name).value, getattr(prediction, name).probability)
                for name in ('number', 'color', 'parity', 'high_low', 'dozen', 'column')}

    def predict_with_filter(
        self, 
        history: List[int],
        min_probability: float = PROBABILITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Enhanced prediction with 3% probability filter from Salirrosas (2016).
        
        Only returns numbers that pass the probability threshold.
        """
        if len(history) < 100:
            return {
                "filtered_numbers": [],
                "message": "Need at least 100 spins for filtered prediction",
                "standard_prediction": self.predict_from_history(history)
            }
        
        counter = Counter(history)
        n = len(history)
        
        filtered = []
        for num in range(37):
            prob = counter.get(num, 0) / n
            if prob >= min_probability:
                ci = self._wilson_interval(counter.get(num, 0), n)
                filtered.append({
                    "number": num,
                    "probability": prob,
                    "confidence_interval": ci,
                    "passes_significance": ci[0] > FAIR_PROBABILITY,
                    "advantage_pct": (prob - FAIR_PROBABILITY) * 100
                })
        
        filtered.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "filtered_numbers": filtered,
            "count": len(filtered),
            "threshold": min_probability,
            "standard_prediction": self.predict_from_history(history)
        }
    
    def _wilson_interval(
        self, 
        successes: int, 
        n: int, 
        z: float = 1.96
    ) -> Tuple[float, float]:
        """Wilson score confidence interval for binomial proportion."""
        if n == 0:
            return (0.0, 1.0)
        
        p_hat = successes / n
        denominator = 1 + z**2 / n
        center = p_hat + z**2 / (2 * n)
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
        
        lower = max(0, (center - spread) / denominator)
        upper = min(1, (center + spread) / denominator)
        
        return (lower, upper)
    
    def create_prediction_object(self, predictions: Dict, session_id: int) -> Prediction:
        return Prediction(
            session_id=session_id,
            predicted_number=predictions["number"][0],
            predicted_color=predictions["color"][0],
            predicted_parity=predictions["parity"][0],
            predicted_high_low=predictions["high_low"][0],
            predicted_dozen=predictions["dozen"][0],
            predicted_column=predictions["column"][0]
        )
    
    def format_prediction(self, predictions: Dict) -> str:
        num, num_conf = predictions["number"]
        color, color_conf = predictions["color"]
        parity, parity_conf = predictions["parity"]
        high_low, hl_conf = predictions["high_low"]
        dozen, dozen_conf = predictions["dozen"]
        column, col_conf = predictions["column"]
        
        spin = Spin(number=num)
        color_emoji = "🔴" if spin.color == "red" else "⚫" if spin.color == "black" else "🟢"
        
        lines = [
            f"🎯 Número: {num} ({color_emoji} {spin.color.capitalize()})",
            f"🎨 Color: {color.capitalize()} ({color_conf*100:.1f}%)",
            f"🔢 Paridad: {parity.capitalize()} ({parity_conf*100:.1f}%)",
            f"📊 Alto/Bajo: {high_low.capitalize()} ({hl_conf*100:.1f}%)",
            f"📈 Docena: {dozen} ({dozen_conf*100:.1f}%)",
            f"📉 Columna: {column} ({col_conf*100:.1f}%)"
        ]
        
        return "\n".join(lines)
