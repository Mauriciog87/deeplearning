import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

from src.utils.predictor import ExtraTreesPredictor
from src.probabilities import (FAIR_PROBABILITY, Availability, ModelStatus, PolicyDecision,
                               validate_probabilities, ranked_numbers)
from src.settlement import validate_number
from src.utils.bias_aware_predictor import BiasAwarePredictor


class PredictorType(Enum):
    LSTM = "lstm"
    DQN = "dqn"
    EXTRA_TREES = "extra_trees"
    BIAS = "bias"
    CONSENSUS = "consensus"
    FAIR = "fair"


@dataclass
class CategoryPrediction:
    value: Any
    probability: float
    all_probabilities: Dict[Any, float] = field(default_factory=dict)


@dataclass
class FullPrediction:
    predictor: PredictorType
    number: CategoryPrediction
    color: CategoryPrediction
    parity: CategoryPrediction
    high_low: CategoryPrediction
    dozen: CategoryPrediction
    column: CategoryPrediction
    top_numbers: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class PredictorStats:
    predictor: PredictorType
    total_predictions: int = 0
    number_correct: int = 0
    color_correct: int = 0
    parity_correct: int = 0
    high_low_correct: int = 0
    dozen1_predictions: int = 0
    dozen1_correct: int = 0
    dozen2_predictions: int = 0
    dozen2_correct: int = 0
    dozen3_predictions: int = 0
    dozen3_correct: int = 0
    column1_predictions: int = 0
    column1_correct: int = 0
    column2_predictions: int = 0
    column2_correct: int = 0
    column3_predictions: int = 0
    column3_correct: int = 0
    
    def get_accuracy(self, category: str) -> float:
        if category.startswith("dozen") or category.startswith("column"):
            predictions = getattr(self, f"{category}_predictions", 0)
            if predictions == 0:
                return 0.0
            correct = getattr(self, f"{category}_correct", 0)
            return (correct / predictions) * 100
        if self.total_predictions == 0:
            return 0.0
        correct = getattr(self, f"{category}_correct", 0)
        return (correct / self.total_predictions) * 100
    
    def get_dozen_column_count(self, category: str) -> int:
        return getattr(self, f"{category}_predictions", 0)


class PredictionEngine:
    RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    
    def __init__(self, model_path: Optional[str] = None, *, device: str = 'auto', seed: int = 42):
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.lstm_predictor = None
        self.policy_agent = None
        self.policy_metadata = {}
        self.extra_trees_predictor = None
        self.bias_aware_predictor = BiasAwarePredictor()
        self.history = []
        self.stats = {pt: PredictorStats(predictor=pt) for pt in PredictorType}
        self.statuses = {pt: ModelStatus(Availability.UNTRAINED) for pt in PredictorType}
        self.is_lstm_trained = False
        self.is_extra_trees_trained = False
        self.training_lock = threading.RLock()
        self.training_in_progress = False
        self.cancel_event = threading.Event()
        self.data_version = 0
        self.session_id = None
        self.last_training_result = None
        self._pending_predictions = None
        self._validated_version = None

    def add_number(self, number: int):
        with self.training_lock:
            self.history.append(validate_number(number))
            self.data_version += 1
            self._pending_predictions = None

    def load_history(self, numbers: List[int], session_id=None):
        validated = [validate_number(n) for n in numbers]
        with self.training_lock:
            self.cancel_event.set()
            self.history = validated
            self.session_id = session_id
            self.data_version += 1
            self._pending_predictions = None
            self._validated_version = None
            self.stats = {pt: PredictorStats(predictor=pt) for pt in PredictorType}
            self.lstm_predictor = None
            self.extra_trees_predictor = None
            self.is_lstm_trained = False
            self.is_extra_trees_trained = False
            self.statuses = {pt: ModelStatus(Availability.UNTRAINED) for pt in PredictorType}

    def get_number_color(self, number: int) -> str:
        if number == 0:
            return "green"
        return "red" if number in self.RED_NUMBERS else "black"
    
    def get_number_parity(self, number: int) -> str:
        if number == 0:
            return "zero"
        return "odd" if number % 2 == 1 else "even"
    
    def get_number_high_low(self, number: int) -> str:
        if number == 0:
            return "zero"
        return "high" if number >= 19 else "low"
    
    def get_number_dozen(self, number: int) -> int:
        if number == 0:
            return 0
        return (number - 1) // 12 + 1
    
    def get_number_column(self, number: int) -> int:
        if number == 0:
            return 0
        return ((number - 1) % 3) + 1
    
    def _compute_category_probabilities(
        self, 
        number_probs: Dict[int, float]
    ) -> Dict[str, Dict[Any, float]]:
        color_probs = {"red": 0.0, "black": 0.0, "green": 0.0}
        parity_probs = {"odd": 0.0, "even": 0.0, "zero": 0.0}
        high_low_probs = {"high": 0.0, "low": 0.0, "zero": 0.0}
        dozen_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        column_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        for num, prob in number_probs.items():
            color_probs[self.get_number_color(num)] += prob
            parity_probs[self.get_number_parity(num)] += prob
            high_low_probs[self.get_number_high_low(num)] += prob
            dozen_probs[self.get_number_dozen(num)] += prob
            column_probs[self.get_number_column(num)] += prob
        
        return {
            "color": color_probs,
            "parity": parity_probs,
            "high_low": high_low_probs,
            "dozen": dozen_probs,
            "column": column_probs
        }
    
    def prediction_from_probabilities(self, predictor, probabilities):
        vector = validate_probabilities(probabilities)
        number_probs = {n: float(vector[n]) for n in range(37)}
        categories = self._compute_category_probabilities(number_probs)
        def category(values):
            best = max(values, key=values.get)
            return CategoryPrediction(best, values[best], values)
        return FullPrediction(predictor=predictor, number=category(number_probs),
                              **{name: category(values) for name, values in categories.items()},
                              top_numbers=ranked_numbers(vector))

    def _forecast(self, predictor, model, minimum, trained=True):
        if len(self.history) < minimum:
            self.statuses[predictor] = ModelStatus(Availability.INSUFFICIENT_DATA, f'Need {minimum} observations')
            return None
        if not trained or model is None:
            if self.statuses[predictor].state not in (Availability.FAILED, Availability.UNAVAILABLE):
                self.statuses[predictor] = ModelStatus(Availability.UNTRAINED, 'Train this model first')
            return None
        try:
            prediction = self.prediction_from_probabilities(predictor, model.predict_proba(self.history))
            self.statuses[predictor] = ModelStatus(Availability.READY)
            return prediction
        except ImportError as error:
            self.statuses[predictor] = ModelStatus(Availability.UNAVAILABLE, str(error))
        except Exception as error:
            self.statuses[predictor] = ModelStatus(Availability.FAILED, str(error))
        return None

    def _predict_lstm(self):
        return self._forecast(PredictorType.LSTM, self.lstm_predictor, 10, self.is_lstm_trained)

    def _predict_extra_trees(self):
        return self._forecast(PredictorType.EXTRA_TREES, self.extra_trees_predictor, 10, self.is_extra_trees_trained)

    def _predict_bias(self):
        return self._forecast(PredictorType.BIAS, self.bias_aware_predictor, self.bias_aware_predictor.config.min_spins)

    def predict_all(self):
        with self.training_lock:
            if self._pending_predictions is None:
                self._pending_predictions = {
                    PredictorType.LSTM: self._predict_lstm(),
                    PredictorType.EXTRA_TREES: self._predict_extra_trees(),
                    PredictorType.BIAS: self._predict_bias(),
                }
            return dict(self._pending_predictions)

    def get_consensus_prediction(self, predictions=None):
        predictions = self.predict_all() if predictions is None else predictions
        valid = [prediction for kind, prediction in predictions.items()
                 if prediction is not None and kind not in (PredictorType.DQN, PredictorType.CONSENSUS, PredictorType.FAIR)]
        if not valid:
            return self.prediction_from_probabilities(PredictorType.FAIR, np.full(37, FAIR_PROBABILITY))
        vectors = [validate_probabilities(prediction.number.all_probabilities) for prediction in valid]
        return self.prediction_from_probabilities(PredictorType.CONSENSUS, np.mean(vectors, axis=0))

    def get_policy_decision(self, bankroll=1000.0, initial_bankroll=1000.0, stake=1.0):
        if not self.model_path:
            return PolicyDecision('dqn', None, None, ModelStatus(Availability.UNAVAILABLE, 'No checkpoint selected'))
        if len(self.history) < 20:
            return PolicyDecision('dqn', None, None, ModelStatus(Availability.INSUFFICIENT_DATA, 'Need 20 observations'))
        try:
            if self.policy_agent is None:
                if self.training_in_progress:
                    return PolicyDecision('dqn', None, None, ModelStatus(
                        Availability.UNAVAILABLE, 'Wait for training to finish before loading the policy'))
                from src.agents.dqn_agent import DQNAgent
                from src.checkpoints import capture_rng, restore_rng
                rng = capture_rng()
                try:
                    candidate = DQNAgent(device=self.device)
                    self.policy_metadata = candidate.load(self.model_path)
                    if candidate.action_size != 47:
                        raise ValueError('Policy must use the 47-action roulette contract')
                    self.policy_agent = candidate
                finally:
                    restore_rng(rng)
            mask = np.ones(self.policy_agent.action_size, dtype=bool)
            if bankroll < stake:
                mask[:46] = False
            action = self.policy_agent.act(np.asarray(self.history[-self.policy_agent.history_size:]),
                                           bankroll / initial_bankroll, training=False, action_mask=mask)
            return PolicyDecision('dqn', action, 0.0 if action == 46 else stake)
        except ImportError as error:
            return PolicyDecision('dqn', None, None, ModelStatus(Availability.UNAVAILABLE, str(error)))
        except Exception as error:
            return PolicyDecision('dqn', None, None, ModelStatus(Availability.FAILED, str(error)))

    def validate_prediction(self, actual_number: int, predictions=None):
        validate_number(actual_number)
        if self._validated_version == self.data_version:
            raise RuntimeError('Predictions for this observation were already validated')
        predictions = self._pending_predictions if predictions is None else predictions
        if predictions is None:
            raise RuntimeError('No prediction was emitted before this outcome')
        self._validated_version = self.data_version
        
        actual_color = self.get_number_color(actual_number)
        actual_parity = self.get_number_parity(actual_number)
        actual_high_low = self.get_number_high_low(actual_number)
        actual_dozen = self.get_number_dozen(actual_number)
        actual_column = self.get_number_column(actual_number)
        
        for pred_type, pred in predictions.items():
            if pred is None:
                continue
            
            stats = self.stats[pred_type]
            stats.total_predictions += 1
            
            if pred.number.value == actual_number:
                stats.number_correct += 1
            if pred.color.value == actual_color:
                stats.color_correct += 1
            if pred.parity.value == actual_parity:
                stats.parity_correct += 1
            if pred.high_low.value == actual_high_low:
                stats.high_low_correct += 1
            
            pred_dozen = pred.dozen.value
            if pred_dozen == 1:
                stats.dozen1_predictions += 1
                if actual_dozen == 1:
                    stats.dozen1_correct += 1
            elif pred_dozen == 2:
                stats.dozen2_predictions += 1
                if actual_dozen == 2:
                    stats.dozen2_correct += 1
            elif pred_dozen == 3:
                stats.dozen3_predictions += 1
                if actual_dozen == 3:
                    stats.dozen3_correct += 1
            
            pred_column = pred.column.value
            if pred_column == 1:
                stats.column1_predictions += 1
                if actual_column == 1:
                    stats.column1_correct += 1
            elif pred_column == 2:
                stats.column2_predictions += 1
                if actual_column == 2:
                    stats.column2_correct += 1
            elif pred_column == 3:
                stats.column3_predictions += 1
                if actual_column == 3:
                    stats.column3_correct += 1
    
    def get_stats_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        categories = ["number", "color", "parity", "high_low", "dozen1", "dozen2", "dozen3", "column1", "column2", "column3"]
        
        for pred_type, stats in self.stats.items():
            summary[pred_type.value] = {
                "total": stats.total_predictions,
                **{cat: stats.get_accuracy(cat) for cat in categories}
            }
        
        return summary
    
    def get_best_predictor(self, category: str) -> Optional[PredictorType]:
        best_type = None
        best_accuracy = -1.0
        
        for pred_type, stats in self.stats.items():
            if stats.total_predictions > 0:
                acc = stats.get_accuracy(category)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_type = pred_type
        
        return best_type
    
    def backtest_history(self, min_history: int = 20, train_ratio: float = 0.7, config=None):
        from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward
        return evaluate_walk_forward(list(self.history), config or EvaluationConfig())

    def _fit_snapshot(self, history, version, epochs, cancel_event, models=('lstm', 'extra_trees')):
        candidates = {}
        results = {}
        for kind in (PredictorType.LSTM, PredictorType.EXTRA_TREES):
            if kind.value not in models:
                results[kind.value] = {'status': 'unavailable', 'message': 'Not requested'}
                continue
            if cancel_event.is_set():
                results[kind.value] = {'status': 'cancelled', 'message': 'Training cancelled'}
                continue
            try:
                if kind == PredictorType.LSTM:
                    from src.utils.lstm_predictor import LSTMPredictor
                    from src.checkpoints import seed_everything
                    seed_everything(self.seed, self.device)
                    model = LSTMPredictor(device=self.device)
                    result = model.fit(history, epochs=epochs, cancel_event=cancel_event)
                    fitted = model.is_trained
                else:
                    model = ExtraTreesPredictor(seed=self.seed)
                    result = model.fit(history)
                    fitted = model.is_fitted
                results[kind.value] = result or {'status': 'failed', 'message': 'Training produced no result'}
                if fitted:
                    candidates[kind] = model
            except ImportError as error:
                results[kind.value] = {'status': 'unavailable', 'message': str(error)}
            except Exception as error:
                results[kind.value] = {'status': 'failed', 'message': str(error)}
        with self.training_lock:
            if version != self.data_version or cancel_event.is_set():
                return {'status': 'cancelled', 'message': 'Session or history changed during training', 'models': results}
            self.lstm_predictor = candidates.get(PredictorType.LSTM)
            self.extra_trees_predictor = candidates.get(PredictorType.EXTRA_TREES)
            self.is_lstm_trained = self.lstm_predictor is not None
            self.is_extra_trees_trained = self.extra_trees_predictor is not None
            for kind in (PredictorType.LSTM, PredictorType.EXTRA_TREES):
                result = results[kind.value]
                state = Availability.READY if kind in candidates else {
                    'unavailable': Availability.UNAVAILABLE, 'insufficient_data': Availability.INSUFFICIENT_DATA,
                    'error': Availability.INSUFFICIENT_DATA}.get(result['status'], Availability.FAILED)
                self.statuses[kind] = ModelStatus(state, result.get('message', ''))
            self._pending_predictions = None
        return results

    def _begin_training(self):
        with self.training_lock:
            if self.training_in_progress:
                raise RuntimeError('Training is already running')
            self.training_in_progress = True
            self.cancel_event = threading.Event()
            return list(self.history), self.data_version, self.cancel_event

    def train_async(self, epochs: int = 30, on_complete: Optional[Callable] = None):
        history, version, cancel_event = self._begin_training()
        def worker():
            try:
                self.last_training_result = self._fit_snapshot(history, version, epochs, cancel_event)
            except Exception as error:
                self.last_training_result = {'status': 'failed', 'message': str(error)}
            finally:
                with self.training_lock:
                    self.training_in_progress = False
                if on_complete:
                    on_complete()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def train_sync(self, epochs: int = 30, models=('lstm', 'extra_trees'), cancel_event=None):
        history, version, local_cancel = self._begin_training()
        cancel_event = cancel_event if cancel_event is not None else local_cancel
        try:
            self.last_training_result = self._fit_snapshot(history, version, epochs, cancel_event, models=models)
            return self.last_training_result
        finally:
            with self.training_lock:
                self.training_in_progress = False

    def cancel_training(self):
        self.cancel_event.set()

    def get_status(self):
        return {
            'history_size': len(self.history), 'lstm_trained': self.is_lstm_trained,
            'extra_trees_trained': self.is_extra_trees_trained,
            'dqn_loaded': self.policy_agent is not None,
            'training_in_progress': self.training_in_progress, 'device': self.device,
            'executed_devices': {'lstm': str(self.lstm_predictor.device) if self.lstm_predictor else None,
                                 'dqn': str(self.policy_agent.device) if self.policy_agent else None,
                                 'extra_trees': 'cpu' if self.is_extra_trees_trained else None},
            'models': {kind.value: {'state': status.state.value, 'reason': status.reason}
                       for kind, status in self.statuses.items()},
        }
