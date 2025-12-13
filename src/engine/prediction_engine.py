import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import Counter
import numpy as np

from src.utils.predictor import LSTMPredictor, RoulettePredictor, ExtraTreesPredictor, FAIR_PROBABILITY
from src.database.models import Spin


class PredictorType(Enum):
    LSTM = "lstm"
    DQN = "dqn"
    EXTRA_TREES = "extra_trees"
    BIAS = "bias"


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
    
    def __init__(self, model_path: Optional[str] = None):
        self.lstm_predictor = LSTMPredictor(sequence_length=10, force_gpu=True)
        self.dqn_predictor = RoulettePredictor(model_path=model_path, use_extra_trees=False)
        self.extra_trees_predictor = ExtraTreesPredictor()
        
        self.history: List[int] = []
        self.stats: Dict[PredictorType, PredictorStats] = {
            pt: PredictorStats(predictor=pt) for pt in PredictorType
        }
        
        self.is_lstm_trained = False
        self.is_extra_trees_trained = False
        self.training_lock = threading.Lock()
        self.training_in_progress = False
        self.on_training_complete: Optional[Callable] = None
    
    def add_number(self, number: int):
        if 0 <= number <= 36:
            self.history.append(number)
    
    def load_history(self, numbers: List[int]):
        self.history = [n for n in numbers if 0 <= n <= 36]
    
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
    
    def _predict_lstm(self) -> Optional[FullPrediction]:
        if not self.is_lstm_trained or len(self.history) < 10:
            return None
        
        top_numbers = self.lstm_predictor.predict_top_n(self.history, n=10)
        
        number_probs = {i: FAIR_PROBABILITY for i in range(37)}
        for num, prob in top_numbers:
            number_probs[num] = prob
        
        total = sum(number_probs.values())
        number_probs = {k: v / total for k, v in number_probs.items()}
        
        cat_probs = self._compute_category_probabilities(number_probs)
        
        best_number = top_numbers[0] if top_numbers else (0, FAIR_PROBABILITY)
        best_color = max(cat_probs["color"].items(), key=lambda x: x[1])
        best_parity = max([(k, v) for k, v in cat_probs["parity"].items() if k != "zero"], key=lambda x: x[1])
        best_high_low = max([(k, v) for k, v in cat_probs["high_low"].items() if k != "zero"], key=lambda x: x[1])
        best_dozen = max([(k, v) for k, v in cat_probs["dozen"].items() if k != 0], key=lambda x: x[1])
        best_column = max([(k, v) for k, v in cat_probs["column"].items() if k != 0], key=lambda x: x[1])
        
        return FullPrediction(
            predictor=PredictorType.LSTM,
            number=CategoryPrediction(best_number[0], best_number[1], number_probs),
            color=CategoryPrediction(best_color[0], best_color[1], cat_probs["color"]),
            parity=CategoryPrediction(best_parity[0], best_parity[1], cat_probs["parity"]),
            high_low=CategoryPrediction(best_high_low[0], best_high_low[1], cat_probs["high_low"]),
            dozen=CategoryPrediction(best_dozen[0], best_dozen[1], cat_probs["dozen"]),
            column=CategoryPrediction(best_column[0], best_column[1], cat_probs["column"]),
            top_numbers=top_numbers
        )
    
    def _predict_bias(self) -> Optional[FullPrediction]:
        if len(self.history) < 50:
            return None
        
        counter = Counter(self.history)
        total = len(self.history)
        
        number_probs = {i: counter.get(i, 0) / total for i in range(37)}
        
        cat_probs = self._compute_category_probabilities(number_probs)
        
        top_numbers = sorted(number_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        best_number = top_numbers[0] if top_numbers else (0, FAIR_PROBABILITY)
        best_color = max(cat_probs["color"].items(), key=lambda x: x[1])
        best_parity = max([(k, v) for k, v in cat_probs["parity"].items() if k != "zero"], key=lambda x: x[1])
        best_high_low = max([(k, v) for k, v in cat_probs["high_low"].items() if k != "zero"], key=lambda x: x[1])
        best_dozen = max([(k, v) for k, v in cat_probs["dozen"].items() if k != 0], key=lambda x: x[1])
        best_column = max([(k, v) for k, v in cat_probs["column"].items() if k != 0], key=lambda x: x[1])
        
        return FullPrediction(
            predictor=PredictorType.BIAS,
            number=CategoryPrediction(best_number[0], best_number[1], number_probs),
            color=CategoryPrediction(best_color[0], best_color[1], cat_probs["color"]),
            parity=CategoryPrediction(best_parity[0], best_parity[1], cat_probs["parity"]),
            high_low=CategoryPrediction(best_high_low[0], best_high_low[1], cat_probs["high_low"]),
            dozen=CategoryPrediction(best_dozen[0], best_dozen[1], cat_probs["dozen"]),
            column=CategoryPrediction(best_column[0], best_column[1], cat_probs["column"]),
            top_numbers=top_numbers
        )
    
    def _predict_extra_trees(self) -> Optional[FullPrediction]:
        if not self.is_extra_trees_trained or len(self.history) < 20:
            return None
        
        predicted, confidence = self.extra_trees_predictor.predict(self.history)
        
        number_probs = {i: FAIR_PROBABILITY for i in range(37)}
        number_probs[predicted] = confidence
        
        total = sum(number_probs.values())
        number_probs = {k: v / total for k, v in number_probs.items()}
        
        cat_probs = self._compute_category_probabilities(number_probs)
        
        top_numbers = sorted(number_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        best_color = max(cat_probs["color"].items(), key=lambda x: x[1])
        best_parity = max([(k, v) for k, v in cat_probs["parity"].items() if k != "zero"], key=lambda x: x[1])
        best_high_low = max([(k, v) for k, v in cat_probs["high_low"].items() if k != "zero"], key=lambda x: x[1])
        best_dozen = max([(k, v) for k, v in cat_probs["dozen"].items() if k != 0], key=lambda x: x[1])
        best_column = max([(k, v) for k, v in cat_probs["column"].items() if k != 0], key=lambda x: x[1])
        
        return FullPrediction(
            predictor=PredictorType.EXTRA_TREES,
            number=CategoryPrediction(predicted, confidence, number_probs),
            color=CategoryPrediction(best_color[0], best_color[1], cat_probs["color"]),
            parity=CategoryPrediction(best_parity[0], best_parity[1], cat_probs["parity"]),
            high_low=CategoryPrediction(best_high_low[0], best_high_low[1], cat_probs["high_low"]),
            dozen=CategoryPrediction(best_dozen[0], best_dozen[1], cat_probs["dozen"]),
            column=CategoryPrediction(best_column[0], best_column[1], cat_probs["column"]),
            top_numbers=top_numbers
        )
    
    def _predict_dqn(self) -> Optional[FullPrediction]:
        if self.dqn_predictor.model is None or len(self.history) < 20:
            return None
        
        predictions = self.dqn_predictor.predict_from_history(self.history)
        
        num_pred, num_conf = predictions["number"]
        
        number_probs = {i: FAIR_PROBABILITY for i in range(37)}
        number_probs[num_pred] = num_conf
        
        total = sum(number_probs.values())
        number_probs = {k: v / total for k, v in number_probs.items()}
        
        cat_probs = self._compute_category_probabilities(number_probs)
        top_numbers = sorted(number_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return FullPrediction(
            predictor=PredictorType.DQN,
            number=CategoryPrediction(num_pred, num_conf, number_probs),
            color=CategoryPrediction(predictions["color"][0], predictions["color"][1], cat_probs["color"]),
            parity=CategoryPrediction(predictions["parity"][0], predictions["parity"][1], cat_probs["parity"]),
            high_low=CategoryPrediction(predictions["high_low"][0], predictions["high_low"][1], cat_probs["high_low"]),
            dozen=CategoryPrediction(predictions["dozen"][0], predictions["dozen"][1], cat_probs["dozen"]),
            column=CategoryPrediction(predictions["column"][0], predictions["column"][1], cat_probs["column"]),
            top_numbers=top_numbers
        )
    
    def predict_all(self) -> Dict[PredictorType, Optional[FullPrediction]]:
        return {
            PredictorType.LSTM: self._predict_lstm(),
            PredictorType.DQN: self._predict_dqn(),
            PredictorType.EXTRA_TREES: self._predict_extra_trees(),
            PredictorType.BIAS: self._predict_bias()
        }
    
    def get_consensus_prediction(self) -> Optional[FullPrediction]:
        predictions = self.predict_all()
        valid_predictions = [p for p in predictions.values() if p is not None]
        
        if not valid_predictions:
            return None
        
        number_votes: Dict[int, float] = {}
        color_votes: Dict[str, float] = {}
        parity_votes: Dict[str, float] = {}
        high_low_votes: Dict[str, float] = {}
        dozen_votes: Dict[int, float] = {}
        column_votes: Dict[int, float] = {}
        
        for pred in valid_predictions:
            for num, prob in pred.number.all_probabilities.items():
                number_votes[num] = number_votes.get(num, 0) + prob
            for color, prob in pred.color.all_probabilities.items():
                color_votes[color] = color_votes.get(color, 0) + prob
            for parity, prob in pred.parity.all_probabilities.items():
                parity_votes[parity] = parity_votes.get(parity, 0) + prob
            for hl, prob in pred.high_low.all_probabilities.items():
                high_low_votes[hl] = high_low_votes.get(hl, 0) + prob
            for dozen, prob in pred.dozen.all_probabilities.items():
                dozen_votes[dozen] = dozen_votes.get(dozen, 0) + prob
            for column, prob in pred.column.all_probabilities.items():
                column_votes[column] = column_votes.get(column, 0) + prob
        
        n = len(valid_predictions)
        number_probs = {k: v / n for k, v in number_votes.items()}
        color_probs = {k: v / n for k, v in color_votes.items()}
        parity_probs = {k: v / n for k, v in parity_votes.items()}
        high_low_probs = {k: v / n for k, v in high_low_votes.items()}
        dozen_probs = {k: v / n for k, v in dozen_votes.items()}
        column_probs = {k: v / n for k, v in column_votes.items()}
        
        top_numbers = sorted(number_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        best_number = top_numbers[0]
        best_color = max(color_probs.items(), key=lambda x: x[1])
        best_parity = max([(k, v) for k, v in parity_probs.items() if k != "zero"], key=lambda x: x[1])
        best_high_low = max([(k, v) for k, v in high_low_probs.items() if k != "zero"], key=lambda x: x[1])
        best_dozen = max([(k, v) for k, v in dozen_probs.items() if k != 0], key=lambda x: x[1])
        best_column = max([(k, v) for k, v in column_probs.items() if k != 0], key=lambda x: x[1])
        
        return FullPrediction(
            predictor=PredictorType.BIAS,
            number=CategoryPrediction(best_number[0], best_number[1], number_probs),
            color=CategoryPrediction(best_color[0], best_color[1], color_probs),
            parity=CategoryPrediction(best_parity[0], best_parity[1], parity_probs),
            high_low=CategoryPrediction(best_high_low[0], best_high_low[1], high_low_probs),
            dozen=CategoryPrediction(best_dozen[0], best_dozen[1], dozen_probs),
            column=CategoryPrediction(best_column[0], best_column[1], column_probs),
            top_numbers=top_numbers
        )
    
    def validate_prediction(self, actual_number: int):
        predictions = self.predict_all()
        
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
    
    def backtest_history(self, min_history: int = 20, train_ratio: float = 0.7) -> Dict[str, Any]:
        if len(self.history) < min_history + 10:
            return {"error": "Not enough history for backtesting"}
        
        train_size = int(len(self.history) * train_ratio)
        train_size = max(train_size, min_history)
        
        train_data = self.history[:train_size]
        test_data = self.history[train_size:]
        
        if len(test_data) < 5:
            return {"error": "Not enough test data after split"}
        
        temp_lstm_predictor = LSTMPredictor(sequence_length=10, force_gpu=True)
        temp_extra_trees = ExtraTreesPredictor()
        
        if len(train_data) >= 20:
            temp_lstm_predictor.fit(train_data, epochs=30)
        
        if len(train_data) >= 30:
            temp_extra_trees.fit(train_data)
        
        for pt in PredictorType:
            self.stats[pt] = PredictorStats(predictor=pt)
        
        results = {"tested": 0, "train_size": train_size, "test_size": len(test_data)}
        
        for i, actual_number in enumerate(test_data):
            context_history = train_data + test_data[:i]
            
            if len(context_history) < 10:
                continue
            
            actual_color = self.get_number_color(actual_number)
            actual_parity = self.get_number_parity(actual_number)
            actual_high_low = self.get_number_high_low(actual_number)
            actual_dozen = self.get_number_dozen(actual_number)
            actual_column = self.get_number_column(actual_number)
            
            if temp_lstm_predictor.model is not None:
                try:
                    top_numbers = temp_lstm_predictor.predict_top_n(context_history, n=10)
                    if top_numbers:
                        pred_num = top_numbers[0][0]
                        pred_dozen = self.get_number_dozen(pred_num)
                        pred_column = self.get_number_column(pred_num)
                        pred_color = self.get_number_color(pred_num)
                        pred_parity = self.get_number_parity(pred_num)
                        pred_high_low = self.get_number_high_low(pred_num)
                        
                        stats = self.stats[PredictorType.LSTM]
                        stats.total_predictions += 1
                        if pred_num == actual_number:
                            stats.number_correct += 1
                        if pred_color == actual_color:
                            stats.color_correct += 1
                        if pred_parity == actual_parity:
                            stats.parity_correct += 1
                        if pred_high_low == actual_high_low:
                            stats.high_low_correct += 1
                        
                        self._update_dozen_column_stats(stats, pred_dozen, actual_dozen, pred_column, actual_column)
                except Exception:
                    pass
            
            if temp_extra_trees.is_fitted:
                try:
                    pred_num, _ = temp_extra_trees.predict(context_history)
                    pred_dozen = self.get_number_dozen(pred_num)
                    pred_column = self.get_number_column(pred_num)
                    pred_color = self.get_number_color(pred_num)
                    pred_parity = self.get_number_parity(pred_num)
                    pred_high_low = self.get_number_high_low(pred_num)
                    
                    stats = self.stats[PredictorType.EXTRA_TREES]
                    stats.total_predictions += 1
                    if pred_num == actual_number:
                        stats.number_correct += 1
                    if pred_color == actual_color:
                        stats.color_correct += 1
                    if pred_parity == actual_parity:
                        stats.parity_correct += 1
                    if pred_high_low == actual_high_low:
                        stats.high_low_correct += 1
                    
                    self._update_dozen_column_stats(stats, pred_dozen, actual_dozen, pred_column, actual_column)
                except Exception:
                    pass
            
            if self.dqn_predictor.model is not None and len(context_history) >= 20:
                try:
                    predictions = self.dqn_predictor.predict_from_history(context_history)
                    pred_num, _ = predictions["number"]
                    pred_dozen = self.get_number_dozen(pred_num)
                    pred_column = self.get_number_column(pred_num)
                    pred_color = self.get_number_color(pred_num)
                    pred_parity = self.get_number_parity(pred_num)
                    pred_high_low = self.get_number_high_low(pred_num)
                    
                    stats = self.stats[PredictorType.DQN]
                    stats.total_predictions += 1
                    if pred_num == actual_number:
                        stats.number_correct += 1
                    if pred_color == actual_color:
                        stats.color_correct += 1
                    if pred_parity == actual_parity:
                        stats.parity_correct += 1
                    if pred_high_low == actual_high_low:
                        stats.high_low_correct += 1
                    
                    self._update_dozen_column_stats(stats, pred_dozen, actual_dozen, pred_column, actual_column)
                except Exception:
                    pass
            
            if len(context_history) >= 50:
                counter = Counter(context_history)
                pred_num = counter.most_common(1)[0][0]
                pred_dozen = self.get_number_dozen(pred_num)
                pred_column = self.get_number_column(pred_num)
                pred_color = self.get_number_color(pred_num)
                pred_parity = self.get_number_parity(pred_num)
                pred_high_low = self.get_number_high_low(pred_num)
                
                stats = self.stats[PredictorType.BIAS]
                stats.total_predictions += 1
                if pred_num == actual_number:
                    stats.number_correct += 1
                if pred_color == actual_color:
                    stats.color_correct += 1
                if pred_parity == actual_parity:
                    stats.parity_correct += 1
                if pred_high_low == actual_high_low:
                    stats.high_low_correct += 1
                
                self._update_dozen_column_stats(stats, pred_dozen, actual_dozen, pred_column, actual_column)
            
            results["tested"] += 1
        
        results["models_status"] = {
            "lstm_trained": temp_lstm_predictor.model is not None,
            "extra_trees_trained": temp_extra_trees.is_fitted,
            "dqn_loaded": self.dqn_predictor.model is not None,
            "history_length": len(self.history)
        }
        
        return results
    
    def _update_dozen_column_stats(self, stats: PredictorStats, pred_dozen: int, actual_dozen: int, pred_column: int, actual_column: int):
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
        
        return results
    
    def train_async(
        self, 
        epochs: int = 50,
        on_complete: Optional[Callable] = None
    ):
        if self.training_in_progress:
            return
        
        self.on_training_complete = on_complete
        
        def train_worker():
            with self.training_lock:
                self.training_in_progress = True
                
                if len(self.history) >= 20:
                    self.lstm_predictor.fit(self.history, epochs=epochs)
                    self.is_lstm_trained = True
                
                if len(self.history) >= 30:
                    self.extra_trees_predictor.fit(self.history)
                    self.is_extra_trees_trained = self.extra_trees_predictor.is_fitted
                
                self.training_in_progress = False
                
                if self.on_training_complete:
                    self.on_training_complete()
        
        thread = threading.Thread(target=train_worker, daemon=True)
        thread.start()
    
    def train_sync(self, epochs: int = 50) -> Dict[str, Any]:
        results = {"lstm": None, "extra_trees": None}
        
        if len(self.history) >= 20:
            results["lstm"] = self.lstm_predictor.fit(self.history, epochs=epochs)
            self.is_lstm_trained = True
        
        if len(self.history) >= 30:
            self.extra_trees_predictor.fit(self.history)
            self.is_extra_trees_trained = self.extra_trees_predictor.is_fitted
            results["extra_trees"] = {"fitted": self.is_extra_trees_trained}
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "history_size": len(self.history),
            "lstm_trained": self.is_lstm_trained,
            "extra_trees_trained": self.is_extra_trees_trained,
            "dqn_loaded": self.dqn_predictor.model is not None,
            "training_in_progress": self.training_in_progress,
            "device": str(self.lstm_predictor.device)
        }
