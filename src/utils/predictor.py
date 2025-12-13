import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

from src.database.models import Spin, Prediction


PROBABILITY_THRESHOLD = 0.03  # 3% filter from Salirrosas (2016)
FAIR_PROBABILITY = 1 / 37  # ~2.7%


def get_optimal_device(force_gpu: bool = True) -> torch.device:
    """
    Get optimal compute device.
    
    Args:
        force_gpu: If True, raises error when CUDA unavailable
    
    Returns:
        torch.device configured for optimal performance
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return device
    elif force_gpu:
        raise RuntimeError(
            "GPU required but CUDA not available. "
            "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    return torch.device("cpu")


class LSTMNetwork(nn.Module):
    """
    LSTM architecture from NeuralRoulette-AI (devddine).
    
    Architecture: LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(64) → Dense(37)
    """
    
    def __init__(
        self, 
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 37,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=64,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out[:, -1, :])
        
        out = self.relu(self.fc1(lstm2_out))
        out = self.fc2(out)
        
        return out


class LSTMPredictor:
    """
    LSTM-based predictor inspired by NeuralRoulette-AI.
    
    Uses sequence learning to predict next roulette number.
    Implements Top-N prediction strategy for different risk levels.
    
    GPU+CPU hybrid mode:
    - CPU: Data loading and preprocessing (parallel workers)
    - GPU: Model training and inference
    - pin_memory + non_blocking transfers for async data movement
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        hidden_size: int = 128,
        force_gpu: bool = True,
        num_workers: int = 2
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = get_optimal_device(force_gpu=force_gpu)
        self.num_workers = num_workers if self.device.type == "cuda" else 0
        self.pin_memory = self.device.type == "cuda"
        
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=hidden_size,
            output_size=37
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.is_trained = False
        self.training_history: List[float] = []
        
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def _prepare_sequences(
        self, 
        history: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training sequences from history (CPU preprocessing)."""
        X, y = [], []
        
        for i in range(len(history) - self.sequence_length):
            seq = history[i:i + self.sequence_length]
            target = history[i + self.sequence_length]
            
            normalized_seq = [n / 36.0 for n in seq]
            X.append(normalized_seq)
            y.append(target)
        
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.LongTensor(y)
        
        return X_tensor, y_tensor
    
    def _create_dataloader(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create optimized DataLoader for GPU+CPU hybrid processing."""
        dataset = TensorDataset(X, y)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def fit(
        self, 
        history: List[int], 
        epochs: int = 50, 
        batch_size: int = 32,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train the LSTM model on historical data.
        
        Uses GPU+CPU hybrid:
        - CPU workers preprocess and load batches
        - GPU computes forward/backward passes
        - Async transfers overlap data movement with computation
        """
        if len(history) < self.sequence_length + 10:
            return {"status": "error", "message": "Not enough data"}
        
        X, y = self._prepare_sequences(history)
        
        if len(X) < batch_size:
            batch_size = len(X)
        
        dataloader = self._create_dataloader(X, y, batch_size)
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / batches
            epoch_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        self.training_history.extend(epoch_losses)
        
        return {
            "status": "success",
            "final_loss": epoch_losses[-1],
            "epochs_trained": epochs,
            "samples": len(X),
            "device": str(self.device)
        }
    
    def predict(self, history: List[int]) -> Tuple[int, float]:
        """Predict the most likely next number."""
        if len(history) < self.sequence_length:
            return (0, FAIR_PROBABILITY)
        
        self.model.eval()
        
        seq = history[-self.sequence_length:]
        normalized = [n / 36.0 for n in seq]
        
        X = torch.FloatTensor([normalized]).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            predicted = probabilities.argmax().item()
            confidence = probabilities[predicted].item()
        
        return (int(predicted), float(confidence))
    
    def predict_top_n(
        self, 
        history: List[int], 
        n: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Predict top N most likely numbers (Top-N strategy from NeuralRoulette-AI).
        
        Top-1: High risk (break-even: 2.70%)
        Top-3: Medium risk (break-even: 8.57%)
        Top-18: Low risk (break-even: 51.43%)
        """
        if len(history) < self.sequence_length:
            return [(i, FAIR_PROBABILITY) for i in range(min(n, 37))]
        
        self.model.eval()
        
        seq = history[-self.sequence_length:]
        normalized = [n / 36.0 for n in seq]
        
        X = torch.FloatTensor([normalized]).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            top_probs, top_indices = torch.topk(probabilities, min(n, 37))
            
            results = [
                (int(idx.item()), float(prob.item()))
                for idx, prob in zip(top_indices, top_probs)
            ]
        
        return results
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "is_trained": self.is_trained,
            "training_history": self.training_history
        }, path)
    
    def load(self, path: str, force_gpu: bool = True):
        """Load model from file (forces GPU by default)."""
        self.device = get_optimal_device(force_gpu=force_gpu)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.sequence_length = checkpoint.get("sequence_length", 10)
        self.hidden_size = checkpoint.get("hidden_size", 128)
        
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            output_size=37
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.is_trained = checkpoint.get("is_trained", True)
        self.training_history = checkpoint.get("training_history", [])
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device configuration info."""
        info = {
            "device": str(self.device),
            "device_type": self.device.type,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers
        }
        
        if self.device.type == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB",
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "tf32_enabled": torch.backends.cuda.matmul.allow_tf32
            })
        
        return info
    
    def get_break_even_info(self, n: int) -> Dict[str, float]:
        """Get break-even win rate for Top-N strategy."""
        payout = 35
        cost_per_spin = n
        break_even = cost_per_spin / (payout + 1) * 100
        
        return {
            "numbers_covered": n,
            "cost_per_spin": cost_per_spin,
            "payout_on_win": payout,
            "break_even_rate": break_even,
            "fair_hit_rate": (n / 37) * 100
        }


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
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
    
    def fit(self, history: List[int], window_size: int = 10):
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            
            if len(history) < window_size + 1:
                print(f"[ExtraTrees] Not enough history: {len(history)} < {window_size + 1}")
                return
            
            X = []
            y = []
            
            for i in range(window_size, len(history)):
                features = self._extract_features(history[i-window_size:i])
                X.append(features)
                y.append(history[i])
            
            unique_classes = len(set(y))
            if unique_classes < 2:
                print(f"[ExtraTrees] Not enough unique classes: {unique_classes}")
                return
            
            self.model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
            self.is_fitted = True
            print(f"[ExtraTrees] Fitted successfully with {len(X)} samples, {unique_classes} classes")
        except ImportError as e:
            print(f"[ExtraTrees] sklearn not installed: {e}")
            self.is_fitted = False
        except Exception as e:
            print(f"[ExtraTrees] Error during fit: {e}")
            self.is_fitted = False
    
    def predict(self, history: List[int], window_size: int = 10) -> Tuple[int, float]:
        if not self.is_fitted or self.model is None:
            return (0, FAIR_PROBABILITY)
        
        if len(history) < window_size:
            return (0, FAIR_PROBABILITY)
        
        features = self._extract_features(history[-window_size:])
        
        probas = self.model.predict_proba([features])[0]
        classes = self.model.classes_
        
        best_idx = np.argmax(probas)
        predicted = classes[best_idx]
        confidence = probas[best_idx]
        
        return (int(predicted), float(confidence))
    
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history_size = 20
        
        self.use_extra_trees = use_extra_trees
        self.extra_trees = ExtraTreesPredictor() if use_extra_trees else None
        
        self.consecutive_losses = 0
        self.loss_history: List[bool] = []
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        try:
            from src.agents.dqn_agent import DQNAgent
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            config = checkpoint.get('config', {})
            self.history_size = config.get('history_size', 20)
            
            self.model = DQNAgent(
                history_size=self.history_size,
                action_size=config.get('action_size', 47),
                embedding_dim=config.get('embedding_dim', 64),
                hidden_size=config.get('hidden_size', 128),
                device=str(self.device)
            )
            
            state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'q_network'
            self.model.q_network.load_state_dict(checkpoint[state_dict_key])
            self.model.q_network.eval()
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
    
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
    
    def predict_from_history(self, history: List[int]) -> Dict:
        if len(history) < 5:
            return self._random_prediction()
        
        padded_history = self._pad_history(history)
        
        predictions = {
            "number": self._predict_number(padded_history),
            "color": self._predict_color(history),
            "parity": self._predict_parity(history),
            "high_low": self._predict_high_low(history),
            "dozen": self._predict_dozen(history),
            "column": self._predict_column(history)
        }
        
        return predictions
    
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
    
    def _pad_history(self, history: List[int]) -> np.ndarray:
        if len(history) >= self.history_size:
            return np.array(history[-self.history_size:])
        padding = [0] * (self.history_size - len(history))
        return np.array(padding + history)
    
    def _predict_number(self, history: np.ndarray) -> Tuple[int, float]:
        if self.use_extra_trees and self.extra_trees is not None and self.extra_trees.is_fitted:
            try:
                predicted, confidence = self.extra_trees.predict(list(history))
                return (predicted, confidence)
            except Exception:
                pass
        
        if self.model is not None:
            try:
                with torch.no_grad():
                    history_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
                    gain_tensor = torch.FloatTensor([[1.0]]).to(self.device)
                    q_values = self.model.q_network(history_tensor, gain_tensor)
                    
                    straight_bets = q_values[0, :37]
                    probabilities = torch.softmax(straight_bets, dim=0)
                    predicted = probabilities.argmax().item()
                    confidence = probabilities[predicted].item()
                    
                    return (predicted, confidence)
            except Exception:
                pass
        
        recent = history[-20:] if len(history) >= 20 else history
        counts = Counter(recent)
        
        cold_numbers = []
        for n in range(37):
            if counts.get(n, 0) == 0:
                cold_numbers.append(n)
        
        if cold_numbers:
            predicted = np.random.choice(cold_numbers)
        else:
            least_common = counts.most_common()[-1][0] if counts else 0
            predicted = least_common
        
        return (predicted, 1/37)
    
    def _predict_color(self, history: List[int]) -> Tuple[str, float]:
        recent = history[-10:] if len(history) >= 10 else history
        
        red_count = sum(1 for n in recent if Spin(number=n).color == "red")
        black_count = sum(1 for n in recent if Spin(number=n).color == "black")
        
        total = red_count + black_count
        if total == 0:
            return ("red", 0.5)
        
        red_ratio = red_count / total
        
        if red_ratio > 0.6:
            return ("black", 0.5 + (red_ratio - 0.5) * 0.2)
        elif red_ratio < 0.4:
            return ("red", 0.5 + (0.5 - red_ratio) * 0.2)
        else:
            return ("red" if np.random.random() > 0.5 else "black", 0.5)
    
    def _predict_parity(self, history: List[int]) -> Tuple[str, float]:
        recent = [n for n in history[-10:] if n != 0]
        if not recent:
            return ("odd", 0.5)
        
        odd_count = sum(1 for n in recent if n % 2 == 1)
        odd_ratio = odd_count / len(recent)
        
        if odd_ratio > 0.6:
            return ("even", 0.5 + (odd_ratio - 0.5) * 0.2)
        elif odd_ratio < 0.4:
            return ("odd", 0.5 + (0.5 - odd_ratio) * 0.2)
        else:
            return ("odd" if np.random.random() > 0.5 else "even", 0.5)
    
    def _predict_high_low(self, history: List[int]) -> Tuple[str, float]:
        recent = [n for n in history[-10:] if n != 0]
        if not recent:
            return ("low", 0.5)
        
        high_count = sum(1 for n in recent if n >= 19)
        high_ratio = high_count / len(recent)
        
        if high_ratio > 0.6:
            return ("low", 0.5 + (high_ratio - 0.5) * 0.2)
        elif high_ratio < 0.4:
            return ("high", 0.5 + (0.5 - high_ratio) * 0.2)
        else:
            return ("high" if np.random.random() > 0.5 else "low", 0.5)
    
    def _predict_dozen(self, history: List[int]) -> Tuple[int, float]:
        recent = [n for n in history[-15:] if n != 0]
        if not recent:
            return (1, 0.33)
        
        dozen_counts = {1: 0, 2: 0, 3: 0}
        for n in recent:
            d = (n - 1) // 12 + 1
            dozen_counts[d] += 1
        
        least_common = min(dozen_counts, key=dozen_counts.get)
        max_count = max(dozen_counts.values())
        min_count = dozen_counts[least_common]
        
        if max_count > min_count:
            confidence = 0.33 + (max_count - min_count) / len(recent) * 0.1
        else:
            confidence = 0.33
        
        return (least_common, confidence)
    
    def _predict_column(self, history: List[int]) -> Tuple[int, float]:
        recent = [n for n in history[-15:] if n != 0]
        if not recent:
            return (1, 0.33)
        
        col_counts = {1: 0, 2: 0, 3: 0}
        for n in recent:
            c = ((n - 1) % 3) + 1
            col_counts[c] += 1
        
        least_common = min(col_counts, key=col_counts.get)
        max_count = max(col_counts.values())
        min_count = col_counts[least_common]
        
        if max_count > min_count:
            confidence = 0.33 + (max_count - min_count) / len(recent) * 0.1
        else:
            confidence = 0.33
        
        return (least_common, confidence)
    
    def _random_prediction(self) -> Dict:
        return {
            "number": (np.random.randint(0, 37), 1/37),
            "color": ("red" if np.random.random() > 0.5 else "black", 0.5),
            "parity": ("odd" if np.random.random() > 0.5 else "even", 0.5),
            "high_low": ("high" if np.random.random() > 0.5 else "low", 0.5),
            "dozen": (np.random.randint(1, 4), 0.33),
            "column": (np.random.randint(1, 4), 0.33)
        }
    
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
