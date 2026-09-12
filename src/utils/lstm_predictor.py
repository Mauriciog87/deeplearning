import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any
from numbers import Integral

from src.probabilities import validate_probabilities, ranked_numbers
from src.settlement import validate_number
from src.checkpoints import atomic_save, load_checkpoint, capture_rng, restore_rng, SCHEMA_VERSION, OBSERVATION_VERSION


def get_optimal_device(force_gpu: bool = False, device: str = 'auto') -> torch.device:
    """
    Get optimal compute device.
    
    Args:
        force_gpu: If True, raises error when CUDA unavailable
    
    Returns:
        torch.device configured for optimal performance
    """
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu or cuda')
    if device == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
        return device
    elif force_gpu or device == 'cuda':
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
        force_gpu: bool = False,
        num_workers: int = 0,
        device: str = 'auto',
        representation: str = 'one_hot'
    ):
        if representation not in ('one_hot', 'ordinal'):
            raise ValueError('Representation must be one_hot or ordinal')
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in (sequence_length, hidden_size)):
            raise ValueError('Sequence length and hidden size must be positive integers')
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.representation = representation
        self.device = get_optimal_device(force_gpu=force_gpu, device=device)
        self.num_workers = num_workers if self.device.type == "cuda" else 0
        self.pin_memory = self.device.type == "cuda"
        
        self.model = LSTMNetwork(
            input_size=37 if self.representation == 'one_hot' else 1,
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
        history = [validate_number(number) for number in history]
        X, y = [], []
        
        for i in range(len(history) - self.sequence_length):
            seq = history[i:i + self.sequence_length]
            target = history[i + self.sequence_length]
            
            X.append(seq)
            y.append(target)
        
        X_tensor = self._encode_sequences(X)
        y_tensor = torch.LongTensor(y)
        
        return X_tensor, y_tensor

    def _encode_sequences(self, sequences):
        numbers = torch.tensor(sequences, dtype=torch.long).reshape(-1, self.sequence_length)
        if self.representation == 'one_hot':
            return nn.functional.one_hot(numbers, num_classes=37).to(torch.float32)
        return numbers.to(torch.float32).unsqueeze(-1) / 36
    
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
        verbose: bool = False,
        cancel_event=None
    ) -> Dict[str, Any]:
        """
        Train the LSTM model on historical data.
        
        Uses GPU+CPU hybrid:
        - CPU workers preprocess and load batches
        - GPU computes forward/backward passes
        - Async transfers overlap data movement with computation
        """
        if epochs < 1 or batch_size < 1:
            raise ValueError('Epochs and batch size must be positive')
        if len(history) < self.sequence_length + 10:
            return {"status": "error", "message": "Not enough data"}
        
        X, y = self._prepare_sequences(history)
        
        if len(X) < batch_size:
            batch_size = len(X)
        
        dataloader = self._create_dataloader(X, y, batch_size)
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            if cancel_event is not None and cancel_event.is_set():
                return {'status': 'cancelled', 'message': 'Training cancelled'}
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
            "device": str(self.device),
            "representation": self.representation
        }
    
    def predict_proba(self, history: List[int]) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError('LSTM is untrained')
        if len(history) < self.sequence_length:
            raise ValueError('Insufficient history for LSTM inference')
        self.model.eval()
        selected = [validate_number(number) for number in history[-self.sequence_length:]]
        inputs = self._encode_sequences([selected]).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(inputs), dim=1)[0].cpu().numpy()
        return validate_probabilities(probabilities)

    def predict(self, history: List[int]) -> Tuple[int, float]:
        return ranked_numbers(self.predict_proba(history))[0]

    def predict_top_n(self, history: List[int], n: int = 3) -> List[Tuple[int, float]]:
        if not 1 <= n <= 37:
            raise ValueError('Top N must be between 1 and 37')
        return ranked_numbers(self.predict_proba(history))[:n]

    def save(self, path: str):
        """Save model to file."""
        atomic_save(path, {
            'schema_version': SCHEMA_VERSION,
            'observation_version': OBSERVATION_VERSION,
            'model_type': 'lstm',
            'rng': capture_rng(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "representation": self.representation,
            "input_contract_version": 2,
            "is_trained": self.is_trained,
            "training_history": self.training_history
        })
    
    def load(self, path: str, force_gpu: bool = False):
        """Load model from file (forces GPU by default)."""
        self.device = get_optimal_device(force_gpu=force_gpu, device='cuda' if force_gpu else self.device.type)
        checkpoint = load_checkpoint(path, 'lstm', device=self.device)
        
        self.sequence_length = checkpoint.get("sequence_length", 10)
        self.hidden_size = checkpoint.get("hidden_size", 128)
        contract_version = checkpoint.get('input_contract_version', 1)
        if contract_version not in (1, 2):
            raise ValueError('Unsupported LSTM input contract')
        self.representation = checkpoint['representation'] if contract_version == 2 else 'ordinal'
        if self.representation not in ('one_hot', 'ordinal'):
            raise ValueError('Unsupported LSTM representation')
        
        self.model = LSTMNetwork(
            input_size=37 if self.representation == 'one_hot' else 1,
            hidden_size=self.hidden_size,
            output_size=37
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.pin_memory = self.device.type == 'cuda'
        self.stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
        self.is_trained = checkpoint.get("is_trained", True)
        self.training_history = checkpoint.get("training_history", [])
        restore_rng(checkpoint['rng'])
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device configuration info."""
        info = {
            "device": str(self.device),
            "device_type": self.device.type,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
            "representation": self.representation
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
