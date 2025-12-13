"""
Deep Q-Network (DQN) Agent for Roulette - Version 2

Enhanced architecture inspired by FAIRS:
- Embedding + BatchNorm + Dense (no LSTM)
- Dual input: history + gain context
- Double DQN with experience replay
- Epsilon-greedy exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional, Dict, Any


class RouletteEmbedding(nn.Module):
    """
    Custom embedding layer for roulette numbers.
    
    Converts discrete roulette numbers (0-36) to dense vectors.
    """
    
    def __init__(
        self,
        num_embeddings: int = 37,
        embedding_dim: int = 64,
        scale: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scale = scale
        self.embedding_dim = embedding_dim
        
        # Initialize with uniform distribution
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len) with values 0-36
        Returns:
            Tensor of shape (batch, seq_len, embedding_dim)
        """
        embedded = self.embedding(x.long())
        if self.scale:
            embedded = embedded * np.sqrt(self.embedding_dim)
        return embedded


class BatchNormDense(nn.Module):
    """Dense layer with BatchNorm and ReLU activation."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # He initialization
        nn.init.kaiming_uniform_(self.dense.weight, nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class QNetwork(nn.Module):
    """
    Q-Network architecture for roulette (FAIRS-inspired).
    
    Architecture:
        Input: (history[seq_len], gain[1])
        
        History branch:
            Embedding(37 -> embed_dim) -> Flatten -> BatchNormDense -> BatchNormDense
        
        Gain branch:
            BatchNormDense(small) -> BatchNormDense(match history dim)
        
        Merge:
            Add + LayerNorm -> QNet head -> Q-values
    """
    
    def __init__(
        self,
        history_size: int = 20,
        action_size: int = 47,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.history_size = history_size
        self.action_size = action_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # History branch
        self.embedding = RouletteEmbedding(
            num_embeddings=37,
            embedding_dim=embedding_dim
        )
        
        # After embedding: (batch, seq_len, embed_dim) -> flatten -> (batch, seq_len * embed_dim)
        flatten_size = history_size * embedding_dim
        
        self.history_net = nn.Sequential(
            BatchNormDense(flatten_size, hidden_size, dropout),
            BatchNormDense(hidden_size, hidden_size, dropout)
        )
        
        # Gain context branch
        self.gain_net = nn.Sequential(
            BatchNormDense(1, hidden_size // 2, 0),
            BatchNormDense(hidden_size // 2, hidden_size, 0)
        )
        
        # Merge layer (Add + LayerNorm)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Q-Network head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Initialize Q-head
        for layer in self.q_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    
    def forward(
        self, 
        history: torch.Tensor, 
        gain: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            history: Tensor of shape (batch, seq_len) with roulette numbers 0-36
            gain: Tensor of shape (batch, 1) with gain ratio
            
        Returns:
            Q-values of shape (batch, action_size)
        """
        # History branch
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        h = self.embedding(history)
        
        # Flatten: (batch, seq_len, embed_dim) -> (batch, seq_len * embed_dim)
        h = h.view(h.size(0), -1)
        
        # Dense layers
        h = self.history_net(h)
        
        # Gain branch
        g = self.gain_net(gain)
        
        # Merge (Add + LayerNorm)
        merged = self.layer_norm(h + g)
        
        # Q-values
        q_values = self.q_head(merged)
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        history: np.ndarray,
        gain: float,
        action: int,
        reward: float,
        next_history: np.ndarray,
        next_gain: float,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.buffer.append((history, gain, action, reward, next_history, next_gain, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for roulette.
    
    Implements Double DQN with:
    - BatchNorm architecture (no LSTM)
    - Dual input (history + gain)
    - Experience replay
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        history_size: int = 20,
        action_size: int = 47,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        learning_rate: float = 0.0005,
        gamma: float = 0.5,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: Optional[str] = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            history_size: Number of past spins in state
            action_size: Number of possible actions (47)
            embedding_dim: Dimension of number embeddings
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network
            device: Device to use ('cuda' or 'cpu')
        """
        self.history_size = history_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step = 0
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create networks
        self.q_network = QNetwork(
            history_size=history_size,
            action_size=action_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size
        ).to(self.device)
        
        self.target_network = QNetwork(
            history_size=history_size,
            action_size=action_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Store config for saving
        self.config = {
            'history_size': history_size,
            'action_size': action_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'target_update_freq': target_update_freq
        }
    
    def act(
        self, 
        history: np.ndarray, 
        gain: float,
        training: bool = True,
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            history: Array of past roulette numbers
            gain: Current gain ratio (bankroll / initial)
            training: Whether in training mode
            action_mask: Optional mask of valid actions
        
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask == 1)[0]
                return random.choice(valid_actions)
            return random.randint(0, self.action_size - 1)
        
        # Get Q-values from network
        with torch.no_grad():
            # Set to eval mode to avoid batch norm issues with single sample
            self.q_network.eval()
            
            history_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
            gain_tensor = torch.FloatTensor([[gain]]).to(self.device)
            q_values = self.q_network(history_tensor, gain_tensor)
            
            # Back to train mode
            if training:
                self.q_network.train()
            
            if action_mask is not None:
                # Mask invalid actions with very negative value
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                q_values = q_values + (1 - mask_tensor) * (-1e9)
            
            return q_values.argmax(dim=1).item()
    
    def remember(
        self,
        history: np.ndarray,
        gain: float,
        action: int,
        reward: float,
        next_history: np.ndarray,
        next_gain: float,
        done: bool
    ):
        """Store a transition in the replay buffer."""
        self.memory.push(history, gain, action, reward, next_history, next_gain, done)
    
    def train(self) -> Optional[float]:
        """
        Train the agent on a batch from the replay buffer.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        histories, gains, actions, rewards, next_histories, next_gains, dones = zip(*batch)
        
        # Convert to tensors
        histories = torch.FloatTensor(np.array(histories)).to(self.device)
        gains = torch.FloatTensor(np.array(gains)).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_histories = torch.FloatTensor(np.array(next_histories)).to(self.device)
        next_gains = torch.FloatTensor(np.array(next_gains)).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(histories, gains).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.q_network(next_histories, next_gains).argmax(dim=1)
            # Use target network to evaluate
            next_q = self.target_network(next_histories, next_gains).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.train_step += 1
        
        # Update target network periodically
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str, extra_data: dict = None):
        """Save the agent's networks and state."""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'config': self.config
        }
        if extra_data:
            checkpoint.update(extra_data)
        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str) -> dict:
        """Load the agent's networks and state. Returns extra data if present."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        print(f"Agent loaded from {path}")
        return checkpoint  # Return for extra_data access
    
    def get_q_values(self, history: np.ndarray, gain: float) -> np.ndarray:
        """Get Q-values for all actions (for analysis)."""
        with torch.no_grad():
            # Set to eval mode to avoid batch norm issues with single sample
            was_training = self.q_network.training
            self.q_network.eval()
            
            history_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
            gain_tensor = torch.FloatTensor([[gain]]).to(self.device)
            q_values = self.q_network(history_tensor, gain_tensor)
            
            # Restore previous mode
            if was_training:
                self.q_network.train()
            
            return q_values.cpu().numpy()[0]


if __name__ == "__main__":
    # Quick test
    print("Testing DQNAgent v2...")
    
    agent = DQNAgent(
        history_size=20,
        action_size=47,
        embedding_dim=64,
        hidden_size=128
    )
    
    print(f"Device: {agent.device}")
    print(f"Network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # Test action selection
    dummy_history = np.random.randint(0, 37, size=(20,))
    dummy_gain = 1.0
    
    action = agent.act(dummy_history, dummy_gain)
    print(f"\nRandom history: {dummy_history}")
    print(f"Selected action: {action}")
    
    # Test Q-values
    q_vals = agent.get_q_values(dummy_history, dummy_gain)
    print(f"Q-values shape: {q_vals.shape}")
    print(f"Top 5 actions: {np.argsort(q_vals)[-5:][::-1]}")
    print(f"Top 5 Q-values: {np.sort(q_vals)[-5:][::-1]}")
    
    # Test training step
    for i in range(100):
        h = np.random.randint(0, 37, size=(20,))
        g = random.uniform(0.5, 1.5)
        a = random.randint(0, 46)
        r = random.uniform(-50, 50)
        nh = np.random.randint(0, 37, size=(20,))
        ng = g + r/1000
        d = random.random() < 0.1
        agent.remember(h, g, a, r, nh, ng, d)
    
    loss = agent.train()
    print(f"\nTraining loss: {loss}")
