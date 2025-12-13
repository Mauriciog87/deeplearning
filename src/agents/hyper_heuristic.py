"""
Hyper-Heuristic Agent using Reinforcement Learning for meta-level strategy selection.

Based on: "A review of reinforcement learning based hyper-heuristics" (Li et al., 2024)
- High-Level Strategy (HLS): Q-Learning based selector
- Low-Level Heuristics (LLH): Multiple betting strategies

The agent learns WHICH strategy to use WHEN, adapting to game state.
"""
import numpy as np
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import random


class LLHType(Enum):
    """Available Low-Level Heuristics (betting strategies)."""
    HOT_NUMBERS = auto()      # Bet on frequently occurring numbers
    COLD_NUMBERS = auto()     # Bet on numbers that haven't appeared
    SECTOR_BETTING = auto()   # Bet on wheel sectors
    MARTINGALE = auto()       # Double after loss
    ANTI_MARTINGALE = auto()  # Double after win
    FLAT_BETTING = auto()     # Consistent bet size
    FIBONACCI = auto()        # Fibonacci sequence betting
    DALEMBERT = auto()        # Increase/decrease by 1 unit
    PASS_ACTION = auto()      # Skip this round (risk management)
    RANDOM_POLICY = auto()    # Exploration via random selection


@dataclass
class LLHState:
    """State for tracking Low-Level Heuristic performance."""
    total_uses: int = 0
    total_reward: float = 0.0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    last_reward: float = 0.0
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.total_uses)
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / max(1, total)


@dataclass
class HHState:
    """State representation for the Hyper-Heuristic."""
    bankroll_level: int = 0      # 0=low, 1=medium, 2=high
    trend: int = 0               # -1=losing, 0=neutral, 1=winning
    volatility: int = 0          # 0=low, 1=high
    recent_llh_performance: int = 0  # 0=bad, 1=ok, 2=good
    bias_detected: int = 0       # 0=no, 1=yes
    
    def to_tuple(self) -> Tuple[int, ...]:
        return (self.bankroll_level, self.trend, self.volatility, 
                self.recent_llh_performance, self.bias_detected)
    
    @classmethod
    def state_space_size(cls) -> int:
        """Total number of possible states."""
        return 3 * 3 * 2 * 3 * 2  # 108 states


class LowLevelHeuristic:
    """Base class for Low-Level Heuristics."""
    
    WHEEL_ORDER = [
        0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
        5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
    ]
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.history: List[int] = []
        self.bet_history: List[Tuple[int, float]] = []  # (action, reward)
        
    def update_history(self, outcome: int, bet_action: int, reward: float):
        """Update internal state after a spin."""
        self.history.append(outcome)
        self.bet_history.append((bet_action, reward))
        # Keep last 100 spins
        if len(self.history) > 100:
            self.history = self.history[-100:]
        if len(self.bet_history) > 100:
            self.bet_history = self.bet_history[-100:]
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        """Select a betting action. Override in subclasses."""
        raise NotImplementedError
    
    def get_bet_amount(self, bankroll: float, base_bet: float) -> float:
        """Get bet amount. Override for progressive systems."""
        return min(base_bet, bankroll)


class HotNumbersLLH(LowLevelHeuristic):
    """Bet on numbers that appear frequently (hot numbers)."""
    
    def __init__(self, window: int = 30, seed: Optional[int] = None):
        super().__init__(seed)
        self.window = window
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        if len(self.history) < 10:
            return self.rng.integers(0, 37)
        
        recent = self.history[-self.window:]
        counts = np.bincount(recent, minlength=37)
        hot_numbers = np.argsort(counts)[-5:]  # Top 5 hot
        return int(self.rng.choice(hot_numbers))


class ColdNumbersLLH(LowLevelHeuristic):
    """Bet on numbers that haven't appeared recently (cold numbers)."""
    
    def __init__(self, window: int = 50, seed: Optional[int] = None):
        super().__init__(seed)
        self.window = window
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        if len(self.history) < 20:
            return self.rng.integers(0, 37)
        
        recent = self.history[-self.window:]
        counts = np.bincount(recent, minlength=37)
        cold_numbers = np.argsort(counts)[:5]  # 5 coldest
        return int(self.rng.choice(cold_numbers))


class SectorBettingLLH(LowLevelHeuristic):
    """Bet on wheel sectors that show activity."""
    
    def __init__(self, sector_size: int = 5, seed: Optional[int] = None):
        super().__init__(seed)
        self.sector_size = sector_size
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        if len(self.history) < 10:
            return self.rng.integers(0, 37)
        
        # Find most active sector
        recent = self.history[-20:]
        sector_hits = defaultdict(int)
        
        for num in recent:
            if num in self.WHEEL_ORDER:
                idx = self.WHEEL_ORDER.index(num)
                sector_start = (idx // self.sector_size) * self.sector_size
                sector_hits[sector_start] += 1
        
        if not sector_hits:
            return self.rng.integers(0, 37)
        
        # Get hottest sector
        hot_sector_start = max(sector_hits, key=sector_hits.get)
        sector_numbers = []
        for i in range(self.sector_size):
            idx = (hot_sector_start + i) % len(self.WHEEL_ORDER)
            sector_numbers.append(self.WHEEL_ORDER[idx])
        
        return int(self.rng.choice(sector_numbers))


class MartingaleLLH(LowLevelHeuristic):
    """Martingale system: double bet after each loss."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.current_multiplier = 1
        self.last_bet_number = None
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        # Bet on even money (using action 37 = RED or similar)
        # For simplicity, bet on a random number
        if self.last_bet_number is None:
            self.last_bet_number = self.rng.integers(0, 37)
        return self.last_bet_number
    
    def get_bet_amount(self, bankroll: float, base_bet: float) -> float:
        amount = base_bet * self.current_multiplier
        return min(amount, bankroll)
    
    def update_history(self, outcome: int, bet_action: int, reward: float):
        super().update_history(outcome, bet_action, reward)
        if reward > 0:
            self.current_multiplier = 1
            self.last_bet_number = self.rng.integers(0, 37)
        else:
            self.current_multiplier = min(self.current_multiplier * 2, 64)  # Cap at 64x


class AntiMartingaleLLH(LowLevelHeuristic):
    """Anti-Martingale: double bet after each win."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.current_multiplier = 1
        self.winning_number = None
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        if self.winning_number is not None:
            return self.winning_number
        return self.rng.integers(0, 37)
    
    def get_bet_amount(self, bankroll: float, base_bet: float) -> float:
        amount = base_bet * self.current_multiplier
        return min(amount, bankroll)
    
    def update_history(self, outcome: int, bet_action: int, reward: float):
        super().update_history(outcome, bet_action, reward)
        if reward > 0:
            self.current_multiplier = min(self.current_multiplier * 2, 8)  # Cap at 8x
            self.winning_number = outcome
        else:
            self.current_multiplier = 1
            self.winning_number = None


class FlatBettingLLH(LowLevelHeuristic):
    """Flat betting: consistent bet amount, random number selection."""
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        return self.rng.integers(0, 37)


class FibonacciLLH(LowLevelHeuristic):
    """Fibonacci betting system."""
    
    FIB_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.fib_index = 0
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        return self.rng.integers(0, 37)
    
    def get_bet_amount(self, bankroll: float, base_bet: float) -> float:
        multiplier = self.FIB_SEQUENCE[min(self.fib_index, len(self.FIB_SEQUENCE) - 1)]
        return min(base_bet * multiplier, bankroll)
    
    def update_history(self, outcome: int, bet_action: int, reward: float):
        super().update_history(outcome, bet_action, reward)
        if reward > 0:
            self.fib_index = max(0, self.fib_index - 2)
        else:
            self.fib_index = min(self.fib_index + 1, len(self.FIB_SEQUENCE) - 1)


class DAlembertLLH(LowLevelHeuristic):
    """D'Alembert system: +1 after loss, -1 after win."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.current_units = 1
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        return self.rng.integers(0, 37)
    
    def get_bet_amount(self, bankroll: float, base_bet: float) -> float:
        return min(base_bet * self.current_units, bankroll)
    
    def update_history(self, outcome: int, bet_action: int, reward: float):
        super().update_history(outcome, bet_action, reward)
        if reward > 0:
            self.current_units = max(1, self.current_units - 1)
        else:
            self.current_units = min(self.current_units + 1, 20)


class PassActionLLH(LowLevelHeuristic):
    """Pass/skip action for risk management."""
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        return 46  # PASS action
    
    def get_bet_amount(self, bankroll: float, base_bet: float) -> float:
        return 0


class RandomPolicyLLH(LowLevelHeuristic):
    """Random action for exploration."""
    
    def select_action(self, bankroll: float, base_bet: float) -> int:
        return self.rng.integers(0, 47)  # Any action including PASS


class HyperHeuristicAgent:
    """
    Reinforcement Learning based Hyper-Heuristic Agent.
    
    Uses Q-Learning at the meta-level to select which Low-Level Heuristic
    to use based on the current game state.
    
    Architecture:
    - High-Level Strategy (HLS): Q-Learning selector
    - Low-Level Heuristics (LLH): Multiple betting strategies
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        initial_bankroll: float = 1000.0,
        base_bet: float = 10.0,
        seed: Optional[int] = None
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.base_bet = base_bet
        
        self.rng = np.random.default_rng(seed)
        seed_seq = self.rng.integers(0, 10000, size=10)
        
        # Initialize Low-Level Heuristics
        self.llhs: Dict[LLHType, LowLevelHeuristic] = {
            LLHType.HOT_NUMBERS: HotNumbersLLH(seed=int(seed_seq[0])),
            LLHType.COLD_NUMBERS: ColdNumbersLLH(seed=int(seed_seq[1])),
            LLHType.SECTOR_BETTING: SectorBettingLLH(seed=int(seed_seq[2])),
            LLHType.MARTINGALE: MartingaleLLH(seed=int(seed_seq[3])),
            LLHType.ANTI_MARTINGALE: AntiMartingaleLLH(seed=int(seed_seq[4])),
            LLHType.FLAT_BETTING: FlatBettingLLH(seed=int(seed_seq[5])),
            LLHType.FIBONACCI: FibonacciLLH(seed=int(seed_seq[6])),
            LLHType.DALEMBERT: DAlembertLLH(seed=int(seed_seq[7])),
            LLHType.PASS_ACTION: PassActionLLH(seed=int(seed_seq[8])),
            LLHType.RANDOM_POLICY: RandomPolicyLLH(seed=int(seed_seq[9])),
        }
        
        # Q-Table: state -> {llh_type -> Q-value}
        self.q_table: Dict[Tuple, Dict[LLHType, float]] = defaultdict(
            lambda: {llh: 0.0 for llh in LLHType}
        )
        
        # Performance tracking per LLH
        self.llh_states: Dict[LLHType, LLHState] = {
            llh: LLHState() for llh in LLHType
        }
        
        # Game history
        self.spin_history: List[int] = []
        self.reward_history: List[float] = []
        self.llh_selection_history: List[LLHType] = []
        
        # Current state tracking
        self.current_hh_state: HHState = HHState()
        self.current_llh: Optional[LLHType] = None
        self.bias_detected: bool = False
        
        # Statistics
        self.total_episodes = 0
        self.total_steps = 0
        self.wins = 0
        self.losses = 0
    
    def _compute_hh_state(self) -> HHState:
        """Compute the current HH state from game history."""
        state = HHState()
        
        # Bankroll level
        ratio = self.bankroll / self.initial_bankroll
        if ratio < 0.5:
            state.bankroll_level = 0  # Low
        elif ratio < 1.5:
            state.bankroll_level = 1  # Medium
        else:
            state.bankroll_level = 2  # High
        
        # Trend (last 10 rewards)
        if len(self.reward_history) >= 5:
            recent_sum = sum(self.reward_history[-10:])
            if recent_sum < -self.base_bet * 3:
                state.trend = -1  # Losing
            elif recent_sum > self.base_bet * 3:
                state.trend = 1   # Winning
            else:
                state.trend = 0   # Neutral
        
        # Volatility
        if len(self.reward_history) >= 10:
            recent = self.reward_history[-10:]
            std = np.std(recent)
            state.volatility = 1 if std > self.base_bet * 2 else 0
        
        # Recent LLH performance
        if self.current_llh and self.llh_states[self.current_llh].total_uses > 0:
            avg = self.llh_states[self.current_llh].avg_reward
            if avg < -self.base_bet * 0.5:
                state.recent_llh_performance = 0  # Bad
            elif avg > self.base_bet * 0.5:
                state.recent_llh_performance = 2  # Good
            else:
                state.recent_llh_performance = 1  # OK
        
        # Bias detection (simplified)
        state.bias_detected = 1 if self.bias_detected else 0
        
        return state
    
    def select_llh(self, explore: bool = True) -> LLHType:
        """Select a Low-Level Heuristic using epsilon-greedy Q-Learning."""
        self.current_hh_state = self._compute_hh_state()
        state_key = self.current_hh_state.to_tuple()
        
        # Epsilon-greedy selection
        if explore and self.rng.random() < self.epsilon:
            selected = self.rng.choice(list(LLHType))
        else:
            # Greedy: select LLH with highest Q-value
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            best_llhs = [llh for llh, q in q_values.items() if q == max_q]
            selected = self.rng.choice(best_llhs)
        
        self.current_llh = selected
        self.llh_selection_history.append(selected)
        return selected
    
    def select_action(self, explore: bool = True) -> Tuple[int, float]:
        """
        Select an action (bet number and amount) using the current LLH.
        
        Returns:
            Tuple of (action_index, bet_amount)
        """
        if self.current_llh is None:
            self.select_llh(explore)
        
        llh = self.llhs[self.current_llh]
        action = llh.select_action(self.bankroll, self.base_bet)
        bet_amount = llh.get_bet_amount(self.bankroll, self.base_bet)
        
        return action, bet_amount
    
    def update(self, outcome: int, action: int, reward: float):
        """
        Update after observing the result of a spin.
        
        Args:
            outcome: The winning number (0-36)
            action: The action that was taken
            reward: The reward received
        """
        # Update histories
        self.spin_history.append(outcome)
        self.reward_history.append(reward)
        self.bankroll += reward
        
        # Keep history bounded
        if len(self.spin_history) > 500:
            self.spin_history = self.spin_history[-500:]
        if len(self.reward_history) > 500:
            self.reward_history = self.reward_history[-500:]
        
        # Update LLH statistics
        if self.current_llh:
            llh_state = self.llh_states[self.current_llh]
            llh_state.total_uses += 1
            llh_state.total_reward += reward
            llh_state.last_reward = reward
            
            if reward > 0:
                llh_state.wins += 1
                llh_state.consecutive_wins += 1
                llh_state.consecutive_losses = 0
                self.wins += 1
            elif reward < 0:
                llh_state.losses += 1
                llh_state.consecutive_losses += 1
                llh_state.consecutive_wins = 0
                self.losses += 1
            
            # Update the LLH's internal state
            self.llhs[self.current_llh].update_history(outcome, action, reward)
        
        # Q-Learning update
        old_state_key = self.current_hh_state.to_tuple()
        new_state = self._compute_hh_state()
        new_state_key = new_state.to_tuple()
        
        if self.current_llh:
            old_q = self.q_table[old_state_key][self.current_llh]
            max_next_q = max(self.q_table[new_state_key].values())
            
            # Q-Learning update rule
            new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
            self.q_table[old_state_key][self.current_llh] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.total_steps += 1
        
        # Check if we should switch LLH (every 5 steps or on big loss)
        should_switch = (
            self.total_steps % 5 == 0 or
            (self.current_llh and self.llh_states[self.current_llh].consecutive_losses >= 3)
        )
        
        if should_switch:
            self.select_llh(explore=True)
    
    def set_bias_detected(self, detected: bool, hot_numbers: Optional[List[int]] = None):
        """Set bias detection flag and optionally update hot numbers LLH."""
        self.bias_detected = detected
        # Could enhance hot numbers LLH with detected bias info
    
    def reset_episode(self):
        """Reset for a new episode."""
        self.bankroll = self.initial_bankroll
        self.reward_history.clear()
        self.current_llh = None
        self.total_episodes += 1
        
        # Reset progressive betting systems
        for llh in [LLHType.MARTINGALE, LLHType.ANTI_MARTINGALE, 
                    LLHType.FIBONACCI, LLHType.DALEMBERT]:
            self.llhs[llh] = type(self.llhs[llh])(seed=self.rng.integers(0, 10000))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        llh_stats = {}
        for llh_type, state in self.llh_states.items():
            if state.total_uses > 0:
                llh_stats[llh_type.name] = {
                    'uses': state.total_uses,
                    'avg_reward': state.avg_reward,
                    'win_rate': state.win_rate,
                }
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'current_bankroll': self.bankroll,
            'epsilon': self.epsilon,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / max(1, self.wins + self.losses),
            'q_table_size': len(self.q_table),
            'llh_statistics': llh_stats,
        }
    
    def get_best_llh_for_state(self, state: Optional[HHState] = None) -> LLHType:
        """Get the best LLH for a given state (or current state)."""
        if state is None:
            state = self._compute_hh_state()
        
        state_key = state.to_tuple()
        q_values = self.q_table[state_key]
        return max(q_values, key=q_values.get)
    
    def save(self, filepath: str):
        """Save the agent to a file."""
        import pickle
        
        state = {
            'q_table': dict(self.q_table),
            'llh_states': self.llh_states,
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'wins': self.wins,
            'losses': self.losses,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> 'HyperHeuristicAgent':
        """Load an agent from a file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        agent = cls(**kwargs)
        agent.q_table = defaultdict(
            lambda: {llh: 0.0 for llh in LLHType},
            state['q_table']
        )
        agent.llh_states = state['llh_states']
        agent.epsilon = state['epsilon']
        agent.total_episodes = state['total_episodes']
        agent.total_steps = state['total_steps']
        agent.wins = state['wins']
        agent.losses = state['losses']
        
        return agent


class DQNHyperHeuristic(HyperHeuristicAgent):
    """
    Deep Q-Network based Hyper-Heuristic Agent.
    
    Uses a neural network instead of Q-table for better generalization
    in larger state spaces.
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            n_actions = len(LLHType)
            
            # Simple MLP for Q-value estimation
            self.q_network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            ).to(self.device)
            
            self.target_network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            ).to(self.device)
            
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
            
            # Experience replay
            self.replay_buffer: List[Tuple] = []
            self.buffer_size = 10000
            self.batch_size = 32
            self.update_target_every = 100
            
            self.torch_available = True
            
        except ImportError:
            self.torch_available = False
            print("Warning: PyTorch not available. DQNHyperHeuristic will use Q-table.")
    
    def _state_to_tensor(self, state: HHState):
        """Convert HHState to tensor."""
        import torch
        
        features = [
            state.bankroll_level / 2.0,
            (state.trend + 1) / 2.0,
            state.volatility,
            state.recent_llh_performance / 2.0,
            state.bias_detected,
        ]
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)
    
    def select_llh(self, explore: bool = True) -> LLHType:
        """Select LLH using DQN."""
        if not self.torch_available:
            return super().select_llh(explore)
        
        import torch
        
        self.current_hh_state = self._compute_hh_state()
        state_tensor = self._state_to_tensor(self.current_hh_state)
        
        # Epsilon-greedy
        if explore and self.rng.random() < self.epsilon:
            selected_idx = self.rng.integers(0, len(LLHType))
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                selected_idx = q_values.argmax().item()
        
        self.current_llh = list(LLHType)[selected_idx]
        self.llh_selection_history.append(self.current_llh)
        return self.current_llh
    
    def update(self, outcome: int, action: int, reward: float):
        """Update with experience replay for DQN."""
        if not self.torch_available:
            return super().update(outcome, action, reward)
        
        import torch
        
        old_state = self._compute_hh_state()
        
        # Call parent update for history and stats
        super().update(outcome, action, reward)
        
        new_state = self._compute_hh_state()
        
        # Store experience
        if self.current_llh:
            llh_idx = list(LLHType).index(self.current_llh)
            self.replay_buffer.append((
                old_state, llh_idx, reward, new_state
            ))
            
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)
        
        # Train if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
        
        # Update target network
        if self.total_steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _train_step(self):
        """Perform one training step."""
        import torch
        
        # Sample batch
        batch_indices = self.rng.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        states = torch.stack([self._state_to_tensor(s).squeeze() for s, _, _, _ in batch])
        actions = torch.LongTensor([a for _, a, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _ in batch]).to(self.device)
        next_states = torch.stack([self._state_to_tensor(ns).squeeze() for _, _, _, ns in batch])
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q
        
        # Update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
