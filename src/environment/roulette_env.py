"""
Roulette Environment for Reinforcement Learning - Version 3

Enhanced environment with:
- Unified action space (47 actions like FAIRS)
- Gain input (capital context)
- Support for real data from SQLite
- European roulette (0-36)
- Near-miss features based on research paper
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..settlement import settle_bet, validate_number, validate_stake, settle_action
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

# Import near-miss utilities
try:
    from ..utils.near_miss import (
        compute_near_miss_features,
        get_wheel_neighbors,
        get_table_neighbors,
        get_wheel_distance,
        is_near_miss_wheel,
        is_near_miss_table,
        encode_wheel_position,
        compute_wheel_sector_frequencies,
        NearMissTracker,
        WHEEL_NEIGHBORS_2,
        TABLE_NEIGHBORS
    )
    NEAR_MISS_AVAILABLE = True
except ImportError:
    NEAR_MISS_AVAILABLE = False


# Roulette number colors (European roulette: 0-36)
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
GREEN_NUMBERS = {0}

# Number ranges
LOW_NUMBERS = set(range(1, 19))      # 1-18
HIGH_NUMBERS = set(range(19, 37))    # 19-36
ODD_NUMBERS = {n for n in range(1, 37) if n % 2 == 1}
EVEN_NUMBERS = {n for n in range(1, 37) if n % 2 == 0}

# Dozens
FIRST_DOZEN = set(range(1, 13))      # 1-12
SECOND_DOZEN = set(range(13, 25))    # 13-24
THIRD_DOZEN = set(range(25, 37))     # 25-36

# Columns
FIRST_COLUMN = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34}
SECOND_COLUMN = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}
THIRD_COLUMN = {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36}


@dataclass
class ActionInfo:
    """Information about an action/bet."""
    id: int
    name: str
    bet_type: str
    payout: int
    winning_numbers: set
    

class RouletteEnv(gym.Env):
    """
    Enhanced Casino Roulette Environment
    
    State: 
        - Last N numbers rolled (integers 0-36)
        - Current gain ratio (bankroll / initial_bankroll)
    
    Actions (47 total):
        - 0-36: Bet on specific number (35:1)
        - 37: Bet on Red (1:1)
        - 38: Bet on Black (1:1)
        - 39: Bet on Odd (1:1)
        - 40: Bet on Even (1:1)
        - 41: Bet on Low 1-18 (1:1)
        - 42: Bet on High 19-36 (1:1)
        - 43: Bet on 1st Dozen (2:1)
        - 44: Bet on 2nd Dozen (2:1)
        - 45: Bet on 3rd Dozen (2:1)
        - 46: PASS (no bet)
    
    Reward: Net change in bankroll
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Action definitions
    NUM_ACTIONS = 47
    
    # Action constants
    ACTION_RED = 37
    ACTION_BLACK = 38
    ACTION_ODD = 39
    ACTION_EVEN = 40
    ACTION_LOW = 41
    ACTION_HIGH = 42
    ACTION_DOZEN_1 = 43
    ACTION_DOZEN_2 = 44
    ACTION_DOZEN_3 = 45
    ACTION_PASS = 46
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        bet_size: float = 1.0,
        max_steps: int = 80,
        win_target: float = 2000.0,
        history_size: int = 20,
        render_mode: Optional[str] = None,
        real_data: Optional[List[int]] = None,
        use_physics_sim: bool = False
    ):
        """
        Initialize the Roulette environment.
        
        Args:
            initial_bankroll: Starting money
            bet_size: Fixed bet amount per round
            max_steps: Maximum steps before episode ends
            win_target: Target bankroll to win
            history_size: Number of past numbers to observe
            render_mode: Rendering mode
            real_data: List of real roulette numbers to use instead of simulation
            use_physics_sim: Whether to use physics simulation (ignored if real_data provided)
        """
        super().__init__()
        if initial_bankroll <= 0 or not np.isfinite(initial_bankroll):
            raise ValueError('Initial bankroll must be positive and finite')
        validate_stake(bet_size, initial_bankroll)
        if bet_size == 0 or max_steps < 1 or history_size < 1:
            raise ValueError('Bet size, max steps and history size must be positive')
        
        self.initial_bankroll = initial_bankroll
        self.bet_size = bet_size
        self.max_steps = max_steps
        self.win_target = win_target
        self.history_size = history_size
        self.render_mode = render_mode
        self.use_physics_sim = use_physics_sim
        
        # Real data support
        self.real_data = None if real_data is None else [validate_number(n) for n in real_data]
        self.real_data_index = 0
        self._episode_done = True
        
        # Action space: 47 discrete actions
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        
        # Build action info
        self._build_action_info()
        
        # Observation space:
        # - history_size numbers (each 0-36) 
        # - Plus gain ratio (0 to ~2+)
        # Using Dict space for clarity
        self.observation_space = spaces.Dict({
            "history": spaces.Box(
                low=0, high=36, shape=(history_size,), dtype=np.int32
            ),
            "gain": spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )
        })
        
        # Internal state
        self.bankroll = initial_bankroll
        self.history = []
        self.current_step = 0
        self.last_pos = 0
        self.counter_clockwise = True
        
    def _build_action_info(self):
        """Build information about each action."""
        self.actions = {}
        self.action_names = []
        
        # Straight up bets (0-36)
        for i in range(37):
            self.actions[i] = ActionInfo(
                id=i,
                name=str(i),
                bet_type="straight",
                payout=35,
                winning_numbers={i}
            )
            self.action_names.append(str(i))
        
        # Outside bets
        outside_bets = [
            (self.ACTION_RED, "RED", "color", 1, RED_NUMBERS),
            (self.ACTION_BLACK, "BLACK", "color", 1, BLACK_NUMBERS),
            (self.ACTION_ODD, "ODD", "parity", 1, ODD_NUMBERS),
            (self.ACTION_EVEN, "EVEN", "parity", 1, EVEN_NUMBERS),
            (self.ACTION_LOW, "LOW", "half", 1, LOW_NUMBERS),
            (self.ACTION_HIGH, "HIGH", "half", 1, HIGH_NUMBERS),
            (self.ACTION_DOZEN_1, "DOZEN_1", "dozen", 2, FIRST_DOZEN),
            (self.ACTION_DOZEN_2, "DOZEN_2", "dozen", 2, SECOND_DOZEN),
            (self.ACTION_DOZEN_3, "DOZEN_3", "dozen", 2, THIRD_DOZEN),
            (self.ACTION_PASS, "PASS", "pass", 0, set()),
        ]
        
        for action_id, name, bet_type, payout, winning in outside_bets:
            self.actions[action_id] = ActionInfo(
                id=action_id,
                name=name,
                bet_type=bet_type,
                payout=payout,
                winning_numbers=winning
            )
            self.action_names.append(name)
    
    def _spin_wheel(self) -> int:
        """
        Get the next roulette number.
        
        Uses real data if available, otherwise simulates.
        """
        # Use real data if available
        if self.real_data is not None:
            if self.real_data_index >= len(self.real_data):
                raise RuntimeError('Real data exhausted; reset with a valid segment')
            number = self.real_data[self.real_data_index]
            self.real_data_index += 1
            return number
        
        # Physics simulation
        if self.use_physics_sim:
            rand_strength = int(self.np_random.integers(350, 501))
            rand_shift = int(self.np_random.integers(6, 13))
            
            if self.counter_clockwise:
                run_ball = self.last_pos + rand_shift - rand_strength
            else:
                run_ball = self.last_pos - rand_shift + rand_strength
            
            self.counter_clockwise = not self.counter_clockwise
            self.last_pos = run_ball % 37
            return self.last_pos
        
        # Pure random
        return int(self.np_random.integers(0, 37))
    
    def _get_color(self, number: int) -> str:
        """Get the color of a roulette number."""
        if number in RED_NUMBERS:
            return "red"
        elif number in BLACK_NUMBERS:
            return "black"
        else:
            return "green"
    
    def _calculate_reward(self, action: int, winning_number: int, stake: Optional[float] = None) -> float:
        """
        Calculate the reward based on action and winning number.
        
        Returns the net change in bankroll.
        """
        action_info = self.actions[action]
        
        return settle_bet(winning_number, action_info.winning_numbers,
                          self.bet_size if stake is None else stake, self.bankroll,
                          payout=action_info.payout, separate_straights=False).net_profit
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        # Pad history if needed
        if len(self.history) < self.history_size:
            padding = [0] * (self.history_size - len(self.history))
            hist = padding + self.history
        else:
            hist = self.history[-self.history_size:]
        
        # Compute gain ratio
        gain = self.bankroll / self.initial_bankroll
        
        return {
            "history": np.array(hist, dtype=np.int32),
            "gain": np.array([gain], dtype=np.float32)
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.bankroll = self.initial_bankroll
        self.current_step = 0
        self.last_pos = int(self.np_random.integers(0, 37))
        self.counter_clockwise = True
        self.action_space.seed(seed)
        self._episode_done = False
        
        # Reset real data index if using real data
        options = options or {}
        self.real_data_index = int(options.get('start_index', 0))
        if self.real_data is not None:
            remaining = len(self.real_data) - self.history_size
            if remaining < 1:
                raise ValueError('Real data requires history plus at least one outcome')
            if options.get('random_start', False):
                self.real_data_index = int(self.np_random.integers(0, remaining))
            if not 0 <= self.real_data_index < remaining:
                raise ValueError('Start index does not leave history and an outcome')
        
        # Pre-fill history with spins
        self.history = []
        for _ in range(self.history_size):
            self.history.append(self._spin_wheel())
        
        info = {
            "bankroll": self.bankroll,
            "step": self.current_step,
            "gain": 1.0
        }
        
        return self._get_observation(), info
    
    def step(self, action: int, *, stake: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The betting action to take (0-46)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self._episode_done:
            raise RuntimeError('Episode ended; call reset before stepping')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action!r}')
        effective_stake = self.bet_size if stake is None else float(stake)
        if action == self.ACTION_PASS:
            effective_stake = 0.0
        elif effective_stake <= 0:
            raise ValueError('A betting action requires a positive stake')
        validate_stake(effective_stake, self.bankroll)
        self.current_step += 1
        
        # Spin the wheel
        winning_number = self._spin_wheel()
        self.history.append(winning_number)
        
        # Calculate reward
        action_info = self.actions[action]
        settlement = settle_action(winning_number, action, effective_stake, self.bankroll)
        reward = settlement.net_profit
        self.bankroll = settlement.balance_after
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if self.bankroll <= 0:
            terminated = True
        elif self.bankroll >= self.win_target:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True
        if self.real_data is not None and self.real_data_index >= len(self.real_data):
            truncated = True
        self._episode_done = terminated or truncated
        
        # Get action info
        action_info = self.actions[action]
        
        info = {
            "bankroll": self.bankroll,
            "gain": self.bankroll / self.initial_bankroll,
            "winning_number": winning_number,
            "winning_color": self._get_color(winning_number),
            "action_name": action_info.name,
            "action_type": action_info.bet_type,
            "bet_won": winning_number in action_info.winning_numbers if action != self.ACTION_PASS else None,
            "step": self.current_step,
            "net_reward": reward,
            "total_stake": settlement.total_stake,
            "balance_before": settlement.balance_before,
            "balance_after": settlement.balance_after,
            "action_mask": self.get_action_mask()
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode in ["human", "ansi"]:
            last_numbers = self.history[-5:] if len(self.history) >= 5 else self.history
            numbers_str = " ".join([f"{n:2d}" for n in last_numbers])
            gain = self.bankroll / self.initial_bankroll
            print(f"Step {self.current_step:3d} | Bankroll: ${self.bankroll:,.0f} | Gain: {gain:.2f} | Last: [{numbers_str}]")
    
    def close(self):
        """Clean up resources."""
        pass
    
    def set_real_data(self, numbers: List[int], start_index: int = 0):
        """
        Set real data for the environment to use.
        
        Args:
            numbers: List of roulette numbers (0-36)
            start_index: Starting index in the data
        """
        self.real_data = [validate_number(n) for n in numbers]
        self.real_data_index = start_index
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get a mask of valid actions based on current bankroll.
        
        Returns array where 1 = valid, 0 = invalid
        """
        mask = np.ones(self.NUM_ACTIONS, dtype=np.int32)
        
        # Can't bet if not enough money
        if self.bankroll < self.bet_size:
            mask[:self.ACTION_PASS] = 0  # All bets invalid
            # Only PASS is valid
        
        return mask


# Wrapper for flat observation space (for simpler agents)
class RouletteEnvFlat(RouletteEnv):
    """
    Roulette environment with flattened observation space.
    
    Observation is a single array: [history..., gain]
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override observation space to be flat
        self.observation_space = spaces.Box(
            low=0, 
            high=np.array([36] * self.history_size + [np.inf], dtype=np.float32),
            shape=(self.history_size + 1,), 
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get flattened observation."""
        obs_dict = super()._get_observation()
        
        # Concatenate history and gain
        # Scale gain to be in similar range as numbers
        scaled_gain = obs_dict["gain"][0]
        
        return np.concatenate([
            obs_dict["history"].astype(np.float32),
            [scaled_gain]
        ]).astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._get_observation(), info
    
    def step(self, action, *, stake=None):
        obs, reward, terminated, truncated, info = super().step(action, stake=stake)
        return self._get_observation(), reward, terminated, truncated, info


class RouletteEnvNearMiss(RouletteEnv):
    """
    Roulette environment with near-miss features.
    
    Based on "The influence of near-miss events on betting behavior" paper.
    
    Additional observation features:
    - Near-miss zone mask (37 numbers): which numbers are "near" the last winning number
    - Wheel cluster density: how clustered recent numbers are on the wheel
    - Table cluster density: how clustered recent numbers are on the table
    - Wheel sector frequencies (8 sectors): distribution of recent numbers
    - Last number wheel position encoding (sin, cos)
    
    Total additional features: 37 + 2 + 8 + 2 = 49
    """
    
    # Number of additional near-miss features
    NEAR_MISS_ZONE_SIZE = 37  # Binary mask for each number
    SECTOR_COUNT = 8  # Number of wheel sectors
    NUM_NEAR_MISS_FEATURES = NEAR_MISS_ZONE_SIZE + 2 + SECTOR_COUNT + 2  # 49 total
    
    def __init__(
        self,
        wheel_threshold: int = 3,
        include_table_proximity: bool = True,
        **kwargs
    ):
        """
        Initialize the near-miss environment.
        
        Args:
            wheel_threshold: Wheel distance to consider "near" (default 3)
            include_table_proximity: Also include table-adjacent numbers in zone
            **kwargs: Arguments passed to parent RouletteEnv
        """
        super().__init__(**kwargs)
        
        if not NEAR_MISS_AVAILABLE:
            raise ImportError("Near-miss utilities not available. Check imports.")
        
        self.wheel_threshold = wheel_threshold
        self.include_table_proximity = include_table_proximity
        
        # Near-miss tracker for statistics
        self.near_miss_tracker = NearMissTracker(wheel_threshold)
        
        # Override observation space to include near-miss features
        self.observation_space = spaces.Dict({
            "history": spaces.Box(
                low=0, high=36, shape=(self.history_size,), dtype=np.int32
            ),
            "gain": spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "near_miss_zone": spaces.Box(
                low=0, high=1, shape=(self.NEAR_MISS_ZONE_SIZE,), dtype=np.float32
            ),
            "wheel_features": spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float32  # cluster, sector_bias, sin, cos
            ),
            "sector_frequencies": spaces.Box(
                low=0, high=1, shape=(self.SECTOR_COUNT,), dtype=np.float32
            )
        })
    
    def _compute_near_miss_zone(self, last_number: int) -> np.ndarray:
        """
        Compute binary mask indicating "near" numbers to the last winning number.
        
        Args:
            last_number: The most recent winning number
            
        Returns:
            Binary array of shape (37,) where 1 = near-miss zone
        """
        zone = np.zeros(37, dtype=np.float32)
        
        # Get wheel neighbors
        wheel_neighbors = get_wheel_neighbors(last_number, self.wheel_threshold)
        for n in wheel_neighbors:
            zone[n] = 1.0
        
        # Optionally add table neighbors
        if self.include_table_proximity:
            table_neighbors = get_table_neighbors(last_number)
            for n in table_neighbors:
                zone[n] = 1.0
        
        return zone
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation with near-miss features."""
        # Get base observation
        if len(self.history) < self.history_size:
            padding = [0] * (self.history_size - len(self.history))
            hist = padding + self.history
        else:
            hist = self.history[-self.history_size:]
        
        gain = self.bankroll / self.initial_bankroll
        
        # Near-miss features
        last_number = self.history[-1] if self.history else 0
        
        # Near-miss zone mask
        near_miss_zone = self._compute_near_miss_zone(last_number)
        
        # Compute features from history
        features = compute_near_miss_features(self.history, window=10)
        
        # Wheel position encoding
        sin_pos, cos_pos = encode_wheel_position(last_number)
        
        # Wheel features: cluster density, table cluster, sin, cos
        wheel_features = np.array([
            features["wheel_cluster_density"],
            features["table_cluster_density"],
            sin_pos,
            cos_pos
        ], dtype=np.float32)
        
        # Sector frequencies
        sector_freq = compute_wheel_sector_frequencies(self.history[-20:], self.SECTOR_COUNT)
        
        return {
            "history": np.array(hist, dtype=np.int32),
            "gain": np.array([gain], dtype=np.float32),
            "near_miss_zone": near_miss_zone,
            "wheel_features": wheel_features,
            "sector_frequencies": sector_freq.astype(np.float32)
        }
    
    def step(self, action: int, *, stake: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step with near-miss tracking.
        """
        # Get winning number before step
        obs, reward, terminated, truncated, info = super().step(action, stake=stake)
        
        # Track near-miss for straight bets
        winning_number = info["winning_number"]
        if action <= 36:  # Straight bet
            nm_result = self.near_miss_tracker.record_bet(action, winning_number)
            info["near_miss_wheel"] = nm_result["near_miss_wheel"]
            info["near_miss_table"] = nm_result["near_miss_table"]
        else:
            info["near_miss_wheel"] = False
            info["near_miss_table"] = False
        
        # Re-compute observation with near-miss features
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset with near-miss tracker."""
        obs, info = super().reset(**kwargs)
        self.near_miss_tracker.reset()
        
        # Re-compute observation with near-miss features
        obs = self._get_observation()
        
        # Add near-miss stats to info
        info["near_miss_stats"] = self.near_miss_tracker.get_statistics()
        
        return obs, info
    
    def get_near_miss_statistics(self) -> Dict[str, float]:
        """Get near-miss statistics for the current episode."""
        return self.near_miss_tracker.get_statistics()


class RouletteEnvNearMissFlat(RouletteEnvNearMiss):
    """
    Near-miss environment with flattened observation space.
    
    Observation is a single array combining all features:
    [history (20), gain (1), near_miss_zone (37), wheel_features (4), sector_freq (8)] = 70 total
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Total observation size
        obs_size = self.history_size + 1 + self.NEAR_MISS_ZONE_SIZE + 4 + self.SECTOR_COUNT
        
        self.observation_space = spaces.Box(
            low=-1,  # Some features can be negative (sin/cos)
            high=np.array([36] * self.history_size + [np.inf] + [1] * (obs_size - self.history_size - 1), dtype=np.float32),
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get flattened observation with near-miss features."""
        obs_dict = super()._get_observation()
        
        return np.concatenate([
            obs_dict["history"].astype(np.float32),
            obs_dict["gain"],
            obs_dict["near_miss_zone"],
            obs_dict["wheel_features"],
            obs_dict["sector_frequencies"]
        ])
    
    def reset(self, **kwargs):
        obs_dict, info = RouletteEnvNearMiss.reset(self, **kwargs)
        return self._get_observation(), info
    
    def step(self, action, *, stake=None):
        obs_dict, reward, terminated, truncated, info = RouletteEnvNearMiss.step(self, action, stake=stake)
        return self._get_observation(), reward, terminated, truncated, info


if __name__ == "__main__":
    # Test the environment
    print("Testing RouletteEnv...")
    
    env = RouletteEnv()
    obs, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space} ({env.NUM_ACTIONS} actions)")
    print(f"\nInitial observation:")
    print(f"  History: {obs['history']}")
    print(f"  Gain: {obs['gain']}")
    print(f"  Info: {info}")
    
    # Test a few steps
    print("\n--- Testing actions ---")
    test_actions = [0, 17, 37, 41, 43, 46]  # Number, Red, Low, Dozen1, Pass
    
    for action in test_actions:
        obs, reward, term, trunc, info = env.step(action)
        print(f"Action {action} ({env.action_names[action]:>8}): "
              f"won={info['bet_won']}, reward=${reward:+.0f}, "
              f"bankroll=${info['bankroll']:.0f}, number={info['winning_number']}")
    
    # Test with real data
    print("\n--- Testing with real data ---")
    real_numbers = [17, 23, 0, 5, 32, 15, 19, 4, 21, 2]
    env.set_real_data(real_numbers)
    obs, info = env.reset(options={"reset_data_index": True})
    
    for i in range(5):
        obs, reward, term, trunc, info = env.step(46)  # PASS
        print(f"Spin {i+1}: {info['winning_number']} ({info['winning_color']})")
    
    # Test near-miss environment
    print("\n\n=== Testing RouletteEnvNearMiss ===")
    if NEAR_MISS_AVAILABLE:
        env_nm = RouletteEnvNearMiss()
        obs, info = env_nm.reset()
        
        print(f"Near-miss observation space: {env_nm.observation_space}")
        print(f"\nObservation keys: {obs.keys()}")
        print(f"  History shape: {obs['history'].shape}")
        print(f"  Near-miss zone shape: {obs['near_miss_zone'].shape}")
        print(f"  Near-miss zone sum: {obs['near_miss_zone'].sum():.0f} numbers in zone")
        print(f"  Wheel features: {obs['wheel_features']}")
        print(f"  Sector frequencies: {obs['sector_frequencies']}")
        
        # Test some straight bets
        print("\n--- Testing straight bets with near-miss tracking ---")
        for _ in range(5):
            action = random.randint(0, 36)  # Random straight bet
            obs, reward, term, trunc, info = env_nm.step(action)
            print(f"Bet {action:2d}, Won {info['winning_number']:2d}: "
                  f"wheel_nm={info['near_miss_wheel']}, table_nm={info['near_miss_table']}, "
                  f"reward={reward:+.0f}")
        
        print(f"\nNear-miss statistics: {env_nm.get_near_miss_statistics()}")
        
        # Test flat version
        print("\n--- Testing RouletteEnvNearMissFlat ---")
        env_flat = RouletteEnvNearMissFlat()
        obs, info = env_flat.reset()
        print(f"Flat observation shape: {obs.shape}")
        print(f"Flat observation space: {env_flat.observation_space}")
    else:
        print("Near-miss utilities not available, skipping test.")
