"""
Behavioral Betting Agents - Based on Real Casino Research

Implements betting agents that simulate human behavioral biases
observed in the paper "The influence of near-miss events on betting behavior"
which analyzed 565,915 real betting decisions.

Key findings from the paper:
- 67% of players exhibit Gambler's Fallacy (avoid near-miss numbers)
- 33% of players exhibit Hot Hand belief (prefer near-miss numbers)
- Odds ratio of 0.87 overall (slight avoidance of near-miss numbers)
- Near-miss effects are strongest for wheel proximity vs table proximity
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod

# Import near-miss utilities
from ..utils.near_miss import (
    get_wheel_neighbors,
    get_table_neighbors,
    get_wheel_distance,
    is_near_miss_wheel,
    is_near_miss_table,
    compute_wheel_sector_frequencies,
    EUROPEAN_WHEEL_ORDER,
    WHEEL_POSITION
)

# Import environment constants
from ..environment.roulette_env import (
    RouletteEnv,
    RED_NUMBERS,
    BLACK_NUMBERS,
    ODD_NUMBERS,
    EVEN_NUMBERS,
    LOW_NUMBERS,
    HIGH_NUMBERS,
    FIRST_DOZEN,
    SECOND_DOZEN,
    THIRD_DOZEN
)


class BehaviorType(Enum):
    """Types of human behavioral biases."""
    GAMBLERS_FALLACY = "gamblers_fallacy"  # Avoid near-miss numbers (67% of players)
    HOT_HAND = "hot_hand"                   # Prefer near-miss numbers (33% of players)
    RANDOM = "random"                       # Random betting (baseline)
    MIXED = "mixed"                         # Randomly switches between behaviors


class BetType(Enum):
    """Types of bets the agent can make."""
    STRAIGHT = "straight"       # Single number (35:1)
    OUTSIDE_EVEN = "outside"    # Red/Black, Odd/Even, High/Low (1:1)
    DOZEN = "dozen"             # Dozens (2:1)
    PASS = "pass"               # No bet


class BehavioralAgent(ABC):
    """
    Base class for behavioral betting agents.
    
    These agents simulate human betting patterns based on
    psychological biases observed in real casino data.
    """
    
    def __init__(
        self,
        bet_type: BetType = BetType.STRAIGHT,
        pass_probability: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the behavioral agent.
        
        Args:
            bet_type: Type of bets to make
            pass_probability: Probability of passing (not betting)
            seed: Random seed for reproducibility
        """
        self.bet_type = bet_type
        self.pass_probability = pass_probability
        self.rng = np.random.RandomState(seed)
        
        # Statistics
        self.total_actions = 0
        self.bet_history = []
    
    @abstractmethod
    def select_action(
        self,
        history: List[int],
        bankroll: float,
        last_winning: Optional[int] = None
    ) -> int:
        """
        Select an action based on behavioral bias.
        
        Args:
            history: List of past winning numbers
            bankroll: Current bankroll
            last_winning: Last winning number (if any)
            
        Returns:
            Action index (0-46)
        """
        pass
    
    def _should_pass(self) -> bool:
        """Check if agent should pass this round."""
        return self.rng.random() < self.pass_probability
    
    def _select_outside_bet(self, preference: str = "random") -> int:
        """
        Select an outside bet action.
        
        Args:
            preference: Which type of outside bet to prefer
            
        Returns:
            Action index for outside bet
        """
        outside_actions = {
            "red": 37,
            "black": 38,
            "odd": 39,
            "even": 40,
            "low": 41,
            "high": 42,
            "dozen1": 43,
            "dozen2": 44,
            "dozen3": 45
        }
        
        if preference in outside_actions:
            return outside_actions[preference]
        
        # Random outside bet
        return self.rng.choice(list(outside_actions.values()))
    
    def reset(self):
        """Reset agent statistics."""
        self.total_actions = 0
        self.bet_history = []


class GamblersFallacyAgent(BehavioralAgent):
    """
    Agent that exhibits Gambler's Fallacy.
    
    This agent AVOIDS betting on numbers that are:
    1. Close to the last winning number on the wheel
    2. Close to the last winning number on the table
    
    Based on the finding that 67% of real players show this behavior.
    The agent believes that numbers that "almost won" are less likely
    to win in the future, which is statistically false.
    """
    
    def __init__(
        self,
        wheel_avoidance_distance: int = 3,
        include_table_avoidance: bool = True,
        avoidance_strength: float = 0.9,
        **kwargs
    ):
        """
        Initialize Gambler's Fallacy agent.
        
        Args:
            wheel_avoidance_distance: How far on wheel to avoid
            include_table_avoidance: Also avoid table-adjacent numbers
            avoidance_strength: Probability of avoiding near-miss (0-1)
            **kwargs: Arguments passed to parent
        """
        super().__init__(**kwargs)
        self.wheel_avoidance_distance = wheel_avoidance_distance
        self.include_table_avoidance = include_table_avoidance
        self.avoidance_strength = avoidance_strength
    
    def _get_avoided_numbers(self, last_winning: int) -> Set[int]:
        """Get set of numbers to avoid based on near-miss."""
        avoided = set()
        
        # Avoid wheel neighbors
        avoided.update(get_wheel_neighbors(last_winning, self.wheel_avoidance_distance))
        
        # Also avoid the last winning number itself (due to "it just hit" bias)
        avoided.add(last_winning)
        
        # Optionally avoid table neighbors
        if self.include_table_avoidance:
            avoided.update(get_table_neighbors(last_winning))
        
        return avoided
    
    def select_action(
        self,
        history: List[int],
        bankroll: float,
        last_winning: Optional[int] = None
    ) -> int:
        """Select action avoiding near-miss numbers."""
        self.total_actions += 1
        
        # Check if should pass
        if self._should_pass():
            self.bet_history.append(46)
            return 46
        
        # Get last winning number
        if last_winning is None and history:
            last_winning = history[-1]
        elif last_winning is None:
            # No history, bet randomly
            action = self.rng.randint(0, 37)
            self.bet_history.append(action)
            return action
        
        # Straight bets with avoidance
        if self.bet_type == BetType.STRAIGHT:
            avoided = self._get_avoided_numbers(last_winning)
            
            # With avoidance_strength probability, avoid near-miss numbers
            if self.rng.random() < self.avoidance_strength:
                available = [n for n in range(37) if n not in avoided]
                if available:
                    action = self.rng.choice(available)
                else:
                    action = self.rng.randint(0, 37)
            else:
                action = self.rng.randint(0, 37)
            
            self.bet_history.append(action)
            return action
        
        # Outside bets - avoid the "hot" side
        elif self.bet_type == BetType.OUTSIDE_EVEN:
            # If last was red, bet black (and vice versa)
            if last_winning in RED_NUMBERS:
                action = 38  # BLACK
            elif last_winning in BLACK_NUMBERS:
                action = 37  # RED
            else:  # Green
                action = self.rng.choice([37, 38])
            
            self.bet_history.append(action)
            return action
        
        # Dozens - avoid the "hot" dozen
        elif self.bet_type == BetType.DOZEN:
            if last_winning in FIRST_DOZEN:
                action = self.rng.choice([44, 45])  # 2nd or 3rd dozen
            elif last_winning in SECOND_DOZEN:
                action = self.rng.choice([43, 45])
            elif last_winning in THIRD_DOZEN:
                action = self.rng.choice([43, 44])
            else:  # Zero
                action = self.rng.choice([43, 44, 45])
            
            self.bet_history.append(action)
            return action
        
        # Default: random action
        action = self.rng.randint(0, 47)
        self.bet_history.append(action)
        return action


class HotHandAgent(BehavioralAgent):
    """
    Agent that exhibits Hot Hand belief.
    
    This agent PREFERS betting on numbers that are:
    1. Close to the last winning number on the wheel
    2. Close to the last winning number on the table
    
    Based on the finding that 33% of real players show this behavior.
    The agent believes that numbers that "almost won" are MORE likely
    to win in the future ("that section is hot"), which is also false.
    """
    
    def __init__(
        self,
        wheel_preference_distance: int = 3,
        include_table_preference: bool = True,
        preference_strength: float = 0.8,
        **kwargs
    ):
        """
        Initialize Hot Hand agent.
        
        Args:
            wheel_preference_distance: How far on wheel to prefer
            include_table_preference: Also prefer table-adjacent numbers
            preference_strength: Probability of preferring near-miss (0-1)
            **kwargs: Arguments passed to parent
        """
        super().__init__(**kwargs)
        self.wheel_preference_distance = wheel_preference_distance
        self.include_table_preference = include_table_preference
        self.preference_strength = preference_strength
    
    def _get_preferred_numbers(self, last_winning: int) -> Set[int]:
        """Get set of numbers to prefer based on near-miss."""
        preferred = set()
        
        # Prefer wheel neighbors (the "hot zone")
        preferred.update(get_wheel_neighbors(last_winning, self.wheel_preference_distance))
        
        # Can also bet on the same number again
        preferred.add(last_winning)
        
        # Optionally prefer table neighbors
        if self.include_table_preference:
            preferred.update(get_table_neighbors(last_winning))
        
        return preferred
    
    def select_action(
        self,
        history: List[int],
        bankroll: float,
        last_winning: Optional[int] = None
    ) -> int:
        """Select action preferring near-miss numbers."""
        self.total_actions += 1
        
        if self._should_pass():
            self.bet_history.append(46)
            return 46
        
        if last_winning is None and history:
            last_winning = history[-1]
        elif last_winning is None:
            action = self.rng.randint(0, 37)
            self.bet_history.append(action)
            return action
        
        if self.bet_type == BetType.STRAIGHT:
            preferred = self._get_preferred_numbers(last_winning)
            
            if self.rng.random() < self.preference_strength and preferred:
                action = self.rng.choice(list(preferred))
            else:
                action = self.rng.randint(0, 37)
            
            self.bet_history.append(action)
            return action
        
        elif self.bet_type == BetType.OUTSIDE_EVEN:
            # Bet on the same color as last (the "hot" color)
            if last_winning in RED_NUMBERS:
                action = 37  # RED
            elif last_winning in BLACK_NUMBERS:
                action = 38  # BLACK
            else:
                action = self.rng.choice([37, 38])
            
            self.bet_history.append(action)
            return action
        
        elif self.bet_type == BetType.DOZEN:
            # Bet on the same dozen as last
            if last_winning in FIRST_DOZEN:
                action = 43
            elif last_winning in SECOND_DOZEN:
                action = 44
            elif last_winning in THIRD_DOZEN:
                action = 45
            else:
                action = self.rng.choice([43, 44, 45])
            
            self.bet_history.append(action)
            return action
        
        action = self.rng.randint(0, 47)
        self.bet_history.append(action)
        return action


class RandomAgent(BehavioralAgent):
    """
    Baseline random agent for comparison.
    
    Makes uniformly random bets with no behavioral bias.
    """
    
    def select_action(
        self,
        history: List[int],
        bankroll: float,
        last_winning: Optional[int] = None
    ) -> int:
        """Select a random action."""
        self.total_actions += 1
        
        if self._should_pass():
            self.bet_history.append(46)
            return 46
        
        if self.bet_type == BetType.STRAIGHT:
            action = self.rng.randint(0, 37)
        elif self.bet_type == BetType.OUTSIDE_EVEN:
            action = self.rng.choice([37, 38, 39, 40, 41, 42])
        elif self.bet_type == BetType.DOZEN:
            action = self.rng.choice([43, 44, 45])
        else:
            action = self.rng.randint(0, 47)
        
        self.bet_history.append(action)
        return action


class MixedBehaviorAgent(BehavioralAgent):
    """
    Agent that switches between Gambler's Fallacy and Hot Hand.
    
    Based on the paper finding that players show mixed behaviors:
    - 67% Gambler's Fallacy
    - 33% Hot Hand
    
    This agent randomly selects behavior each round according to
    these probabilities.
    """
    
    def __init__(
        self,
        fallacy_probability: float = 0.67,
        **kwargs
    ):
        """
        Initialize mixed behavior agent.
        
        Args:
            fallacy_probability: Probability of Gambler's Fallacy (default 0.67)
            **kwargs: Arguments passed to sub-agents
        """
        super().__init__(**kwargs)
        self.fallacy_probability = fallacy_probability
        
        # Create sub-agents
        self.fallacy_agent = GamblersFallacyAgent(**kwargs)
        self.hot_hand_agent = HotHandAgent(**kwargs)
        
        # Track behavior selection
        self.behavior_history = []
    
    def select_action(
        self,
        history: List[int],
        bankroll: float,
        last_winning: Optional[int] = None
    ) -> int:
        """Select action based on randomly chosen behavior."""
        self.total_actions += 1
        
        if self._should_pass():
            self.bet_history.append(46)
            self.behavior_history.append("pass")
            return 46
        
        # Choose behavior probabilistically
        if self.rng.random() < self.fallacy_probability:
            action = self.fallacy_agent.select_action(history, bankroll, last_winning)
            self.behavior_history.append("fallacy")
        else:
            action = self.hot_hand_agent.select_action(history, bankroll, last_winning)
            self.behavior_history.append("hot_hand")
        
        self.bet_history.append(action)
        return action
    
    def reset(self):
        """Reset all agents."""
        super().reset()
        self.fallacy_agent.reset()
        self.hot_hand_agent.reset()
        self.behavior_history = []


class SectorBiasAgent(BehavioralAgent):
    """
    Agent that bets based on observed wheel sector frequencies.
    
    This agent looks for "bias" in the wheel by tracking which
    sectors have been hitting more frequently and betting accordingly.
    
    Note: This represents a more "rational" bias-seeking behavior
    that some players exhibit, looking for physical wheel defects.
    """
    
    def __init__(
        self,
        n_sectors: int = 8,
        lookback_window: int = 50,
        bias_threshold: float = 0.15,  # 15% above expected
        **kwargs
    ):
        """
        Initialize sector bias agent.
        
        Args:
            n_sectors: Number of wheel sectors to track
            lookback_window: How many spins to consider
            bias_threshold: Minimum excess frequency to consider "biased"
            **kwargs: Arguments passed to parent
        """
        super().__init__(**kwargs)
        self.n_sectors = n_sectors
        self.lookback_window = lookback_window
        self.bias_threshold = bias_threshold
        self.sector_size = 37 / n_sectors
    
    def _get_sector_numbers(self, sector_idx: int) -> List[int]:
        """Get numbers in a wheel sector."""
        start = int(sector_idx * self.sector_size)
        end = int((sector_idx + 1) * self.sector_size)
        return [EUROPEAN_WHEEL_ORDER[i % 37] for i in range(start, end)]
    
    def _find_hot_sector(self, history: List[int]) -> Optional[int]:
        """Find the most biased sector, if any."""
        recent = history[-self.lookback_window:] if len(history) >= self.lookback_window else history
        
        if len(recent) < 10:
            return None
        
        frequencies = compute_wheel_sector_frequencies(recent, self.n_sectors)
        expected = 1.0 / self.n_sectors
        
        # Find sector with highest excess frequency
        excesses = frequencies - expected
        max_excess_idx = np.argmax(excesses)
        
        if excesses[max_excess_idx] > self.bias_threshold * expected:
            return max_excess_idx
        
        return None
    
    def select_action(
        self,
        history: List[int],
        bankroll: float,
        last_winning: Optional[int] = None
    ) -> int:
        """Select action targeting biased wheel sector."""
        self.total_actions += 1
        
        if self._should_pass():
            self.bet_history.append(46)
            return 46
        
        hot_sector = self._find_hot_sector(history)
        
        if hot_sector is not None and self.bet_type == BetType.STRAIGHT:
            # Bet on a random number in the hot sector
            sector_numbers = self._get_sector_numbers(hot_sector)
            action = self.rng.choice(sector_numbers)
            self.bet_history.append(action)
            return action
        
        # No bias detected or not making straight bets - bet randomly
        if self.bet_type == BetType.STRAIGHT:
            action = self.rng.randint(0, 37)
        else:
            action = self._select_outside_bet()
        
        self.bet_history.append(action)
        return action


def create_agent(
    behavior: str,
    bet_type: str = "straight",
    **kwargs
) -> BehavioralAgent:
    """
    Factory function to create behavioral agents.
    
    Args:
        behavior: Agent behavior type ("fallacy", "hot_hand", "random", "mixed", "sector")
        bet_type: Type of bets ("straight", "outside", "dozen")
        **kwargs: Additional arguments for the agent
        
    Returns:
        Configured BehavioralAgent instance
    """
    bet_type_enum = {
        "straight": BetType.STRAIGHT,
        "outside": BetType.OUTSIDE_EVEN,
        "dozen": BetType.DOZEN,
        "pass": BetType.PASS
    }.get(bet_type, BetType.STRAIGHT)
    
    agents = {
        "fallacy": GamblersFallacyAgent,
        "gamblers_fallacy": GamblersFallacyAgent,
        "hot_hand": HotHandAgent,
        "hot": HotHandAgent,
        "random": RandomAgent,
        "mixed": MixedBehaviorAgent,
        "sector": SectorBiasAgent,
        "sector_bias": SectorBiasAgent
    }
    
    agent_class = agents.get(behavior.lower(), RandomAgent)
    return agent_class(bet_type=bet_type_enum, **kwargs)


def compare_behavioral_agents(
    env: RouletteEnv,
    n_episodes: int = 100,
    seed: Optional[int] = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compare different behavioral agents on the same environment.
    
    Args:
        env: Roulette environment
        n_episodes: Number of episodes to run
        seed: Random seed
        
    Returns:
        Dictionary of results for each agent type
    """
    agents = {
        "Random": RandomAgent(bet_type=BetType.STRAIGHT, seed=seed),
        "Gambler's Fallacy": GamblersFallacyAgent(bet_type=BetType.STRAIGHT, seed=seed),
        "Hot Hand": HotHandAgent(bet_type=BetType.STRAIGHT, seed=seed),
        "Mixed (67/33)": MixedBehaviorAgent(bet_type=BetType.STRAIGHT, seed=seed),
        "Sector Bias": SectorBiasAgent(bet_type=BetType.STRAIGHT, seed=seed)
    }
    
    results = {}
    
    for name, agent in agents.items():
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep if seed else None)
            agent.reset()
            
            total_reward = 0
            done = False
            steps = 0
            history = list(obs.get("history", obs[:20] if isinstance(obs, np.ndarray) else []))
            
            while not done:
                last_num = history[-1] if history else None
                action = agent.select_action(history, info.get("bankroll", 1000), last_num)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                
                # Update history
                if "history" in obs if isinstance(obs, dict) else False:
                    history = list(obs["history"])
                elif "winning_number" in info:
                    history.append(info["winning_number"])
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        results[name] = {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "avg_length": np.mean(episode_lengths),
            "win_rate": np.mean([r > 0 for r in episode_rewards])
        }
    
    return results


if __name__ == "__main__":
    print("=== Behavioral Agents Test ===\n")
    
    # Create environment
    env = RouletteEnv(max_steps=80, initial_bankroll=1000, bet_size=50)
    
    # Test each agent type
    agents = [
        ("Random", RandomAgent(bet_type=BetType.STRAIGHT, seed=42)),
        ("Gambler's Fallacy", GamblersFallacyAgent(bet_type=BetType.STRAIGHT, seed=42)),
        ("Hot Hand", HotHandAgent(bet_type=BetType.STRAIGHT, seed=42)),
        ("Mixed", MixedBehaviorAgent(bet_type=BetType.STRAIGHT, seed=42)),
        ("Sector Bias", SectorBiasAgent(bet_type=BetType.STRAIGHT, seed=42))
    ]
    
    for name, agent in agents:
        print(f"--- {name} Agent ---")
        
        obs, info = env.reset(seed=42)
        history = list(obs["history"])
        
        # Run 10 steps
        total_reward = 0
        for i in range(10):
            action = agent.select_action(history, info["bankroll"], history[-1] if history else None)
            obs, reward, term, trunc, info = env.step(action)
            history.append(info["winning_number"])
            total_reward += reward
            
            if i < 3:
                print(f"  Step {i+1}: Action={action:2d}, Won={info['winning_number']:2d}, Reward={reward:+.0f}")
        
        print(f"  Total reward (10 steps): ${total_reward:+.0f}")
        print()
    
    # Full comparison
    print("\n=== Full Agent Comparison (100 episodes) ===\n")
    results = compare_behavioral_agents(env, n_episodes=100, seed=42)
    
    for name, stats in results.items():
        print(f"{name:20s}: Avg=${stats['avg_reward']:+7.1f} ± ${stats['std_reward']:6.1f}, "
              f"Win%={stats['win_rate']*100:5.1f}%, MaxReward=${stats['max_reward']:+.0f}")
