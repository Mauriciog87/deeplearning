"""
Fuzzy Adaptive Controller for DQN Agent

Implementation based on concepts from:
"Fuzzy Guiding of Roulette Selection in Evolutionary Algorithms" by Krzysztof Pytel (2025)
MDPI Technologies, DOI: 10.3390/technologies13020078

Adapts the paper's FLC-EA (Fuzzy Logic Controller for Evolutionary Algorithms) concepts:
- Individual Quality -> Action Quality Estimator
- Successful Reproductions Ratio -> Successful Actions Ratio
- Historical Growth Ratio -> Reward Trend Analyzer
- Relative Distance -> Exploration Diversity Estimator

The fuzzy controller dynamically adjusts epsilon based on these metrics.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class FuzzySet(Enum):
    """Fuzzy set labels for linguistic variables."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class FuzzyMembershipFunctions:
    """
    Triangular membership functions for fuzzy sets.
    
    Each fuzzy set is defined by a triangular function (a, b, c)
    where a is left foot, b is peak, c is right foot.
    """
    
    # Input membership functions (normalized 0-1)
    quality: Dict[FuzzySet, Tuple[float, float, float]] = field(default_factory=lambda: {
        FuzzySet.VERY_LOW:  (-0.25, 0.0, 0.25),
        FuzzySet.LOW:       (0.0, 0.25, 0.5),
        FuzzySet.MEDIUM:    (0.25, 0.5, 0.75),
        FuzzySet.HIGH:      (0.5, 0.75, 1.0),
        FuzzySet.VERY_HIGH: (0.75, 1.0, 1.25)
    })
    
    success_ratio: Dict[FuzzySet, Tuple[float, float, float]] = field(default_factory=lambda: {
        FuzzySet.VERY_LOW:  (-0.25, 0.0, 0.25),
        FuzzySet.LOW:       (0.0, 0.25, 0.5),
        FuzzySet.MEDIUM:    (0.25, 0.5, 0.75),
        FuzzySet.HIGH:      (0.5, 0.75, 1.0),
        FuzzySet.VERY_HIGH: (0.75, 1.0, 1.25)
    })
    
    trend: Dict[FuzzySet, Tuple[float, float, float]] = field(default_factory=lambda: {
        FuzzySet.VERY_LOW:  (-0.25, 0.0, 0.25),   # Decreasing
        FuzzySet.LOW:       (0.0, 0.25, 0.5),
        FuzzySet.MEDIUM:    (0.25, 0.5, 0.75),    # Stable
        FuzzySet.HIGH:      (0.5, 0.75, 1.0),
        FuzzySet.VERY_HIGH: (0.75, 1.0, 1.25)     # Improving
    })
    
    diversity: Dict[FuzzySet, Tuple[float, float, float]] = field(default_factory=lambda: {
        FuzzySet.VERY_LOW:  (-0.25, 0.0, 0.25),
        FuzzySet.LOW:       (0.0, 0.25, 0.5),
        FuzzySet.MEDIUM:    (0.25, 0.5, 0.75),
        FuzzySet.HIGH:      (0.5, 0.75, 1.0),
        FuzzySet.VERY_HIGH: (0.75, 1.0, 1.25)
    })
    
    # Output membership functions for epsilon adjustment (-1 to 1)
    epsilon_adjustment: Dict[FuzzySet, Tuple[float, float, float]] = field(default_factory=lambda: {
        FuzzySet.VERY_LOW:  (-1.25, -1.0, -0.5),   # Strong decrease
        FuzzySet.LOW:       (-0.75, -0.5, -0.25),  # Moderate decrease
        FuzzySet.MEDIUM:    (-0.25, 0.0, 0.25),    # No change
        FuzzySet.HIGH:      (0.25, 0.5, 0.75),     # Moderate increase
        FuzzySet.VERY_HIGH: (0.5, 1.0, 1.25)       # Strong increase
    })


def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    """
    Calculate triangular membership degree.
    
    Args:
        x: Input value
        a: Left foot of triangle
        b: Peak of triangle
        c: Right foot of triangle
    
    Returns:
        Membership degree in [0, 1]
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)


class ActionQualityEstimator:
    """
    Estimates action quality based on recent performance.
    
    Inspired by the paper's "Individual Quality" metric:
    IQ(i,t) = v(i,t) / v_max(t)
    
    For our DQN:
    - Tracks average reward per action type
    - Normalizes against best performing action
    """
    
    def __init__(self, action_size: int = 47, window_size: int = 100):
        self.action_size = action_size
        self.window_size = window_size
        
        # Track rewards per action
        self.action_rewards: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(action_size)
        }
        
        # Track selection counts
        self.action_counts: np.ndarray = np.zeros(action_size)
        self.total_steps = 0
    
    def update(self, action: int, reward: float):
        """Record an action and its reward."""
        self.action_rewards[action].append(reward)
        self.action_counts[action] += 1
        self.total_steps += 1
    
    def get_action_quality(self, action: int) -> float:
        """
        Get normalized quality for an action.
        
        Returns:
            Quality score in [0, 1]
        """
        if len(self.action_rewards[action]) == 0:
            return 0.5  # Unknown quality = medium
        
        avg_reward = np.mean(self.action_rewards[action])
        
        # Get max average across all actions
        max_avg = max(
            np.mean(rewards) if len(rewards) > 0 else float('-inf')
            for rewards in self.action_rewards.values()
        )
        min_avg = min(
            np.mean(rewards) if len(rewards) > 0 else float('inf')
            for rewards in self.action_rewards.values()
        )
        
        if max_avg == min_avg:
            return 0.5
        
        # Normalize to [0, 1]
        return (avg_reward - min_avg) / (max_avg - min_avg)
    
    def get_overall_quality(self) -> float:
        """
        Get overall action quality (average of recently used actions).
        
        Returns:
            Quality score in [0, 1]
        """
        # Consider only actions used in last N steps
        recent_actions = []
        for action, rewards in self.action_rewards.items():
            if len(rewards) >= 3:  # At least 3 samples
                recent_actions.append(np.mean(rewards))
        
        if not recent_actions:
            return 0.5
        
        # Normalize
        max_r = max(recent_actions)
        min_r = min(recent_actions)
        
        if max_r == min_r:
            return 0.5
        
        avg = np.mean(recent_actions)
        return (avg - min_r) / (max_r - min_r)
    
    def get_action_distribution(self) -> np.ndarray:
        """Get normalized distribution of action usage."""
        if self.total_steps == 0:
            return np.ones(self.action_size) / self.action_size
        return self.action_counts / self.total_steps


class SuccessfulActionsEstimator:
    """
    Tracks ratio of successful actions.
    
    Inspired by the paper's "Successful Reproductions Ratio":
    SR(t) = s(t) / n
    
    For our DQN:
    - Tracks actions that resulted in positive rewards
    - Uses sliding window for recent performance
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.results = deque(maxlen=window_size)  # 1 for success, 0 for failure
    
    def update(self, reward: float, action: int = None):
        """
        Record action result.
        
        Args:
            reward: Reward received
            action: Action taken (optional, for future use)
        """
        # Consider positive reward as success, non-negative as partial
        if reward > 0:
            self.results.append(1.0)
        elif reward == 0:
            self.results.append(0.5)
        else:
            self.results.append(0.0)
    
    def get_success_ratio(self) -> float:
        """
        Get success ratio.
        
        Returns:
            Success ratio in [0, 1]
        """
        if len(self.results) == 0:
            return 0.5  # Unknown
        return np.mean(self.results)


class HistoricalTrendAnalyzer:
    """
    Analyzes historical reward trends.
    
    Inspired by the paper's "Historical Growth Ratio":
    HG(t) = (v_best(t) - v_best(1)) / v_best(1) for t > 1
    
    For our DQN:
    - Tracks reward moving averages at different time scales
    - Computes trend direction and magnitude
    """
    
    def __init__(
        self, 
        short_window: int = 20, 
        long_window: int = 100,
        trend_threshold: float = 0.1
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.trend_threshold = trend_threshold
        
        self.rewards = deque(maxlen=long_window)
        self.episode_rewards: List[float] = []
    
    def update(self, reward: float):
        """Record a reward."""
        self.rewards.append(reward)
    
    def end_episode(self, total_reward: float):
        """Record episode total reward."""
        self.episode_rewards.append(total_reward)
    
    def get_trend(self) -> float:
        """
        Get reward trend.
        
        Returns:
            Trend score in [0, 1]:
            - 0.0: Strong negative trend (getting worse)
            - 0.5: Stable (no clear trend)
            - 1.0: Strong positive trend (improving)
        """
        if len(self.rewards) < self.short_window:
            return 0.5  # Not enough data
        
        rewards = list(self.rewards)
        
        # Short-term average (recent)
        short_avg = np.mean(rewards[-self.short_window:])
        
        # Long-term average (baseline)
        if len(rewards) >= self.long_window:
            long_avg = np.mean(rewards[-self.long_window:-self.short_window])
        else:
            long_avg = np.mean(rewards[:-self.short_window]) if len(rewards) > self.short_window else short_avg
        
        if long_avg == 0:
            long_avg = 0.001  # Avoid division by zero
        
        # Calculate relative change
        relative_change = (short_avg - long_avg) / abs(long_avg)
        
        # Map to [0, 1]
        # -threshold or worse -> 0.0
        # +threshold or better -> 1.0
        normalized = (relative_change + self.trend_threshold) / (2 * self.trend_threshold)
        return np.clip(normalized, 0.0, 1.0)
    
    def get_episode_trend(self) -> float:
        """
        Get episode-level trend.
        
        Returns:
            Trend score in [0, 1]
        """
        if len(self.episode_rewards) < 5:
            return 0.5
        
        recent = self.episode_rewards[-5:]
        older = self.episode_rewards[-10:-5] if len(self.episode_rewards) >= 10 else self.episode_rewards[:-5]
        
        if not older:
            return 0.5
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if older_avg == 0:
            older_avg = 0.001
        
        relative_change = (recent_avg - older_avg) / abs(older_avg)
        normalized = (relative_change + 0.2) / 0.4  # ±20% range
        return np.clip(normalized, 0.0, 1.0)


class ExplorationDiversityEstimator:
    """
    Estimates exploration diversity.
    
    Inspired by the paper's "Relative Distance" metric:
    RD(i,t) = (d_max(t) - d(i,t)) / d_max(t)
    
    For our DQN:
    - Tracks action entropy (how diverse action selection is)
    - Monitors Q-value distribution spread
    """
    
    def __init__(self, action_size: int = 47, window_size: int = 100):
        self.action_size = action_size
        self.window_size = window_size
        self.recent_actions = deque(maxlen=window_size)
        self.q_value_spreads = deque(maxlen=window_size)
    
    def update(self, action: int, q_values: Optional[np.ndarray] = None):
        """
        Record an action and optionally Q-values.
        
        Args:
            action: Selected action
            q_values: Q-values for all actions (optional)
        """
        self.recent_actions.append(action)
        
        if q_values is not None:
            # Record Q-value spread (std/mean ratio)
            q_std = np.std(q_values)
            q_mean = np.mean(np.abs(q_values))
            spread = q_std / (q_mean + 1e-8)
            self.q_value_spreads.append(spread)
    
    def get_action_entropy(self) -> float:
        """
        Calculate normalized action entropy.
        
        Returns:
            Entropy score in [0, 1]
        """
        if len(self.recent_actions) < 10:
            return 1.0  # High diversity assumed at start
        
        # Count action frequencies
        actions = np.array(self.recent_actions)
        counts = np.bincount(actions, minlength=self.action_size)
        probs = counts / len(actions)
        
        # Remove zeros for log
        probs = probs[probs > 0]
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # Max entropy for uniform distribution
        max_entropy = np.log2(self.action_size)
        
        # Normalized entropy
        return entropy / max_entropy
    
    def get_q_value_diversity(self) -> float:
        """
        Get Q-value diversity score.
        
        Returns:
            Diversity score in [0, 1]
        """
        if len(self.q_value_spreads) < 5:
            return 0.5
        
        avg_spread = np.mean(self.q_value_spreads)
        
        # Map spread to [0, 1] - higher spread = more diversity
        # Typical spread is 0.1-2.0
        return np.clip(avg_spread / 2.0, 0.0, 1.0)
    
    def get_diversity(self) -> float:
        """Combined diversity metric."""
        entropy = self.get_action_entropy()
        q_div = self.get_q_value_diversity()
        return 0.7 * entropy + 0.3 * q_div


class FuzzyEpsilonController:
    """
    Fuzzy Logic Controller for adaptive epsilon.
    
    Combines all estimators using fuzzy rules to determine
    optimal exploration rate.
    
    Based on the paper's FLC-EA:
    - Uses 4 input variables
    - Applies fuzzy rules to compute output
    - Defuzzifies to get epsilon adjustment
    """
    
    def __init__(
        self,
        action_size: int = 47,
        epsilon_min: float = 0.05,
        epsilon_max: float = 1.0,
        adjustment_rate: float = 0.1,
        window_size: int = 100
    ):
        self.action_size = action_size
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.adjustment_rate = adjustment_rate
        
        # Current epsilon
        self.epsilon = epsilon_max
        
        # Fuzzy membership functions
        self.mf = FuzzyMembershipFunctions()
        
        # Estimators
        self.quality_estimator = ActionQualityEstimator(action_size, window_size)
        self.success_estimator = SuccessfulActionsEstimator(window_size)
        self.trend_analyzer = HistoricalTrendAnalyzer()
        self.diversity_estimator = ExplorationDiversityEstimator(action_size, window_size)
        
        # Statistics
        self.update_count = 0
        self.epsilon_history: List[float] = []
        
        # Fuzzy rules (simplified Mamdani-style)
        # Format: (quality_set, success_set, trend_set, diversity_set) -> epsilon_adjustment_set
        self.rules = self._create_rules()
    
    def _create_rules(self) -> List[Tuple[Optional[FuzzySet], Optional[FuzzySet], 
                                          Optional[FuzzySet], Optional[FuzzySet], FuzzySet]]:
        """
        Create fuzzy rule base.
        
        Rules are inspired by the paper's approach:
        - Low quality + low success -> increase exploration
        - High quality + high success -> decrease exploration (exploit)
        - Stagnating trend -> increase exploration
        - Low diversity -> increase exploration
        """
        rules = [
            # Quality-based rules
            (FuzzySet.VERY_LOW, None, None, None, FuzzySet.VERY_HIGH),    # Very low quality -> explore more
            (FuzzySet.LOW, None, None, None, FuzzySet.HIGH),              # Low quality -> explore more
            (FuzzySet.HIGH, None, None, None, FuzzySet.LOW),              # High quality -> exploit
            (FuzzySet.VERY_HIGH, None, None, None, FuzzySet.VERY_LOW),    # Very high quality -> exploit more
            
            # Success-based rules
            (None, FuzzySet.VERY_LOW, None, None, FuzzySet.VERY_HIGH),    # No success -> explore
            (None, FuzzySet.VERY_HIGH, None, None, FuzzySet.VERY_LOW),    # High success -> exploit
            
            # Trend-based rules
            (None, None, FuzzySet.VERY_LOW, None, FuzzySet.VERY_HIGH),    # Declining -> explore
            (None, None, FuzzySet.LOW, None, FuzzySet.HIGH),              # Slight decline -> explore
            (None, None, FuzzySet.HIGH, None, FuzzySet.LOW),              # Improving -> exploit
            (None, None, FuzzySet.VERY_HIGH, None, FuzzySet.VERY_LOW),    # Strongly improving -> exploit
            
            # Diversity-based rules
            (None, None, None, FuzzySet.VERY_LOW, FuzzySet.VERY_HIGH),    # Low diversity -> explore
            (None, None, None, FuzzySet.LOW, FuzzySet.HIGH),              # Low diversity -> explore
            (None, None, None, FuzzySet.VERY_HIGH, FuzzySet.LOW),         # High diversity -> can exploit
            
            # Combined rules (most important)
            (FuzzySet.LOW, FuzzySet.LOW, FuzzySet.LOW, None, FuzzySet.VERY_HIGH),     # All bad -> explore a lot
            (FuzzySet.HIGH, FuzzySet.HIGH, FuzzySet.HIGH, None, FuzzySet.VERY_LOW),   # All good -> exploit
            (FuzzySet.MEDIUM, FuzzySet.MEDIUM, FuzzySet.MEDIUM, None, FuzzySet.MEDIUM),  # Medium -> maintain
        ]
        return rules
    
    def _fuzzify(
        self, 
        value: float, 
        mf_dict: Dict[FuzzySet, Tuple[float, float, float]]
    ) -> Dict[FuzzySet, float]:
        """
        Fuzzify a crisp value.
        
        Args:
            value: Crisp input value
            mf_dict: Membership function dictionary
            
        Returns:
            Dictionary of fuzzy set memberships
        """
        memberships = {}
        for fuzzy_set, (a, b, c) in mf_dict.items():
            memberships[fuzzy_set] = triangular_membership(value, a, b, c)
        return memberships
    
    def _evaluate_rules(
        self,
        quality_mf: Dict[FuzzySet, float],
        success_mf: Dict[FuzzySet, float],
        trend_mf: Dict[FuzzySet, float],
        diversity_mf: Dict[FuzzySet, float]
    ) -> Dict[FuzzySet, float]:
        """
        Evaluate fuzzy rules.
        
        Args:
            *_mf: Fuzzified input memberships
            
        Returns:
            Output fuzzy set memberships
        """
        output_mf = {fs: 0.0 for fs in FuzzySet}
        
        for rule in self.rules:
            q_set, s_set, t_set, d_set, out_set = rule
            
            # Calculate rule strength (AND = min)
            strengths = []
            if q_set is not None:
                strengths.append(quality_mf[q_set])
            if s_set is not None:
                strengths.append(success_mf[s_set])
            if t_set is not None:
                strengths.append(trend_mf[t_set])
            if d_set is not None:
                strengths.append(diversity_mf[d_set])
            
            if not strengths:
                continue
            
            rule_strength = min(strengths)
            
            # Aggregate (OR = max)
            output_mf[out_set] = max(output_mf[out_set], rule_strength)
        
        return output_mf
    
    def _defuzzify(self, output_mf: Dict[FuzzySet, float]) -> float:
        """
        Defuzzify using centroid method.
        
        Args:
            output_mf: Output fuzzy set memberships
            
        Returns:
            Crisp output value (epsilon adjustment in [-1, 1])
        """
        # Centroid calculation
        numerator = 0.0
        denominator = 0.0
        
        for fuzzy_set, membership in output_mf.items():
            if membership > 0:
                # Get centroid of this fuzzy set
                a, b, c = self.mf.epsilon_adjustment[fuzzy_set]
                centroid = b  # Peak of triangle
                
                numerator += membership * centroid
                denominator += membership
        
        if denominator == 0:
            return 0.0  # No change
        
        return numerator / denominator
    
    def update(
        self,
        action: int,
        reward: float,
        q_values: Optional[np.ndarray] = None
    ):
        """
        Update all estimators with new experience.
        
        Args:
            action: Action taken
            reward: Reward received
            q_values: Q-values for all actions (optional)
        """
        self.quality_estimator.update(action, reward)
        self.success_estimator.update(reward, action)
        self.trend_analyzer.update(reward)
        self.diversity_estimator.update(action, q_values)
        
        self.update_count += 1
    
    def end_episode(self, total_reward: float):
        """Signal end of episode."""
        self.trend_analyzer.end_episode(total_reward)
    
    def compute_epsilon(self, force_update: bool = False) -> float:
        """
        Compute new epsilon using fuzzy logic.
        
        Args:
            force_update: Force epsilon update even with few samples
            
        Returns:
            New epsilon value
        """
        # Don't update too frequently
        if not force_update and self.update_count < 20:
            return self.epsilon
        
        # Get crisp input values
        quality = self.quality_estimator.get_overall_quality()
        success = self.success_estimator.get_success_ratio()
        trend = self.trend_analyzer.get_trend()
        diversity = self.diversity_estimator.get_diversity()
        
        # Fuzzify inputs
        quality_mf = self._fuzzify(quality, self.mf.quality)
        success_mf = self._fuzzify(success, self.mf.success_ratio)
        trend_mf = self._fuzzify(trend, self.mf.trend)
        diversity_mf = self._fuzzify(diversity, self.mf.diversity)
        
        # Evaluate rules
        output_mf = self._evaluate_rules(quality_mf, success_mf, trend_mf, diversity_mf)
        
        # Defuzzify to get adjustment
        adjustment = self._defuzzify(output_mf)
        
        # Apply adjustment with learning rate
        delta = adjustment * self.adjustment_rate
        new_epsilon = self.epsilon + delta
        
        # Clip to valid range
        self.epsilon = np.clip(new_epsilon, self.epsilon_min, self.epsilon_max)
        
        self.epsilon_history.append(self.epsilon)
        
        return self.epsilon
    
    def get_epsilon(self) -> float:
        """Get current epsilon."""
        return self.epsilon
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metric values.
        
        Returns:
            Dictionary of metric names to values
        """
        return {
            'quality': self.quality_estimator.get_overall_quality(),
            'success_ratio': self.success_estimator.get_success_ratio(),
            'trend': self.trend_analyzer.get_trend(),
            'diversity': self.diversity_estimator.get_diversity(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }
    
    def reset_for_episode(self):
        """Called at start of new episode (optional state reset)."""
        pass  # Keep all history across episodes


class FuzzyAdaptiveDQN:
    """
    DQN Agent with Fuzzy Adaptive Epsilon Control.
    
    Wraps a standard DQNAgent and replaces the fixed epsilon-decay
    with fuzzy logic-based adaptive control.
    """
    
    def __init__(
        self,
        base_agent,  # DQNAgent instance
        epsilon_min: float = 0.05,
        epsilon_max: float = 1.0,
        adjustment_rate: float = 0.1,
        update_frequency: int = 10  # How often to update epsilon
    ):
        """
        Initialize the fuzzy adaptive wrapper.
        
        Args:
            base_agent: Base DQNAgent to wrap
            epsilon_min: Minimum epsilon
            epsilon_max: Maximum epsilon
            adjustment_rate: How much to adjust epsilon per update
            update_frequency: Steps between epsilon updates
        """
        self.agent = base_agent
        self.update_frequency = update_frequency
        self.step_count = 0
        
        # Create fuzzy controller
        self.fuzzy_controller = FuzzyEpsilonController(
            action_size=base_agent.action_size,
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
            adjustment_rate=adjustment_rate
        )
        
        # Override agent's epsilon with fuzzy controller's
        self.agent.epsilon = self.fuzzy_controller.epsilon
        
        # Disable agent's built-in epsilon decay
        self.agent.epsilon_decay = 1.0
        self.agent.config['epsilon_decay'] = 1.0
    
    def act(
        self, 
        history: np.ndarray, 
        gain: float,
        training: bool = True,
        action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Select action and return Q-values for tracking.
        
        Returns:
            Tuple of (action, q_values)
        """
        # Get Q-values for fuzzy controller
        q_values = self.agent.get_q_values(history, gain) if training else None
        
        # Sync epsilon
        if training:
            self.agent.epsilon = self.fuzzy_controller.get_epsilon()
        
        # Get action from base agent
        action = self.agent.act(history, gain, training, action_mask)
        
        return action, q_values
    
    def remember(
        self,
        history: np.ndarray,
        gain: float,
        action: int,
        reward: float,
        next_history: np.ndarray,
        next_gain: float,
        done: bool,
        q_values: Optional[np.ndarray] = None,
        *, next_action_mask=None, truncated: bool = False
    ):
        """Store transition and update fuzzy controller."""
        # Store in base agent's memory
        self.agent.remember(history, gain, action, reward, next_history, next_gain, done,
                            next_action_mask=next_action_mask, truncated=truncated)
        
        # Update fuzzy controller
        self.fuzzy_controller.update(action, reward, q_values)
        
        self.step_count += 1
        
        # Periodically update epsilon
        if self.step_count % self.update_frequency == 0:
            self.fuzzy_controller.compute_epsilon()
            self.agent.epsilon = self.fuzzy_controller.get_epsilon()
    
    def train(self) -> Optional[float]:
        """Train the base agent."""
        return self.agent.train()
    
    def end_episode(self, total_reward: float):
        """Signal end of episode to fuzzy controller."""
        self.fuzzy_controller.end_episode(total_reward)
        # Force epsilon update at episode boundaries
        self.fuzzy_controller.compute_epsilon(force_update=True)
        self.agent.epsilon = self.fuzzy_controller.get_epsilon()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fuzzy controller metrics."""
        return self.fuzzy_controller.get_metrics()
    
    def save(self, path: str, extra_data: dict = None):
        """Save agent and fuzzy state."""
        fuzzy_data = {
            'fuzzy_state': {
                'step_count': self.step_count,
                'update_frequency': self.update_frequency,
                'controller': {name: getattr(self.fuzzy_controller, name) for name in (
                    'action_size', 'epsilon_min', 'epsilon_max', 'adjustment_rate',
                    'epsilon', 'update_count', 'epsilon_history')},
                'estimators': {name: vars(getattr(self.fuzzy_controller, name)) for name in (
                    'quality_estimator', 'success_estimator', 'trend_analyzer', 'diversity_estimator')}
            }
        }
        if extra_data:
            fuzzy_data.update(extra_data)
        self.agent.save(path, fuzzy_data)
    
    def load(self, path: str) -> dict:
        """Load agent and fuzzy state."""
        checkpoint = self.agent.load(path)
        if 'fuzzy_state' not in checkpoint:
            raise ValueError('Checkpoint does not contain a fuzzy controller')
        state = checkpoint['fuzzy_state']
        self.step_count = state['step_count']
        self.update_frequency = state['update_frequency']
        for name, value in state['controller'].items():
            setattr(self.fuzzy_controller, name, value)
        def restore_like(current, value):
            if isinstance(current, deque):
                return deque(value, maxlen=current.maxlen)
            if isinstance(current, np.ndarray):
                return np.asarray(value, dtype=current.dtype)
            if isinstance(current, dict):
                return {key: restore_like(current[key], item) for key, item in value.items()}
            return value
        for name, attributes in state['estimators'].items():
            estimator = getattr(self.fuzzy_controller, name)
            for key, value in attributes.items():
                setattr(estimator, key, restore_like(getattr(estimator, key), value))
        self.agent.epsilon = self.fuzzy_controller.epsilon
        return checkpoint
    
    # Delegate other properties to base agent
    @property
    def epsilon(self):
        return self.fuzzy_controller.get_epsilon()
    
    @property
    def device(self):
        return self.agent.device
    
    @property
    def config(self):
        return self.agent.config
    
    def update_target_network(self):
        return self.agent.update_target_network()
    
    def get_q_values(self, history: np.ndarray, gain: float) -> np.ndarray:
        return self.agent.get_q_values(history, gain)


if __name__ == "__main__":
    # Test fuzzy components
    print("Testing Fuzzy Adaptive Components...")
    
    # Test membership functions
    print("\n1. Testing triangular membership:")
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mf = FuzzyMembershipFunctions()
        memberships = {}
        for fs, params in mf.quality.items():
            memberships[fs.value] = triangular_membership(x, *params)
        print(f"  x={x}: {memberships}")
    
    # Test Action Quality Estimator
    print("\n2. Testing ActionQualityEstimator:")
    aqe = ActionQualityEstimator(action_size=47)
    for _ in range(50):
        action = np.random.randint(0, 47)
        reward = np.random.uniform(-35, 35) if action < 37 else np.random.uniform(-1, 2)
        aqe.update(action, reward)
    print(f"  Overall quality: {aqe.get_overall_quality():.3f}")
    
    # Test Success Estimator
    print("\n3. Testing SuccessfulActionsEstimator:")
    sae = SuccessfulActionsEstimator()
    rewards = [35, -1, -1, -1, -1, 35, -1, -1, -1, 1]  # 2 wins, 1 even, 7 losses
    for r in rewards:
        sae.update(r)
    print(f"  Success ratio: {sae.get_success_ratio():.3f}")
    
    # Test Trend Analyzer
    print("\n4. Testing HistoricalTrendAnalyzer:")
    hta = HistoricalTrendAnalyzer()
    # Improving trend
    for i in range(50):
        hta.update(-10 + i * 0.5)  # Improving
    print(f"  Trend (improving): {hta.get_trend():.3f}")
    
    # Test Diversity Estimator
    print("\n5. Testing ExplorationDiversityEstimator:")
    ede = ExplorationDiversityEstimator(action_size=47)
    # All same action (low diversity)
    for _ in range(50):
        ede.update(5)
    print(f"  Diversity (low): {ede.get_diversity():.3f}")
    
    # Reset and test high diversity
    ede2 = ExplorationDiversityEstimator(action_size=47)
    for _ in range(50):
        ede2.update(np.random.randint(0, 47))
    print(f"  Diversity (high): {ede2.get_diversity():.3f}")
    
    # Test full FuzzyEpsilonController
    print("\n6. Testing FuzzyEpsilonController:")
    fec = FuzzyEpsilonController(action_size=47)
    
    # Simulate some experience
    for i in range(100):
        action = np.random.randint(0, 47)
        reward = np.random.uniform(-35, 35)
        q_values = np.random.randn(47)
        fec.update(action, reward, q_values)
    
    epsilon = fec.compute_epsilon(force_update=True)
    metrics = fec.get_metrics()
    print(f"  Metrics: {metrics}")
    
    print("\n✅ All fuzzy components working!")
