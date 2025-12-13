"""
Near-Miss Utilities for Roulette Analysis

Based on the research paper "The influence of near-miss events on betting behavior"
which analyzed 565,915 betting decisions from real casino data.

Key Findings from the paper:
- 67% of players show Gambler's Fallacy (avoid near-miss numbers)
- 33% of players show Hot Hand belief (prefer near-miss numbers)
- Near-miss proximity (wheel and table) significantly affects betting behavior
- Odds ratio of 0.87 for betting on near-miss numbers (overall avoidance)

Near-miss definitions:
1. Wheel proximity: Numbers physically adjacent on the roulette wheel
2. Table proximity: Numbers adjacent on the betting table layout
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from collections import Counter


# European roulette wheel order (clockwise from 0)
# This is the physical arrangement of numbers on a European wheel
EUROPEAN_WHEEL_ORDER = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
    5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

# Create lookup tables for wheel positions
WHEEL_POSITION = {num: idx for idx, num in enumerate(EUROPEAN_WHEEL_ORDER)}
POSITION_TO_NUMBER = {idx: num for idx, num in enumerate(EUROPEAN_WHEEL_ORDER)}

# Table layout for proximity (3 columns x 12 rows + 0)
# Row i contains: [3*i+1, 3*i+2, 3*i+3] for i in 0..11
TABLE_LAYOUT = {
    # Number -> (row, col) where row 0-11, col 0-2
    0: (-1, 1),  # 0 is above the table
}
for i in range(12):
    for j in range(3):
        num = 3 * i + j + 1
        TABLE_LAYOUT[num] = (i, j)


def get_wheel_neighbors(number: int, distance: int = 1) -> Set[int]:
    """
    Get numbers that are within 'distance' positions on the physical wheel.
    
    Args:
        number: The reference number (0-36)
        distance: How many positions to look in each direction
        
    Returns:
        Set of neighboring numbers (excluding the number itself)
    """
    if number not in WHEEL_POSITION:
        return set()
    
    pos = WHEEL_POSITION[number]
    wheel_size = len(EUROPEAN_WHEEL_ORDER)
    
    neighbors = set()
    for d in range(-distance, distance + 1):
        if d != 0:
            neighbor_pos = (pos + d) % wheel_size
            neighbors.add(EUROPEAN_WHEEL_ORDER[neighbor_pos])
    
    return neighbors


def get_wheel_distance(num1: int, num2: int) -> int:
    """
    Get the minimum wheel distance between two numbers.
    
    Args:
        num1: First number (0-36)
        num2: Second number (0-36)
        
    Returns:
        Minimum number of positions between numbers on the wheel
    """
    if num1 not in WHEEL_POSITION or num2 not in WHEEL_POSITION:
        return -1
    
    pos1 = WHEEL_POSITION[num1]
    pos2 = WHEEL_POSITION[num2]
    wheel_size = len(EUROPEAN_WHEEL_ORDER)
    
    # Minimum of clockwise and counter-clockwise distance
    clockwise = (pos2 - pos1) % wheel_size
    counter_clockwise = (pos1 - pos2) % wheel_size
    
    return min(clockwise, counter_clockwise)


def get_table_neighbors(number: int) -> Set[int]:
    """
    Get numbers that are adjacent on the betting table.
    Adjacent means sharing an edge (not diagonal).
    
    Args:
        number: The reference number (0-36)
        
    Returns:
        Set of table-adjacent numbers
    """
    if number not in TABLE_LAYOUT:
        return set()
    
    neighbors = set()
    row, col = TABLE_LAYOUT[number]
    
    # Special case for 0
    if number == 0:
        return {1, 2, 3}  # 0 is adjacent to first row
    
    # Check all 4 directions (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        
        # Check bounds
        if 0 <= new_row < 12 and 0 <= new_col < 3:
            neighbor_num = 3 * new_row + new_col + 1
            neighbors.add(neighbor_num)
        
        # Special: first row is adjacent to 0
        if new_row == -1 and col == 1:  # Going up from middle column
            neighbors.add(0)
    
    return neighbors


def get_table_distance(num1: int, num2: int) -> int:
    """
    Get the Manhattan distance between two numbers on the table.
    
    Args:
        num1: First number (0-36)
        num2: Second number (0-36)
        
    Returns:
        Manhattan distance on the table layout
    """
    if num1 not in TABLE_LAYOUT or num2 not in TABLE_LAYOUT:
        return -1
    
    r1, c1 = TABLE_LAYOUT[num1]
    r2, c2 = TABLE_LAYOUT[num2]
    
    # Handle 0 specially
    if num1 == 0:
        r1, c1 = -1, c2  # Treat as directly above
    if num2 == 0:
        r2, c2 = -1, c1
    
    return abs(r1 - r2) + abs(c1 - c2)


def is_near_miss_wheel(bet_number: int, winning_number: int, threshold: int = 2) -> bool:
    """
    Check if a bet was a near-miss based on wheel proximity.
    
    Args:
        bet_number: The number that was bet on (0-36)
        winning_number: The number that won
        threshold: Maximum wheel distance to consider a near-miss
        
    Returns:
        True if it was a near-miss (close on wheel but didn't win)
    """
    if bet_number == winning_number:
        return False  # Not a miss if you won
    
    distance = get_wheel_distance(bet_number, winning_number)
    return 0 < distance <= threshold


def is_near_miss_table(bet_number: int, winning_number: int) -> bool:
    """
    Check if a bet was a near-miss based on table proximity.
    
    Args:
        bet_number: The number that was bet on (0-36)
        winning_number: The number that won
        
    Returns:
        True if numbers are adjacent on the table layout
    """
    if bet_number == winning_number:
        return False
    
    return winning_number in get_table_neighbors(bet_number)


def compute_near_miss_features(
    history: List[int],
    current_number: Optional[int] = None,
    window: int = 5
) -> Dict[str, float]:
    """
    Compute near-miss related features from a sequence of numbers.
    
    Args:
        history: List of past winning numbers
        current_number: Current winning number (if available)
        window: How many recent numbers to consider
        
    Returns:
        Dictionary of computed features:
        - wheel_cluster_density: How clustered recent numbers are on the wheel
        - table_cluster_density: How clustered recent numbers are on the table
        - wheel_sector_bias: Which wheel sector is "hot"
        - consecutive_neighbors: Count of consecutive wheel neighbors
        - hot_zone_center: Center of the hottest wheel zone
    """
    if len(history) < 2:
        return {
            "wheel_cluster_density": 0.0,
            "table_cluster_density": 0.0,
            "wheel_sector_bias": 0.0,
            "consecutive_neighbors": 0,
            "hot_zone_center": 0
        }
    
    recent = history[-window:] if len(history) >= window else history
    
    # Wheel cluster density: average pairwise wheel distance
    wheel_distances = []
    for i, n1 in enumerate(recent):
        for n2 in recent[i+1:]:
            d = get_wheel_distance(n1, n2)
            if d >= 0:
                wheel_distances.append(d)
    
    wheel_cluster = 1.0 - (np.mean(wheel_distances) / 18.0) if wheel_distances else 0.0
    
    # Table cluster density: average pairwise table distance
    table_distances = []
    for i, n1 in enumerate(recent):
        for n2 in recent[i+1:]:
            d = get_table_distance(n1, n2)
            if d >= 0:
                table_distances.append(d)
    
    max_table_dist = 14  # Max possible Manhattan distance on table
    table_cluster = 1.0 - (np.mean(table_distances) / max_table_dist) if table_distances else 0.0
    
    # Count consecutive wheel neighbors
    consecutive_neighbors = 0
    for i in range(len(recent) - 1):
        if get_wheel_distance(recent[i], recent[i+1]) <= 2:
            consecutive_neighbors += 1
    
    # Find wheel sector bias (divide wheel into 4 sectors)
    sector_counts = [0, 0, 0, 0]
    for num in recent:
        if num in WHEEL_POSITION:
            sector = WHEEL_POSITION[num] // 10  # Roughly 4 sectors
            if sector < 4:
                sector_counts[sector] += 1
    
    # Sector bias: how uneven the distribution is
    expected = len(recent) / 4
    sector_bias = sum(abs(c - expected) for c in sector_counts) / len(recent) if recent else 0.0
    
    # Hot zone center: weighted average position on wheel
    positions = [WHEEL_POSITION[n] for n in recent if n in WHEEL_POSITION]
    if positions:
        # Circular mean
        angles = [2 * np.pi * p / 37 for p in positions]
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        hot_zone_center = int((mean_angle / (2 * np.pi)) * 37) % 37
        hot_zone_center = EUROPEAN_WHEEL_ORDER[hot_zone_center]
    else:
        hot_zone_center = 0
    
    return {
        "wheel_cluster_density": float(np.clip(wheel_cluster, 0, 1)),
        "table_cluster_density": float(np.clip(table_cluster, 0, 1)),
        "wheel_sector_bias": float(np.clip(sector_bias, 0, 1)),
        "consecutive_neighbors": consecutive_neighbors,
        "hot_zone_center": hot_zone_center
    }


def get_near_miss_zones(
    last_number: int,
    wheel_distance: int = 3
) -> Dict[str, Set[int]]:
    """
    Get the "near-miss zones" relative to the last winning number.
    
    Based on the paper's findings about player behavior:
    - Gambler's Fallacy players avoid these zones
    - Hot Hand players prefer these zones
    
    Args:
        last_number: The most recent winning number
        wheel_distance: How far on the wheel to consider "near"
        
    Returns:
        Dictionary with 'wheel_zone' and 'table_zone' sets
    """
    return {
        "wheel_zone": get_wheel_neighbors(last_number, wheel_distance),
        "table_zone": get_table_neighbors(last_number),
        "combined_zone": get_wheel_neighbors(last_number, wheel_distance) | get_table_neighbors(last_number)
    }


def encode_wheel_position(number: int) -> Tuple[float, float]:
    """
    Encode a number's wheel position as (sin, cos) for neural network input.
    This preserves the circular nature of the wheel.
    
    Args:
        number: Roulette number (0-36)
        
    Returns:
        Tuple of (sin, cos) encoding the position on the wheel
    """
    if number not in WHEEL_POSITION:
        return (0.0, 0.0)
    
    pos = WHEEL_POSITION[number]
    angle = 2 * np.pi * pos / 37
    
    return (float(np.sin(angle)), float(np.cos(angle)))


def compute_wheel_sector_frequencies(
    history: List[int],
    n_sectors: int = 8
) -> np.ndarray:
    """
    Compute the frequency of numbers landing in each wheel sector.
    
    Args:
        history: List of winning numbers
        n_sectors: Number of sectors to divide the wheel into
        
    Returns:
        Array of shape (n_sectors,) with normalized frequencies
    """
    sector_size = 37 / n_sectors
    counts = np.zeros(n_sectors)
    
    for num in history:
        if num in WHEEL_POSITION:
            sector = int(WHEEL_POSITION[num] / sector_size)
            sector = min(sector, n_sectors - 1)  # Handle edge case
            counts[sector] += 1
    
    # Normalize
    total = counts.sum()
    if total > 0:
        counts = counts / total
    
    return counts


class NearMissTracker:
    """
    Track near-miss events during gameplay for analysis.
    
    This class maintains statistics about near-miss occurrences
    to help understand patterns and potential biases.
    """
    
    def __init__(self, wheel_threshold: int = 2):
        """
        Initialize the tracker.
        
        Args:
            wheel_threshold: Wheel distance threshold for near-miss
        """
        self.wheel_threshold = wheel_threshold
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_bets = 0
        self.near_miss_wheel = 0
        self.near_miss_table = 0
        self.near_miss_both = 0
        self.wins = 0
        self.bet_history = []  # (bet_number, winning_number, is_near_miss_w, is_near_miss_t)
    
    def record_bet(self, bet_number: int, winning_number: int) -> Dict[str, bool]:
        """
        Record a bet and update statistics.
        
        Args:
            bet_number: Number bet on (0-36)
            winning_number: Number that won
            
        Returns:
            Dictionary with near-miss analysis
        """
        self.total_bets += 1
        
        won = (bet_number == winning_number)
        nm_wheel = is_near_miss_wheel(bet_number, winning_number, self.wheel_threshold)
        nm_table = is_near_miss_table(bet_number, winning_number)
        
        if won:
            self.wins += 1
        if nm_wheel:
            self.near_miss_wheel += 1
        if nm_table:
            self.near_miss_table += 1
        if nm_wheel and nm_table:
            self.near_miss_both += 1
        
        self.bet_history.append((bet_number, winning_number, nm_wheel, nm_table))
        
        return {
            "won": won,
            "near_miss_wheel": nm_wheel,
            "near_miss_table": nm_table,
            "near_miss_both": nm_wheel and nm_table
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.total_bets == 0:
            return {
                "win_rate": 0.0,
                "near_miss_wheel_rate": 0.0,
                "near_miss_table_rate": 0.0,
                "near_miss_both_rate": 0.0,
                "total_bets": 0
            }
        
        return {
            "win_rate": self.wins / self.total_bets,
            "near_miss_wheel_rate": self.near_miss_wheel / self.total_bets,
            "near_miss_table_rate": self.near_miss_table / self.total_bets,
            "near_miss_both_rate": self.near_miss_both / self.total_bets,
            "total_bets": self.total_bets
        }


# Pre-computed lookup tables for fast access
WHEEL_NEIGHBORS_1 = {n: get_wheel_neighbors(n, 1) for n in range(37)}
WHEEL_NEIGHBORS_2 = {n: get_wheel_neighbors(n, 2) for n in range(37)}
WHEEL_NEIGHBORS_3 = {n: get_wheel_neighbors(n, 3) for n in range(37)}
TABLE_NEIGHBORS = {n: get_table_neighbors(n) for n in range(37)}


if __name__ == "__main__":
    # Test the utilities
    print("=== Near-Miss Utilities Test ===\n")
    
    # Test wheel neighbors
    print("Wheel neighbors of 0 (distance 2):", get_wheel_neighbors(0, 2))
    print("Wheel neighbors of 17 (distance 1):", get_wheel_neighbors(17, 1))
    
    # Test wheel distance
    print("\nWheel distance 0 to 32:", get_wheel_distance(0, 32))  # Should be 1
    print("Wheel distance 0 to 26:", get_wheel_distance(0, 26))  # Should be 1
    
    # Test table neighbors  
    print("\nTable neighbors of 0:", get_table_neighbors(0))
    print("Table neighbors of 17:", get_table_neighbors(17))
    
    # Test near-miss detection
    print("\nNear-miss wheel (17 bet, 34 won):", is_near_miss_wheel(17, 34))  # Adjacent on wheel
    print("Near-miss table (17 bet, 18 won):", is_near_miss_table(17, 18))   # Adjacent on table
    
    # Test feature computation
    history = [17, 34, 6, 27, 13]  # Numbers close on wheel
    features = compute_near_miss_features(history)
    print("\nFeatures for clustered history:", features)
    
    history = [0, 18, 36, 9, 27]  # More spread out
    features = compute_near_miss_features(history)
    print("Features for spread history:", features)
    
    # Test wheel position encoding
    print("\nWheel position encoding for 0:", encode_wheel_position(0))
    print("Wheel position encoding for 18:", encode_wheel_position(18))
    
    # Test NearMissTracker
    print("\n=== NearMissTracker Test ===")
    tracker = NearMissTracker()
    
    # Simulate some bets
    test_bets = [
        (17, 34),  # Near-miss wheel
        (17, 18),  # Near-miss table  
        (17, 17),  # Win
        (5, 32),   # Far miss
        (6, 27),   # Near-miss wheel
    ]
    
    for bet, win in test_bets:
        result = tracker.record_bet(bet, win)
        print(f"Bet {bet}, Won {win}: {result}")
    
    print("\nFinal statistics:", tracker.get_statistics())
