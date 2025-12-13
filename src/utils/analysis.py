"""
Analysis utilities for roulette data.

Provides statistical analysis, transition matrices, and pattern detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


# Roulette constants
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}


def compute_transition_matrix(numbers: List[int], normalize: bool = True) -> np.ndarray:
    """
    Compute transition probability matrix P[i,j] = P(next=j | current=i).
    
    Args:
        numbers: List of roulette numbers
        normalize: If True, return probabilities; if False, return counts
    
    Returns:
        37x37 matrix
    """
    matrix = np.zeros((37, 37), dtype=np.float32)
    
    for i in range(len(numbers) - 1):
        current = numbers[i]
        next_num = numbers[i + 1]
        matrix[current, next_num] += 1
    
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums
    
    return matrix


def compute_chi_square(numbers: List[int]) -> Dict[str, Any]:
    """
    Compute chi-square test for uniformity.
    
    Tests if the distribution of numbers is uniform (as expected for fair roulette).
    
    Returns:
        dict with chi_square value, p_value estimate, and is_uniform flag
    """
    n = len(numbers)
    expected = n / 37
    
    observed = Counter(numbers)
    chi_sq = sum(
        (observed.get(i, 0) - expected) ** 2 / expected
        for i in range(37)
    )
    
    # Critical value for df=36, alpha=0.05 is 50.998
    # Critical value for df=36, alpha=0.01 is 58.619
    
    return {
        "chi_square": chi_sq,
        "degrees_of_freedom": 36,
        "critical_value_95": 50.998,
        "critical_value_99": 58.619,
        "is_uniform_95": chi_sq < 50.998,
        "is_uniform_99": chi_sq < 58.619,
        "sample_size": n
    }


def compute_autocorrelation(numbers: List[int], max_lag: int = 20) -> Dict[int, float]:
    """
    Compute autocorrelation at different lags.
    
    High autocorrelation at any lag would indicate non-randomness.
    
    Args:
        numbers: List of roulette numbers
        max_lag: Maximum lag to compute
    
    Returns:
        Dict mapping lag to autocorrelation value
    """
    arr = np.array(numbers, dtype=np.float32)
    mean = arr.mean()
    var = arr.var()
    
    if var == 0:
        return {lag: 0.0 for lag in range(1, max_lag + 1)}
    
    autocorr = {}
    for lag in range(1, min(max_lag + 1, len(arr))):
        cov = np.mean((arr[:-lag] - mean) * (arr[lag:] - mean))
        autocorr[lag] = cov / var
    
    return autocorr


def compute_runs_test(numbers: List[int]) -> Dict[str, Any]:
    """
    Compute runs test for randomness.
    
    A "run" is a sequence of consecutive same values (or same category).
    This tests if there are too many or too few runs.
    """
    # Convert to binary (above/below median)
    median = np.median(numbers)
    binary = [1 if n > median else 0 for n in numbers]
    
    # Count runs
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1
    
    # Count n1 (above median) and n2 (below median)
    n1 = sum(binary)
    n2 = len(binary) - n1
    
    # Expected runs and variance
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    
    # Z-score
    z_score = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    
    return {
        "runs": runs,
        "expected_runs": expected_runs,
        "z_score": z_score,
        "is_random_95": abs(z_score) < 1.96,  # 95% confidence
        "is_random_99": abs(z_score) < 2.58,  # 99% confidence
    }


def analyze_streaks(numbers: List[int]) -> Dict[str, Any]:
    """
    Analyze streaks in the data.
    
    Looks for:
    - Repeat streaks (same number)
    - Color streaks
    - Dozen streaks
    """
    # Repeat streaks
    max_repeat = 1
    current_repeat = 1
    repeat_positions = []
    
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i-1]:
            current_repeat += 1
            if current_repeat > max_repeat:
                max_repeat = current_repeat
                repeat_positions.append((i - current_repeat + 1, numbers[i], current_repeat))
        else:
            current_repeat = 1
    
    # Color streaks
    def get_color(n):
        if n == 0:
            return 'green'
        return 'red' if n in RED_NUMBERS else 'black'
    
    colors = [get_color(n) for n in numbers]
    max_color_streak = 1
    current_color_streak = 1
    
    for i in range(1, len(colors)):
        if colors[i] == colors[i-1] and colors[i] != 'green':
            current_color_streak += 1
            max_color_streak = max(max_color_streak, current_color_streak)
        else:
            current_color_streak = 1
    
    # Dozen streaks
    def get_dozen(n):
        if n == 0:
            return 0
        return (n - 1) // 12 + 1
    
    dozens = [get_dozen(n) for n in numbers]
    max_dozen_streak = 1
    current_dozen_streak = 1
    
    for i in range(1, len(dozens)):
        if dozens[i] == dozens[i-1] and dozens[i] != 0:
            current_dozen_streak += 1
            max_dozen_streak = max(max_dozen_streak, current_dozen_streak)
        else:
            current_dozen_streak = 1
    
    return {
        "max_repeat_streak": max_repeat,
        "max_color_streak": max_color_streak,
        "max_dozen_streak": max_dozen_streak,
        "notable_repeats": [r for r in repeat_positions if r[2] >= 2][-5:]
    }


def compute_hot_cold_numbers(
    numbers: List[int], 
    recent_window: int = 50
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Identify hot and cold numbers.
    
    Hot = appearing more frequently than expected in recent spins
    Cold = appearing less frequently than expected
    """
    if len(numbers) < recent_window:
        recent_window = len(numbers)
    
    recent = numbers[-recent_window:]
    recent_counts = Counter(recent)
    
    expected = recent_window / 37
    threshold_hot = expected * 1.5
    threshold_cold = expected * 0.5
    
    hot = [(n, count) for n, count in recent_counts.items() if count >= threshold_hot]
    cold = [(n, 0) for n in range(37) if recent_counts.get(n, 0) <= threshold_cold]
    
    # Sort by count
    hot = sorted(hot, key=lambda x: x[1], reverse=True)
    cold = sorted(cold, key=lambda x: x[1])
    
    return {
        "hot": hot[:10],
        "cold": cold[:10],
        "recent_window": recent_window,
        "expected_per_number": expected
    }


def quick_stats(numbers: List[int]) -> Dict[str, Any]:
    if not numbers:
        return {"red": 0, "black": 0, "green": 0, "red_pct": 0, "black_pct": 0, "green_pct": 0}
    
    total = len(numbers)
    red = sum(1 for n in numbers if n in RED_NUMBERS)
    black = sum(1 for n in numbers if n in BLACK_NUMBERS)
    green = sum(1 for n in numbers if n == 0)
    
    return {
        "red": red,
        "black": black,
        "green": green,
        "red_pct": red / total * 100,
        "black_pct": black / total * 100,
        "green_pct": green / total * 100,
        "total": total
    }


def full_analysis(numbers: List[int]) -> Dict[str, Any]:
    """
    Run complete analysis on a sequence of numbers.
    
    Returns comprehensive statistics and tests.
    """
    if len(numbers) < 10:
        return {"error": "Need at least 10 numbers for analysis"}
    
    # Basic statistics
    counter = Counter(numbers)
    total = len(numbers)
    
    # Color distribution
    red_count = sum(1 for n in numbers if n in RED_NUMBERS)
    black_count = sum(1 for n in numbers if n in BLACK_NUMBERS)
    green_count = sum(1 for n in numbers if n == 0)
    
    # Section distribution
    low_count = sum(1 for n in numbers if 1 <= n <= 18)
    high_count = sum(1 for n in numbers if 19 <= n <= 36)
    odd_count = sum(1 for n in numbers if n != 0 and n % 2 == 1)
    even_count = sum(1 for n in numbers if n != 0 and n % 2 == 0)
    
    # Dozens
    dozen1 = sum(1 for n in numbers if 1 <= n <= 12)
    dozen2 = sum(1 for n in numbers if 13 <= n <= 24)
    dozen3 = sum(1 for n in numbers if 25 <= n <= 36)
    
    return {
        "total_spins": total,
        "unique_numbers_seen": len(counter),
        
        "colors": {
            "red": red_count,
            "black": black_count,
            "green": green_count,
            "red_pct": red_count / total * 100,
            "black_pct": black_count / total * 100,
            "green_pct": green_count / total * 100
        },
        
        "halves": {
            "low_1_18": low_count,
            "high_19_36": high_count,
            "low_pct": low_count / total * 100,
            "high_pct": high_count / total * 100
        },
        
        "parity": {
            "odd": odd_count,
            "even": even_count,
            "odd_pct": odd_count / total * 100,
            "even_pct": even_count / total * 100
        },
        
        "dozens": {
            "first": dozen1,
            "second": dozen2,
            "third": dozen3,
            "first_pct": dozen1 / total * 100,
            "second_pct": dozen2 / total * 100,
            "third_pct": dozen3 / total * 100
        },
        
        "chi_square": compute_chi_square(numbers),
        "runs_test": compute_runs_test(numbers),
        "autocorrelation": compute_autocorrelation(numbers, max_lag=10),
        "streaks": analyze_streaks(numbers),
        "hot_cold": compute_hot_cold_numbers(numbers),
        
        "most_common": counter.most_common(5),
        "least_common": counter.most_common()[:-6:-1] if len(counter) >= 5 else counter.most_common()[::-1]
    }


def plot_transition_matrix(
    numbers: List[int],
    title: str = "Transition Probability Matrix",
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot the transition matrix as a heatmap."""
    matrix = compute_transition_matrix(numbers)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel('Next Number')
    ax.set_ylabel('Current Number')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability')
    
    # Set ticks
    ax.set_xticks(range(0, 37, 5))
    ax.set_yticks(range(0, 37, 5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_number_distribution(
    numbers: List[int],
    title: str = "Number Distribution",
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot the distribution of numbers."""
    counter = Counter(numbers)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    x = range(37)
    heights = [counter.get(i, 0) for i in x]
    
    # Color bars by roulette color
    colors = []
    for i in range(37):
        if i == 0:
            colors.append('green')
        elif i in RED_NUMBERS:
            colors.append('red')
        else:
            colors.append('black')
    
    bars = ax.bar(x, heights, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add expected line
    expected = len(numbers) / 37
    ax.axhline(y=expected, color='blue', linestyle='--', label=f'Expected ({expected:.1f})')
    
    ax.set_xlabel('Number')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(range(0, 37, 2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test with random data
    import random
    
    numbers = [random.randint(0, 36) for _ in range(500)]
    
    print("=== Full Analysis ===")
    analysis = full_analysis(numbers)
    
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
