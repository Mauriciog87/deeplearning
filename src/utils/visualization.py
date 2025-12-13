"""
Visualization utilities for RL Roulette training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os


def plot_training_results(
    rewards: List[float],
    losses: List[float],
    epsilons: List[float],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training progress with rewards, losses, and epsilon decay.
    
    Args:
        rewards: List of episode rewards
        losses: List of training losses
        epsilons: List of epsilon values
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Rewards
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.6, color='blue', label='Episode Reward')
    if len(rewards) >= 20:
        window = min(50, len(rewards) // 5)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, color='red', 
                linewidth=2, label=f'Moving Avg ({window})')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward ($)')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Losses
    ax2 = axes[1]
    if losses and any(l is not None for l in losses):
        valid_losses = [l for l in losses if l is not None]
        ax2.plot(valid_losses, alpha=0.6, color='orange')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No loss data available', ha='center', va='center')
    
    # Epsilon
    ax3 = axes[2]
    ax3.plot(epsilons, color='green')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results: Dict[str, Dict],
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot comparison between different strategies.
    
    Args:
        results: Dict with strategy names as keys and results dict as values
                 Each result dict should have: 'total_reward', 'win_rate', 'avg_steps'
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    strategies = list(results.keys())
    
    # Extract metrics
    total_rewards = [results[s].get('total_reward', 0) for s in strategies]
    win_rates = [results[s].get('win_rate', 0) * 100 for s in strategies]
    avg_steps = [results[s].get('avg_steps', 0) for s in strategies]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors for each strategy
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Total Reward
    ax1 = axes[0]
    bars1 = ax1.bar(strategies, total_rewards, color=colors[:len(strategies)])
    ax1.set_ylabel('Total Reward ($)')
    ax1.set_title('Total Reward (100 episodes)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, value in zip(bars1, total_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'${value:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # Win Rate
    ax2 = axes[1]
    bars2 = ax2.bar(strategies, win_rates, color=colors[:len(strategies)])
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate')
    ax2.set_ylim(0, 100)
    for bar, value in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Average Steps
    ax3 = axes[2]
    bars3 = ax3.bar(strategies, avg_steps, color=colors[:len(strategies)])
    ax3.set_ylabel('Average Steps')
    ax3.set_title('Average Steps per Episode')
    for bar, value in zip(bars3, avg_steps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_episode_details(
    rewards: List[float],
    bankrolls: List[float],
    actions: List[str],
    title: str = "Episode Details",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot details of a single episode.
    
    Args:
        rewards: List of rewards at each step
        bankrolls: List of bankroll values at each step
        actions: List of action names taken
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    steps = range(len(rewards))
    
    # Bankroll over time
    ax1 = axes[0]
    ax1.plot(bankrolls, color='green', linewidth=2)
    ax1.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax1.axhline(y=2000, color='gold', linestyle='--', alpha=0.5, label='Win Target')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Bankrupt')
    ax1.fill_between(steps, bankrolls, 1000, alpha=0.3, 
                     where=[b > 1000 for b in bankrolls], color='green')
    ax1.fill_between(steps, bankrolls, 1000, alpha=0.3,
                     where=[b < 1000 for b in bankrolls], color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bankroll ($)')
    ax1.set_title('Bankroll Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rewards per step
    ax2 = axes[1]
    colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in rewards]
    ax2.bar(steps, rewards, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward ($)')
    ax2.set_title('Reward per Step')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Episode details plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing visualization utilities...")
    
    # Test training results plot
    np.random.seed(42)
    n_episodes = 100
    rewards = np.cumsum(np.random.randn(n_episodes) * 100).tolist()
    losses = [abs(x) for x in np.random.randn(n_episodes * 10) * 0.1]
    epsilons = [max(0.1, 1.0 * (0.995 ** i)) for i in range(n_episodes)]
    
    print("Creating training results plot...")
    plot_training_results(rewards, losses, epsilons, 
                         title="Test: Training Progress", show=False)
    
    # Test comparison plot
    results = {
        'Straight Up': {'total_reward': 122700, 'win_rate': 0.62, 'avg_steps': 45},
        'One-to-One': {'total_reward': 0, 'win_rate': 0.0, 'avg_steps': 80},
        'Two-to-One': {'total_reward': 47450, 'win_rate': 0.75, 'avg_steps': 80}
    }
    
    print("Creating comparison plot...")
    plot_comparison(results, title="Test: Strategy Comparison", show=False)
    
    print("\n✅ Visualization tests passed!")
