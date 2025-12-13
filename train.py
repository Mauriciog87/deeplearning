"""
Training script for RL Roulette agents.

Supports training with simulated data or real roulette data from SQLite database.

Usage:
    # Train with simulated random data
    python train.py --episodes 1000
    
    # Train with real data from database
    python train.py --use-real-data --episodes 500
    
    # Train with specific session
    python train.py --use-real-data --session-id 3 --episodes 500
    
    # Resume training from checkpoint
    python train.py --resume models/checkpoint.pt --episodes 500
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

from src.environment.roulette_env import RouletteEnv, RouletteEnvFlat
from src.agents.dqn_agent import DQNAgent
from src.agents.fuzzy_adaptive import FuzzyAdaptiveDQN, FuzzyEpsilonController
from src.utils.visualization import plot_training_results


def get_real_data(session_id: int = None, min_spins: int = 100) -> list:
    """Load real roulette data from database."""
    try:
        from src.database import RouletteRepository
        
        repo = RouletteRepository()
        
        if session_id:
            # Get specific session
            numbers = repo.get_numbers_by_session(session_id)
            print(f"📊 Loaded {len(numbers)} spins from session {session_id}")
        else:
            # Get all numbers
            numbers = repo.get_all_numbers()
            print(f"📊 Loaded {len(numbers)} total spins from database")
        
        if len(numbers) < min_spins:
            print(f"⚠️ Warning: Only {len(numbers)} spins available (minimum recommended: {min_spins})")
        
        return numbers
    
    except ImportError:
        print("❌ Database module not available")
        return []
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []


def train_agent(
    episodes: int = 1000,
    save_dir: str = "models",
    model_name: str = "roulette_agent",
    verbose: bool = True,
    use_real_data: bool = False,
    real_numbers: list = None,
    resume_path: str = None,
    # Hyperparameters
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.998,
    batch_size: int = 64,
    buffer_size: int = 50000,
    target_update_freq: int = 10,
    max_spins: int = 100,
    initial_bankroll: float = 1000.0,
    use_flat_env: bool = False,
    use_fuzzy_adaptive: bool = False,
    fuzzy_adjustment_rate: float = 0.1,
) -> dict:
    """
    Train a DQN agent for roulette.
    
    Args:
        episodes: Number of training episodes
        save_dir: Directory to save models
        model_name: Name for saved model
        verbose: Whether to print progress
        use_real_data: Whether to use real data from database
        real_numbers: Pre-loaded real numbers (optional)
        resume_path: Path to checkpoint to resume from
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        epsilon_start: Starting epsilon for exploration
        epsilon_min: Minimum epsilon
        epsilon_decay: Epsilon decay rate per episode
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        target_update_freq: Frequency to update target network
        max_spins: Maximum spins per episode
        initial_bankroll: Starting bankroll
        use_flat_env: Use flattened observation space (simpler)
        use_fuzzy_adaptive: Use fuzzy adaptive epsilon controller
        fuzzy_adjustment_rate: How quickly fuzzy controller adjusts epsilon
    
    Returns:
        Dictionary with training results
    """
    # Create environment
    if use_flat_env:
        env = RouletteEnvFlat(
            history_size=20,
            max_steps=max_spins,
            initial_bankroll=initial_bankroll
        )
        observation_type = "flat"
    else:
        env = RouletteEnv(
            history_size=20,
            max_steps=max_spins,
            initial_bankroll=initial_bankroll
        )
        observation_type = "dict"
    
    # Set real data if available
    if use_real_data and real_numbers:
        env.set_real_data(real_numbers)
    
    action_size = env.action_space.n  # 47 actions
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎰 RL Roulette Training")
        print(f"{'='*60}")
        print(f"Observation type: {observation_type}")
        print(f"Action size: {action_size}")
        print(f"History length: {env.history_size}")
        print(f"Max spins per episode: {max_spins}")
        print(f"Initial bankroll: ${initial_bankroll:.0f}")
        print(f"Episodes: {episodes}")
        print(f"Data source: {'Real data' if use_real_data else 'Simulated'}")
        if use_real_data and real_numbers:
            print(f"Real data spins: {len(real_numbers)}")
        print(f"Fuzzy adaptive: {'Yes' if use_fuzzy_adaptive else 'No'}")
        print(f"{'='*60}\n")
    
    # Create agent
    base_agent = DQNAgent(
        history_size=20,
        embedding_dim=64,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_freq=target_update_freq
    )
    
    # Wrap with fuzzy adaptive controller if requested
    if use_fuzzy_adaptive:
        agent = FuzzyAdaptiveDQN(
            base_agent=base_agent,
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_start,
            adjustment_rate=fuzzy_adjustment_rate,
            update_frequency=10
        )
        if verbose:
            print("🧠 Using Fuzzy Adaptive DQN (FLC-EA inspired)")
    else:
        agent = base_agent
    
    # Resume from checkpoint if specified
    start_episode = 0
    all_rewards = []
    all_losses = []
    all_epsilons = []
    all_fuzzy_metrics = []  # For fuzzy adaptive tracking
    
    if resume_path and os.path.exists(resume_path):
        checkpoint = agent.load(resume_path)
        if checkpoint and 'episode' in checkpoint:
            start_episode = checkpoint['episode']
            all_rewards = checkpoint.get('rewards', [])
            all_losses = checkpoint.get('losses', [])
            all_epsilons = checkpoint.get('epsilons', [])
            print(f"📂 Resumed from episode {start_episode}")
    
    # Training loop
    pbar = tqdm(range(start_episode, start_episode + episodes), desc="Training", disable=not verbose)
    
    for episode in pbar:
        obs, info = env.reset()
        
        # Extract history and gain from observation
        if use_flat_env:
            # Flat env returns numpy array
            history = obs[:20].astype(np.int64)
            gain = obs[20]
        else:
            # Dict env returns dict
            history = obs["history"]
            gain = obs["gain"][0]
        
        episode_reward = 0
        episode_losses = []
        done = False
        steps = 0
        
        while not done:
            # Select action (fuzzy adaptive returns tuple with q_values)
            if use_fuzzy_adaptive:
                action, q_values = agent.act(history, gain, training=True)
            else:
                action = agent.act(history, gain, training=True)
                q_values = None
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract next history and gain
            if use_flat_env:
                next_history = next_obs[:20].astype(np.int64)
                next_gain = next_obs[20]
            else:
                next_history = next_obs["history"]
                next_gain = next_obs["gain"][0]
            
            # Store transition (fuzzy adaptive needs q_values)
            if use_fuzzy_adaptive:
                agent.remember(history, gain, action, reward, next_history, next_gain, done, q_values)
            else:
                agent.remember(history, gain, action, reward, next_history, next_gain, done)
            
            # Train
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            history = next_history
            gain = next_gain
            episode_reward += reward
            steps += 1
        
        # Record metrics
        all_rewards.append(episode_reward)
        all_epsilons.append(agent.epsilon)
        if episode_losses:
            all_losses.extend(episode_losses)
        
        # Signal episode end to fuzzy controller
        if use_fuzzy_adaptive:
            agent.end_episode(episode_reward)
            fuzzy_metrics = agent.get_metrics()
            all_fuzzy_metrics.append(fuzzy_metrics)
        
        # Update progress bar
        avg_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
        postfix = {
            'reward': f'${episode_reward:.0f}',
            'avg_50': f'${avg_reward:.0f}',
            'eps': f'{agent.epsilon:.3f}',
            'steps': steps
        }
        
        # Add fuzzy metrics to progress bar
        if use_fuzzy_adaptive and fuzzy_metrics:
            postfix['Q'] = f"{fuzzy_metrics['quality']:.2f}"
            postfix['T'] = f"{fuzzy_metrics['trend']:.2f}"
        
        pbar.set_postfix(postfix)
        
        # Periodic checkpoint save
        if (episode + 1) % 100 == 0:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_checkpoint.pt")
            agent.save(checkpoint_path, extra_data={
                'episode': episode + 1,
                'rewards': all_rewards,
                'losses': all_losses,
                'epsilons': all_epsilons
            })
    
    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    
    extra_save_data = {
        'episode': start_episode + episodes,
        'rewards': all_rewards,
        'losses': all_losses,
        'epsilons': all_epsilons,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'use_fuzzy_adaptive': use_fuzzy_adaptive
        }
    }
    
    if use_fuzzy_adaptive and all_fuzzy_metrics:
        extra_save_data['fuzzy_metrics_history'] = all_fuzzy_metrics
    
    agent.save(model_path, extra_data=extra_save_data)
    
    # Save training plot
    plot_path = os.path.join(save_dir, f"{model_name}_training.png")
    try:
        plot_training_results(
            all_rewards, all_losses, all_epsilons,
            title=f"Training: {model_name}",
            save_path=plot_path,
            show=False
        )
    except Exception as e:
        print(f"⚠️ Could not save plot: {e}")
        plot_path = None
    
    # Compute final statistics
    results = {
        'model_name': model_name,
        'total_episodes': start_episode + episodes,
        'final_epsilon': agent.epsilon,
        'avg_reward_last_100': np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards),
        'avg_reward_last_50': np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards),
        'max_reward': max(all_rewards) if all_rewards else 0,
        'min_reward': min(all_rewards) if all_rewards else 0,
        'std_reward': np.std(all_rewards) if all_rewards else 0,
        'model_path': model_path,
        'plot_path': plot_path,
        'data_source': 'real' if use_real_data else 'simulated',
        'fuzzy_adaptive': use_fuzzy_adaptive
    }
    
    # Add final fuzzy metrics
    if use_fuzzy_adaptive:
        final_fuzzy = agent.get_metrics()
        results['fuzzy_final_metrics'] = final_fuzzy
    
    if verbose:
        print(f"\n✅ Training complete!")
        print(f"   Model saved: {model_path}")
        if plot_path:
            print(f"   Plot saved: {plot_path}")
        print(f"   Total episodes: {results['total_episodes']}")
        print(f"   Avg reward (last 100): ${results['avg_reward_last_100']:.0f}")
        print(f"   Max reward: ${results['max_reward']:.0f}")
        print(f"   Min reward: ${results['min_reward']:.0f}")
        
        if use_fuzzy_adaptive:
            print(f"\n   Fuzzy Adaptive Metrics:")
            print(f"   - Final Quality: {final_fuzzy['quality']:.3f}")
            print(f"   - Final Success Ratio: {final_fuzzy['success_ratio']:.3f}")
            print(f"   - Final Trend: {final_fuzzy['trend']:.3f}")
            print(f"   - Final Diversity: {final_fuzzy['diversity']:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train RL Roulette agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic options
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=500,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="roulette_agent",
        help="Name for saved model"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )
    
    # Data options
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real data from database"
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=None,
        help="Specific session ID to load from database"
    )
    
    # Environment options
    parser.add_argument(
        "--max-spins",
        type=int,
        default=100,
        help="Maximum spins per episode"
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll"
    )
    parser.add_argument(
        "--flat-env",
        action="store_true",
        help="Use flattened observation space"
    )
    
    # Fuzzy Adaptive options
    parser.add_argument(
        "--fuzzy-adaptive", "-f",
        action="store_true",
        help="Use Fuzzy Adaptive DQN (FLC-EA inspired epsilon control)"
    )
    parser.add_argument(
        "--fuzzy-rate",
        type=float,
        default=0.1,
        help="Fuzzy adjustment rate (how quickly epsilon adapts)"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=3e-4,
        dest="learning_rate",
        help="Learning rate"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Starting epsilon"
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="Minimum epsilon"
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.998,
        help="Epsilon decay rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50000,
        help="Replay buffer size"
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=10,
        help="Target network update frequency"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎰 RL Roulette - Training")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load real data if requested
    real_numbers = None
    if args.use_real_data:
        real_numbers = get_real_data(args.session_id)
        if not real_numbers:
            print("⚠️ No real data available, falling back to simulated data")
            args.use_real_data = False
    
    # Train agent
    results = train_agent(
        episodes=args.episodes,
        save_dir=args.save_dir,
        model_name=args.model_name,
        verbose=not args.quiet,
        use_real_data=args.use_real_data,
        real_numbers=real_numbers,
        resume_path=args.resume,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update,
        max_spins=args.max_spins,
        initial_bankroll=args.initial_bankroll,
        use_flat_env=args.flat_env,
        use_fuzzy_adaptive=args.fuzzy_adaptive,
        fuzzy_adjustment_rate=args.fuzzy_rate
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 Training Summary")
    print("="*60)
    print(f"Model: {results['model_name']}")
    print(f"Data source: {results['data_source']}")
    print(f"Fuzzy Adaptive: {'Yes' if results['fuzzy_adaptive'] else 'No'}")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Final epsilon: {results['final_epsilon']:.4f}")
    print(f"Avg Reward (last 100): ${results['avg_reward_last_100']:.0f}")
    print(f"Avg Reward (last 50): ${results['avg_reward_last_50']:.0f}")
    print(f"Max Reward: ${results['max_reward']:.0f}")
    print(f"Min Reward: ${results['min_reward']:.0f}")
    print(f"Std Reward: ${results['std_reward']:.1f}")
    
    # Print fuzzy metrics if available
    if results.get('fuzzy_final_metrics'):
        fm = results['fuzzy_final_metrics']
        print(f"\n📊 Fuzzy Adaptive Final Metrics:")
        print(f"   Quality: {fm['quality']:.3f}")
        print(f"   Success Ratio: {fm['success_ratio']:.3f}")
        print(f"   Trend: {fm['trend']:.3f}")
        print(f"   Diversity: {fm['diversity']:.3f}")
    
    print(f"\nModel saved: {results['model_path']}")
    print("\n✅ Training complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
