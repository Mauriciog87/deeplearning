"""
Testing/Evaluation script for RL Roulette agents.

Supports evaluation with simulated data or real roulette data from SQLite database.

Usage:
    # Test with simulated random data
    python test.py --episodes 100
    
    # Test with real data from database
    python test.py --use-real-data --episodes 100
    
    # Test specific model
    python test.py --model models/roulette_agent.pt --episodes 100
    
    # Compare trained agent vs random agent
    python test.py --compare-random --episodes 100
"""

import argparse
import os
import sys
from datetime import datetime
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tqdm import tqdm

from src.environment.roulette_env import RouletteEnv, RouletteEnvFlat
from src.agents.dqn_agent import DQNAgent
from src.utils.visualization import plot_comparison, plot_episode_details


from train import get_real_data
from src.datasets import held_out_segments
from src.checkpoints import seed_everything


def test_agent(
    episodes: int = 100,
    model_path: str = None,
    verbose: bool = True,
    use_real_data: bool = False,
    real_numbers: list = None,
    max_spins: int = 100,
    initial_bankroll: float = 1000.0,
    use_flat_env: bool = False,
    show_episode: bool = False,
    agent_name: str = "Trained Agent",
    seed: int = 42,
    device: str = "auto"
) -> dict:
    """
    Test a trained DQN agent.
    
    Args:
        episodes: Number of test episodes
        model_path: Path to saved model
        verbose: Whether to print progress
        use_real_data: Use real data from database
        real_numbers: Pre-loaded real numbers
        max_spins: Max spins per episode
        initial_bankroll: Starting bankroll
        use_flat_env: Use flattened observation space
        show_episode: Show sample episode plot
        agent_name: Name for display
    
    Returns:
        Dictionary with test results
    """
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    device = seed_everything(seed, device)
    # Create environment
    if use_flat_env:
        env = RouletteEnvFlat(
            history_size=20,
            max_steps=max_spins,
            initial_bankroll=initial_bankroll
        )
    else:
        env = RouletteEnv(
            history_size=20,
            max_steps=max_spins,
            initial_bankroll=initial_bankroll
        )
    
    
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        history_size=20,
        embedding_dim=64,
        action_size=action_size,
        device=device
    )
    
    checkpoint = agent.load(model_path)
    partition = checkpoint.get('training_config', {}).get('partition')
    segments = []
    if use_real_data:
        if partition is None:
            raise ValueError('Checkpoint has no reserved real-data test partition')
        segments = held_out_segments(real_numbers, partition)
        episodes = len(segments)
    env.bet_size = checkpoint.get('training_config', {}).get('unit_stake', env.bet_size)

    # Test metrics
    all_rewards = []
    all_steps = []
    wins = 0
    losses = 0
    draws = 0
    
    # Track action distribution
    action_counts = Counter()
    
    # For sample episode visualization
    sample_rewards = []
    sample_bankrolls = []
    sample_actions = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing: {agent_name}")
        print(f"{'='*60}")
        print(f"Data source: {'Real data' if use_real_data else 'Simulated'}")
        if use_real_data and real_numbers:
            print(f"Real data spins: {len(real_numbers)}")
        print(f"Episodes: {episodes}")
    
    pbar = tqdm(range(episodes), desc=f"Testing", disable=not verbose)
    
    for episode in pbar:
        if use_real_data:
            env.set_real_data(segments[episode])
        obs, info = env.reset(seed=seed + episode)
        
        # Extract history and gain
        if use_flat_env:
            history = obs[:20].astype(np.int64)
            gain = obs[20]
        else:
            history = obs["history"]
            gain = obs["gain"][0]
        
        episode_reward = 0
        step = 0
        done = False
        
        # Track first episode
        if episode == 0:
            sample_bankrolls.append(info['bankroll'])
        
        while not done:
            action = agent.act(history, gain, training=False, action_mask=env.get_action_mask())
            action_counts[action] += 1
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if use_flat_env:
                history = next_obs[:20].astype(np.int64)
                gain = next_obs[20]
            else:
                history = next_obs["history"]
                gain = next_obs["gain"][0]
            
            episode_reward += reward
            step += 1
            
            if episode == 0:
                sample_rewards.append(reward)
                sample_bankrolls.append(info['bankroll'])
                sample_actions.append(info.get('action_name', f'action_{action}'))
        
        all_rewards.append(episode_reward)
        all_steps.append(step)
        
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
        
        pbar.set_postfix({
            'reward': f'${episode_reward:.0f}',
            'win_rate': f'{wins/(episode+1)*100:.1f}%'
        })
    
    # Calculate results
    results = {
        'partition': partition,
        'unit_stake': env.bet_size,
        'agent_name': agent_name,
        'episodes': episodes,
        'total_reward': sum(all_rewards),
        'avg_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'max_reward': max(all_rewards),
        'min_reward': min(all_rewards),
        'median_reward': np.median(all_rewards),
        'win_rate': wins / episodes,
        'loss_rate': losses / episodes,
        'draw_rate': draws / episodes,
        'avg_steps': np.mean(all_steps),
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'action_distribution': dict(action_counts.most_common(10)),
        'data_source': 'real' if use_real_data else 'simulated'
    }
    
    if verbose:
        print(f"\n📊 Results for {agent_name}:")
        print(f"   Total Reward: ${results['total_reward']:,.0f}")
        print(f"   Avg Reward: ${results['avg_reward']:,.0f} ± ${results['std_reward']:,.0f}")
        print(f"   Median Reward: ${results['median_reward']:,.0f}")
        print(f"   Win Rate: {results['win_rate']*100:.1f}% ({wins}/{episodes})")
        print(f"   Loss Rate: {results['loss_rate']*100:.1f}% ({losses}/{episodes})")
        print(f"   Avg Steps: {results['avg_steps']:.1f}")
        print(f"\n   Top actions: {results['action_distribution']}")
    
    # Show sample episode
    if show_episode and sample_rewards:
        try:
            plot_episode_details(
                sample_rewards, sample_bankrolls, sample_actions,
                title=f"Sample Episode: {agent_name}",
                show=True
            )
        except Exception as e:
            print(f"⚠️ Could not plot episode: {e}")
    
    return results


def test_random_agent(
    episodes: int = 100,
    use_real_data: bool = False,
    real_numbers: list = None,
    max_spins: int = 100,
    initial_bankroll: float = 1000.0,
    verbose: bool = True,
    seed: int = 42,
    partition=None,
    unit_stake: float = 50.0
) -> dict:
    """Test a random agent as baseline."""
    env = RouletteEnv(
        history_size=20,
        max_steps=max_spins,
        initial_bankroll=initial_bankroll
    )
    
    segments = []
    env.bet_size = unit_stake
    if use_real_data:
        if partition is None:
            raise ValueError('Random comparison requires the same held-out partition')
        segments = held_out_segments(real_numbers, partition)
        episodes = len(segments)
    
    all_rewards = []
    wins = 0
    losses = 0
    
    pbar = tqdm(range(episodes), desc="Testing Random", disable=not verbose)
    
    for episode in pbar:
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample(mask=env.get_action_mask().astype(np.int8))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        all_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        
        pbar.set_postfix({
            'reward': f'${episode_reward:.0f}',
            'win_rate': f'{wins/(episode+1)*100:.1f}%'
        })
    
    return {
        'agent_name': 'Random Agent',
        'episodes': episodes,
        'total_reward': sum(all_rewards),
        'avg_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'win_rate': wins / episodes,
        'wins': wins,
        'losses': losses
    }


def analyze_real_data():
    """Analyze real data from database."""
    try:
        from src.database import RouletteRepository
        from src.utils.analysis import full_analysis
        
        repo = RouletteRepository()
        numbers = repo.get_all_numbers()
        
        if not numbers:
            print("No real data in database")
            return
        
        print("\n" + "="*60)
        print("📊 Real Data Analysis")
        print("="*60)
        
        analysis = full_analysis(numbers)
        
        print(f"\nTotal spins: {analysis['total_spins']}")
        print(f"Unique numbers seen: {analysis['unique_numbers_seen']}")
        
        print(f"\nColors:")
        print(f"  Red: {analysis['colors']['red_pct']:.1f}%")
        print(f"  Black: {analysis['colors']['black_pct']:.1f}%")
        print(f"  Green: {analysis['colors']['green_pct']:.1f}%")
        
        print(f"\nChi-square test:")
        chi = analysis['chi_square']
        print(f"  Chi-square: {chi['chi_square']:.2f}")
        print(f"  Is uniform (95%): {'Yes' if chi['is_uniform_95'] else 'No'}")
        
        print(f"\nMost common: {analysis['most_common']}")
        print(f"Least common: {analysis['least_common']}")
        
        print(f"\nStreaks:")
        streaks = analysis['streaks']
        print(f"  Max repeat: {streaks['max_repeat_streak']}")
        print(f"  Max color streak: {streaks['max_color_streak']}")
        print(f"  Max dozen streak: {streaks['max_dozen_streak']}")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")


def main():
    from src.console import configure_console
    configure_console()
    parser = argparse.ArgumentParser(
        description="Test RL Roulette agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=100,
        help="Number of test episodes"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/roulette_agent.pt",
        help="Path to saved model"
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real data from database"
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=None,
        help="Specific session ID to load"
    )
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="Compare against random agent"
    )
    parser.add_argument(
        "--analyze-data",
        action="store_true",
        help="Analyze real data statistics"
    )
    parser.add_argument(
        "--show-episode",
        action="store_true",
        help="Show sample episode plot"
    )
    parser.add_argument(
        "--max-spins",
        type=int,
        default=100,
        help="Max spins per episode"
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
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎰 RL Roulette - Testing")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze real data if requested
    if args.analyze_data:
        analyze_real_data()
        return
    
    # Load real data if requested
    real_numbers = None
    if args.use_real_data:
        real_numbers = get_real_data(args.session_id)
        print(f"Loaded {len(real_numbers)} real sessions")
    
    all_results = {}
    
    # Test trained agent
    results = test_agent(
        episodes=args.episodes,
        model_path=args.model,
        verbose=not args.quiet,
        use_real_data=args.use_real_data,
        real_numbers=real_numbers,
        max_spins=args.max_spins,
        initial_bankroll=args.initial_bankroll,
        use_flat_env=args.flat_env,
        show_episode=args.show_episode,
        agent_name="Trained Agent", seed=args.seed, device=args.device
    )
    all_results['trained'] = results
    
    # Compare with random agent if requested
    if args.compare_random:
        random_results = test_random_agent(
            episodes=args.episodes,
            use_real_data=args.use_real_data,
            real_numbers=real_numbers,
            max_spins=args.max_spins,
            initial_bankroll=args.initial_bankroll,
            verbose=not args.quiet, seed=args.seed,
            partition=results.get('partition'), unit_stake=results['unit_stake']
        )
        all_results['random'] = random_results
        
        # Print comparison
        print("\n" + "="*60)
        print("📊 Comparison: Trained vs Random")
        print("="*60)
        print(f"\n{'Agent':<20} {'Total $':>12} {'Avg $':>12} {'Win Rate':>12}")
        print("-" * 58)
        
        for key, r in all_results.items():
            print(f"{r['agent_name']:<20} ${r['total_reward']:>10,.0f} ${r['avg_reward']:>10,.0f} {r['win_rate']*100:>10.1f}%")
        
        # Calculate improvement
        trained_avg = all_results['trained']['avg_reward']
        random_avg = all_results['random']['avg_reward']
        improvement = trained_avg - random_avg
        
        print(f"\n🎯 Trained agent improvement: ${improvement:,.0f} per episode")
        
        if improvement > 0:
            print("   ✅ Trained agent outperforms random!")
        elif improvement < -10:
            print("   ⚠️ Random agent is doing better - may need more training")
        else:
            print("   ➡️ Performance similar to random")
    
    print("\n✅ Testing complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
