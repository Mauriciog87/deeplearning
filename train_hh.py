#!/usr/bin/env python3
"""
Training script for Hyper-Heuristic Agent.

Based on: "A review of reinforcement learning based hyper-heuristics" (Li et al., 2024)

This script trains the HyperHeuristicAgent which uses Q-Learning at the meta-level
to select from multiple Low-Level Heuristics (betting strategies).
"""
import argparse
import os
import time
from datetime import datetime
from typing import Optional
import numpy as np

from src.environment import RouletteEnv, RouletteEnvNearMissFlat
from src.agents import HyperHeuristicAgent, DQNHyperHeuristic, LLHType
from src.utils import WheelBiasAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Hyper-Heuristic Agent for Roulette',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Episodes between evaluations')
    parser.add_argument('--save-interval', type=int, default=200,
                        help='Episodes between model saves')
    
    # Agent parameters
    parser.add_argument('--agent-type', type=str, default='qlearning',
                        choices=['qlearning', 'dqn'],
                        help='Type of hyper-heuristic agent')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon-end', type=float, default=0.05,
                        help='Final exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate per step')
    
    # Environment parameters
    parser.add_argument('--initial-bankroll', type=float, default=1000.0,
                        help='Starting bankroll')
    parser.add_argument('--base-bet', type=float, default=10.0,
                        help='Base bet amount')
    parser.add_argument('--use-near-miss', action='store_true',
                        help='Use near-miss enhanced environment')
    
    # Bias detection
    parser.add_argument('--enable-bias-detection', action='store_true',
                        help='Enable wheel bias detection')
    parser.add_argument('--bias-window', type=int, default=200,
                        help='Window size for bias detection')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--model-name', type=str, default='hh_agent',
                        help='Base name for saved models')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=silent, 1=normal, 2=detailed)')
    
    return parser.parse_args()


def create_agent(args) -> HyperHeuristicAgent:
    """Create the hyper-heuristic agent based on arguments."""
    agent_kwargs = {
        'learning_rate': args.lr,
        'discount_factor': args.gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_end': args.epsilon_end,
        'epsilon_decay': args.epsilon_decay,
        'initial_bankroll': args.initial_bankroll,
        'base_bet': args.base_bet,
        'seed': args.seed,
    }
    
    if args.agent_type == 'dqn':
        return DQNHyperHeuristic(**agent_kwargs)
    else:
        return HyperHeuristicAgent(**agent_kwargs)


def train_episode(
    agent: HyperHeuristicAgent,
    env,
    max_steps: int,
    bias_analyzer: Optional[WheelBiasAnalyzer] = None
) -> dict:
    """Train for one episode."""
    obs, info = env.reset()
    agent.reset_episode()
    
    episode_reward = 0.0
    episode_wins = 0
    episode_losses = 0
    llh_usage = {llh: 0 for llh in LLHType}
    
    for step in range(max_steps):
        # Select LLH and action
        agent.select_llh(explore=True)
        action, bet_amount = agent.select_action(explore=True)
        
        # Track LLH usage
        if agent.current_llh:
            llh_usage[agent.current_llh] += 1
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get outcome (winning number)
        outcome = info.get('winning_number', np.random.randint(0, 37))
        
        # Update bias detector
        if bias_analyzer:
            bias_analyzer.add_spin(outcome)
            if bias_analyzer.total_spins % 50 == 0:
                report = bias_analyzer.get_report()
                if report is not None:
                    agent.set_bias_detected(
                        report.overall_bias.value != 'none',
                        bias_analyzer.get_hot_numbers()[:3]
                    )
        
        # Update agent
        agent.update(outcome, action, reward)
        
        episode_reward += reward
        if reward > 0:
            episode_wins += 1
        elif reward < 0:
            episode_losses += 1
        
        # Check termination
        if terminated or truncated:
            break
        
        # Check bankruptcy
        if agent.bankroll <= 0:
            break
    
    return {
        'reward': episode_reward,
        'wins': episode_wins,
        'losses': episode_losses,
        'final_bankroll': agent.bankroll,
        'steps': step + 1,
        'llh_usage': llh_usage,
    }


def evaluate_agent(
    agent: HyperHeuristicAgent,
    env,
    num_episodes: int = 10,
    max_steps: int = 500
) -> dict:
    """Evaluate agent without exploration."""
    total_reward = 0.0
    total_wins = 0
    total_losses = 0
    final_bankrolls = []
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during eval
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        agent.reset_episode()
        
        for step in range(max_steps):
            agent.select_llh(explore=False)
            action, _ = agent.select_action(explore=False)
            
            obs, reward, terminated, truncated, info = env.step(action)
            outcome = info.get('winning_number', np.random.randint(0, 37))
            
            # Update without learning
            agent.spin_history.append(outcome)
            agent.reward_history.append(reward)
            agent.bankroll += reward
            
            total_reward += reward
            if reward > 0:
                total_wins += 1
            elif reward < 0:
                total_losses += 1
            
            if terminated or truncated or agent.bankroll <= 0:
                break
        
        final_bankrolls.append(agent.bankroll)
    
    agent.epsilon = original_epsilon
    
    return {
        'avg_reward': total_reward / num_episodes,
        'avg_final_bankroll': np.mean(final_bankrolls),
        'std_final_bankroll': np.std(final_bankrolls),
        'win_rate': total_wins / max(1, total_wins + total_losses),
    }


def print_llh_statistics(agent: HyperHeuristicAgent):
    """Print detailed LLH usage statistics."""
    print("\n📊 Low-Level Heuristic Statistics:")
    print("-" * 60)
    
    stats = agent.get_statistics()['llh_statistics']
    
    # Sort by usage
    sorted_llhs = sorted(stats.items(), key=lambda x: x[1]['uses'], reverse=True)
    
    for llh_name, llh_stats in sorted_llhs:
        print(f"  {llh_name:20s} | Uses: {llh_stats['uses']:5d} | "
              f"Avg Reward: {llh_stats['avg_reward']:+7.2f} | "
              f"Win Rate: {llh_stats['win_rate']*100:5.1f}%")


def main():
    args = parse_args()
    
    # Set random seeds
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    if args.use_near_miss:
        env = RouletteEnvNearMissFlat(initial_bankroll=args.initial_bankroll)
    else:
        env = RouletteEnv(initial_bankroll=args.initial_bankroll)
    
    # Create agent
    agent = create_agent(args)
    
    # Create bias analyzer if enabled
    bias_analyzer = None
    if args.enable_bias_detection:
        bias_analyzer = WheelBiasAnalyzer(min_spins_for_analysis=args.bias_window)
    
    print("=" * 70)
    print("🎰 Hyper-Heuristic Agent Training")
    print("=" * 70)
    print(f"  Agent Type:      {args.agent_type.upper()}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Max Steps:       {args.max_steps}")
    print(f"  Learning Rate:   {args.lr}")
    print(f"  Discount Factor: {args.gamma}")
    print(f"  Initial ε:       {args.epsilon_start}")
    print(f"  Final ε:         {args.epsilon_end}")
    print(f"  Bankroll:        ${args.initial_bankroll:.2f}")
    print(f"  Base Bet:        ${args.base_bet:.2f}")
    print(f"  Bias Detection:  {'Enabled' if args.enable_bias_detection else 'Disabled'}")
    print(f"  Near-Miss Env:   {'Yes' if args.use_near_miss else 'No'}")
    print("=" * 70)
    
    # Training metrics
    all_rewards = []
    all_bankrolls = []
    best_eval_reward = float('-inf')
    
    start_time = time.time()
    
    for episode in range(1, args.episodes + 1):
        # Train one episode
        result = train_episode(agent, env, args.max_steps, bias_analyzer)
        
        all_rewards.append(result['reward'])
        all_bankrolls.append(result['final_bankroll'])
        
        # Logging
        if args.verbose >= 1 and episode % 10 == 0:
            recent_rewards = all_rewards[-100:]
            recent_bankrolls = all_bankrolls[-100:]
            
            print(f"Episode {episode:5d} | "
                  f"Reward: {result['reward']:+8.2f} | "
                  f"Bankroll: ${result['final_bankroll']:8.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Avg(100): {np.mean(recent_rewards):+7.2f} | "
                  f"Best LLH: {max(result['llh_usage'], key=result['llh_usage'].get).name}")
        
        # Evaluation
        if episode % args.eval_interval == 0:
            eval_result = evaluate_agent(agent, env, num_episodes=20, max_steps=args.max_steps)
            
            print(f"\n📈 Evaluation at Episode {episode}:")
            print(f"   Avg Reward:      {eval_result['avg_reward']:+.2f}")
            print(f"   Avg Bankroll:    ${eval_result['avg_final_bankroll']:.2f} (±{eval_result['std_final_bankroll']:.2f})")
            print(f"   Win Rate:        {eval_result['win_rate']*100:.1f}%")
            print(f"   Q-Table Size:    {len(agent.q_table)} states")
            
            if args.verbose >= 2:
                print_llh_statistics(agent)
            
            print()
            
            # Save best model
            if eval_result['avg_reward'] > best_eval_reward:
                best_eval_reward = eval_result['avg_reward']
                best_path = os.path.join(args.output_dir, f'{args.model_name}_best.pkl')
                agent.save(best_path)
                print(f"   💾 New best model saved: {best_path}")
        
        # Periodic save
        if episode % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'{args.model_name}_ep{episode}.pkl')
            agent.save(checkpoint_path)
            if args.verbose >= 1:
                print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    # Training complete
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🏁 Training Complete!")
    print("=" * 70)
    print(f"  Total Time:       {elapsed_time/60:.1f} minutes")
    print(f"  Total Episodes:   {args.episodes}")
    print(f"  Total Steps:      {agent.total_steps}")
    print(f"  Final ε:          {agent.epsilon:.4f}")
    print(f"  Q-Table States:   {len(agent.q_table)}")
    
    # Final statistics
    stats = agent.get_statistics()
    print(f"\n  Overall Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"  Total Wins:       {stats['wins']}")
    print(f"  Total Losses:     {stats['losses']}")
    
    print_llh_statistics(agent)
    
    # Save final model
    final_path = os.path.join(args.output_dir, f'{args.model_name}_final.pkl')
    agent.save(final_path)
    print(f"\n💾 Final model saved: {final_path}")
    
    # Show best LLH for different states
    print("\n🎯 Best LLH by State:")
    print("-" * 60)
    
    from src.agents.hyper_heuristic import HHState
    
    test_states = [
        HHState(bankroll_level=0, trend=-1, volatility=1, recent_llh_performance=0, bias_detected=0),
        HHState(bankroll_level=1, trend=0, volatility=0, recent_llh_performance=1, bias_detected=0),
        HHState(bankroll_level=2, trend=1, volatility=0, recent_llh_performance=2, bias_detected=0),
        HHState(bankroll_level=1, trend=-1, volatility=1, recent_llh_performance=0, bias_detected=1),
    ]
    
    state_names = [
        "Low bankroll, losing, high volatility",
        "Medium bankroll, neutral, stable",
        "High bankroll, winning, stable",
        "Medium bankroll, losing, bias detected",
    ]
    
    for state, name in zip(test_states, state_names):
        best_llh = agent.get_best_llh_for_state(state)
        q_values = agent.q_table[state.to_tuple()]
        best_q = q_values[best_llh]
        print(f"  {name:45s} → {best_llh.name} (Q={best_q:.2f})")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
