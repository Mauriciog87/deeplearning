#!/usr/bin/env python3
"""
Evaluation script for Hyper-Heuristic Agent.

Tests a trained HH agent and provides detailed performance analysis.
"""
import argparse
import os
import numpy as np
from collections import defaultdict

from src.environment import RouletteEnv, RouletteEnvNearMissFlat
from src.agents import HyperHeuristicAgent, DQNHyperHeuristic, LLHType


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Hyper-Heuristic Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='models/hh_agent_final.json',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of test episodes')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--initial-bankroll', type=float, default=1000.0,
                        help='Starting bankroll')
    parser.add_argument('--base-bet', type=float, default=1.0,
                        help='Base bet amount')
    parser.add_argument('--use-near-miss', action='store_true',
                        help='Use near-miss enhanced environment')
    parser.add_argument('--compare-random', action='store_true',
                        help='Compare with random agent')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    
    return parser.parse_args()


def run_episode(agent, env, max_steps, explore=False, seed=42):
    """Run a single episode and return metrics."""
    obs, info = env.reset(seed=seed)
    agent.reset_episode()
    
    total_reward = 0.0
    wins = 0
    losses = 0
    llh_usage = defaultdict(int)
    llh_rewards = defaultdict(list)
    
    for step in range(max_steps):
        llh = agent.select_llh(explore=explore)
        action, bet = agent.select_action(explore=explore)
        
        llh_usage[llh] += 1
        
        obs, reward, term, trunc, info = env.step(action, stake=bet)
        outcome = info['winning_number']
        
        llh_rewards[llh].append(reward)
        
        # Update agent state (but no learning during test)
        agent.update(outcome, action, reward, learn=False, terminated=term, truncated=trunc)
        
        total_reward += reward
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        
        if term or trunc or agent.bankroll <= 0:
            break
    
    return {
        'reward': total_reward,
        'wins': wins,
        'losses': losses,
        'final_bankroll': agent.bankroll,
        'steps': step + 1,
        'llh_usage': dict(llh_usage),
        'llh_rewards': {k: sum(v) for k, v in llh_rewards.items()},
        'bankrupt': agent.bankroll <= 0,
    }


def run_random_episode(env, max_steps, initial_bankroll, base_bet, seed=42):
    """Run episode with random action selection for comparison."""
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    bankroll = initial_bankroll
    total_reward = 0.0
    wins = 0
    losses = 0
    
    for step in range(max_steps):
        action = int(rng.choice(np.flatnonzero(env.get_action_mask())))
        
        obs, reward, term, trunc, info = env.step(action)
        bankroll += reward
        total_reward += reward
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        
        if term or trunc or bankroll <= 0:
            break
    
    return {
        'reward': total_reward,
        'wins': wins,
        'losses': losses,
        'final_bankroll': bankroll,
        'steps': step + 1,
        'bankrupt': bankroll <= 0,
    }


def print_llh_analysis(all_results):
    """Print detailed LLH usage analysis."""
    print("\n📊 LLH Usage Analysis Across All Episodes:")
    print("-" * 70)
    
    total_usage = defaultdict(int)
    total_rewards = defaultdict(float)
    
    for result in all_results:
        for llh, count in result['llh_usage'].items():
            total_usage[llh] += count
        for llh, reward in result['llh_rewards'].items():
            total_rewards[llh] += reward
    
    sorted_llhs = sorted(total_usage.items(), key=lambda x: x[1], reverse=True)
    
    for llh, count in sorted_llhs:
        avg_reward = total_rewards[llh] / max(1, count)
        total_reward = total_rewards[llh]
        pct = count / sum(total_usage.values()) * 100
        print(f"  {llh.name:20s} | Uses: {count:6d} ({pct:5.1f}%) | "
              f"Total Reward: {total_reward:+10.2f} | Avg: {avg_reward:+7.2f}")


def main():
    from src.console import configure_console
    configure_console()
    args = parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Load agent
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train an agent first with: python train_hh.py")
        return
    
    agent_class = DQNHyperHeuristic if args.model.endswith('.pt') else HyperHeuristicAgent
    agent = agent_class.load(
        args.model,
        initial_bankroll=args.initial_bankroll,
        base_bet=args.base_bet
    )
    agent.initial_bankroll = args.initial_bankroll
    agent.base_bet = args.base_bet
    
    # Create environment
    if args.use_near_miss:
        env = RouletteEnvNearMissFlat(initial_bankroll=args.initial_bankroll, bet_size=args.base_bet, max_steps=args.max_steps)
    else:
        env = RouletteEnv(initial_bankroll=args.initial_bankroll, bet_size=args.base_bet, max_steps=args.max_steps)
    
    print("=" * 70)
    print("🎰 Hyper-Heuristic Agent Evaluation")
    print("=" * 70)
    print(f"  Model:            {args.model}")
    print(f"  Episodes:         {args.episodes}")
    print(f"  Max Steps:        {args.max_steps}")
    print(f"  Initial Bankroll: ${args.initial_bankroll:.2f}")
    print(f"  Base Bet:         ${args.base_bet:.2f}")
    print(f"  Q-Table States:   {len(agent.q_table)}")
    print("=" * 70)
    
    # Run evaluation
    all_results = []
    
    for ep in range(1, args.episodes + 1):
        result = run_episode(agent, env, args.max_steps, explore=False, seed=args.seed + ep)
        all_results.append(result)
        
        if args.verbose >= 2:
            print(f"Episode {ep:4d} | Reward: {result['reward']:+8.2f} | "
                  f"Bankroll: ${result['final_bankroll']:8.2f} | "
                  f"{'💸 BANKRUPT' if result['bankrupt'] else ''}")
        elif args.verbose >= 1 and ep % 10 == 0:
            recent = all_results[-10:]
            avg_reward = np.mean([r['reward'] for r in recent])
            avg_bankroll = np.mean([r['final_bankroll'] for r in recent])
            bankrupts = sum(1 for r in recent if r['bankrupt'])
            print(f"Episodes {ep-9:4d}-{ep:4d} | Avg Reward: {avg_reward:+8.2f} | "
                  f"Avg Bankroll: ${avg_bankroll:8.2f} | Bankrupts: {bankrupts}")
    
    # Summary statistics
    rewards = [r['reward'] for r in all_results]
    bankrolls = [r['final_bankroll'] for r in all_results]
    bankrupts = sum(1 for r in all_results if r['bankrupt'])
    total_wins = sum(r['wins'] for r in all_results)
    total_losses = sum(r['losses'] for r in all_results)
    
    print("\n" + "=" * 70)
    print("📈 HH Agent Results:")
    print("-" * 70)
    print(f"  Episodes Run:     {args.episodes}")
    print(f"  Avg Reward:       {np.mean(rewards):+.2f} (±{np.std(rewards):.2f})")
    print(f"  Avg Bankroll:     ${np.mean(bankrolls):.2f} (±{np.std(bankrolls):.2f})")
    print(f"  Min Bankroll:     ${np.min(bankrolls):.2f}")
    print(f"  Max Bankroll:     ${np.max(bankrolls):.2f}")
    print(f"  Bankruptcy Rate:  {bankrupts}/{args.episodes} ({bankrupts/args.episodes*100:.1f}%)")
    print(f"  Win Rate:         {total_wins/(total_wins+total_losses)*100:.1f}%")
    print(f"  Profitable Eps:   {sum(1 for r in rewards if r > 0)}/{args.episodes}")
    
    print_llh_analysis(all_results)
    
    # Compare with random if requested
    if args.compare_random:
        print("\n" + "=" * 70)
        print("🎲 Random Agent Comparison:")
        print("-" * 70)
        
        random_results = []
        for ep in range(1, args.episodes + 1):
            result = run_random_episode(env, args.max_steps, args.initial_bankroll, args.base_bet, seed=args.seed + ep)
            random_results.append(result)
        
        random_rewards = [r['reward'] for r in random_results]
        random_bankrolls = [r['final_bankroll'] for r in random_results]
        random_bankrupts = sum(1 for r in random_results if r['bankrupt'])
        random_wins = sum(r['wins'] for r in random_results)
        random_losses = sum(r['losses'] for r in random_results)
        
        print(f"  Avg Reward:       {np.mean(random_rewards):+.2f} (±{np.std(random_rewards):.2f})")
        print(f"  Avg Bankroll:     ${np.mean(random_bankrolls):.2f} (±{np.std(random_bankrolls):.2f})")
        print(f"  Bankruptcy Rate:  {random_bankrupts}/{args.episodes} ({random_bankrupts/args.episodes*100:.1f}%)")
        print(f"  Win Rate:         {random_wins/(random_wins+random_losses)*100:.1f}%")
        
        print("\n📊 Comparison Summary:")
        print("-" * 70)
        reward_diff = np.mean(rewards) - np.mean(random_rewards)
        bankroll_diff = np.mean(bankrolls) - np.mean(random_bankrolls)
        
        print(f"  Reward Advantage:    {reward_diff:+.2f} ({'✅ HH Better' if reward_diff > 0 else '❌ Random Better'})")
        print(f"  Bankroll Advantage:  ${bankroll_diff:+.2f} ({'✅ HH Better' if bankroll_diff > 0 else '❌ Random Better'})")
        print(f"  Bankruptcy Diff:     {random_bankrupts - bankrupts:+d} less bankruptcies for HH")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
