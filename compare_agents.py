#!/usr/bin/env python3
"""
Compare all agent types in the RL Roulette project.

Agents compared:
1. Random Agent (baseline)
2. Behavioral Agents (Gambler's Fallacy, Hot Hand, Mixed)
3. DQN Agent (if trained model exists)
4. Fuzzy Adaptive DQN (if trained model exists)
5. Hyper-Heuristic Agent (if trained model exists)
"""
import argparse
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from src.environment import RouletteEnv
from src.agents import (
    GamblersFallacyAgent, HotHandAgent, MixedBehaviorAgent, 
    RandomAgent, BetType
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare all agent types',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes per agent')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--initial-bankroll', type=float, default=1000.0,
                        help='Starting bankroll')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model paths
    parser.add_argument('--dqn-model', type=str, default='models/roulette_agent.pt',
                        help='Path to DQN model')
    parser.add_argument('--fuzzy-model', type=str, default='models/fuzzy_agent.pt',
                        help='Path to Fuzzy DQN model')
    parser.add_argument('--hh-model', type=str, default='models/hh_agent_best.pkl',
                        help='Path to HH agent model')
    
    return parser.parse_args()


class AgentWrapper:
    """Wrapper to provide unified interface for all agent types."""
    
    def __init__(self, name: str, agent_type: str, agent=None, **kwargs):
        self.name = name
        self.agent_type = agent_type
        self.agent = agent
        self.kwargs = kwargs
        self.bankroll = kwargs.get('initial_bankroll', 1000.0)
        self.initial_bankroll = self.bankroll
    
    def reset(self):
        self.bankroll = self.initial_bankroll
        if self.agent_type == 'hh' and self.agent:
            self.agent.reset_episode()
    
    def select_action(self, obs, history: List[int]) -> int:
        if self.agent_type == 'random':
            return np.random.randint(0, 47)
        
        elif self.agent_type == 'behavioral':
            return self.agent.select_action(
                history, 
                self.bankroll, 
                history[-1] if history else None
            )
        
        elif self.agent_type == 'dqn':
            import torch
            with torch.no_grad():
                return self.agent.select_action(obs, explore=False)
        
        elif self.agent_type == 'hh':
            self.agent.select_llh(explore=False)
            action, _ = self.agent.select_action(explore=False)
            return action
        
        return np.random.randint(0, 47)
    
    def update(self, outcome: int, action: int, reward: float):
        self.bankroll += reward
        if self.agent_type == 'hh' and self.agent:
            self.agent.spin_history.append(outcome)
            self.agent.reward_history.append(reward)
            self.agent.bankroll = self.bankroll


def create_agents(args) -> List[AgentWrapper]:
    """Create all available agents."""
    agents = []
    
    # 1. Random Agent (baseline)
    agents.append(AgentWrapper(
        "Random", "random",
        initial_bankroll=args.initial_bankroll
    ))
    
    # 2. Behavioral Agents
    agents.append(AgentWrapper(
        "Gambler's Fallacy", "behavioral",
        agent=GamblersFallacyAgent(bet_type=BetType.STRAIGHT, seed=args.seed),
        initial_bankroll=args.initial_bankroll
    ))
    
    agents.append(AgentWrapper(
        "Hot Hand", "behavioral",
        agent=HotHandAgent(bet_type=BetType.STRAIGHT, seed=args.seed),
        initial_bankroll=args.initial_bankroll
    ))
    
    agents.append(AgentWrapper(
        "Mixed Behavioral", "behavioral",
        agent=MixedBehaviorAgent(bet_type=BetType.STRAIGHT, seed=args.seed),
        initial_bankroll=args.initial_bankroll
    ))
    
    # 3. DQN Agent (if exists)
    if os.path.exists(args.dqn_model):
        try:
            import torch
            from src.agents import DQNAgent
            
            dqn = DQNAgent(
                history_size=20,
                action_size=47,
                hidden_size=128
            )
            checkpoint = torch.load(args.dqn_model, map_location='cpu')
            if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
                dqn.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            else:
                dqn.q_network.load_state_dict(checkpoint)
            dqn.q_network.eval()
            
            agents.append(AgentWrapper(
                "DQN Agent", "dqn",
                agent=dqn,
                initial_bankroll=args.initial_bankroll
            ))
            print(f"✅ Loaded DQN model: {args.dqn_model}")
        except Exception as e:
            print(f"⚠️ Could not load DQN model: {e}")
    
    # 4. Fuzzy DQN Agent (if exists)
    if os.path.exists(args.fuzzy_model):
        try:
            import torch
            from src.agents import FuzzyAdaptiveDQN
            
            fuzzy = FuzzyAdaptiveDQN(
                history_size=20,
                action_size=47,
                hidden_size=128
            )
            checkpoint = torch.load(args.fuzzy_model, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                fuzzy.q_network.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
                fuzzy.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            else:
                fuzzy.q_network.load_state_dict(checkpoint)
            fuzzy.q_network.eval()
            
            agents.append(AgentWrapper(
                "Fuzzy DQN", "dqn",
                agent=fuzzy,
                initial_bankroll=args.initial_bankroll
            ))
            print(f"✅ Loaded Fuzzy DQN model: {args.fuzzy_model}")
        except Exception as e:
            print(f"⚠️ Could not load Fuzzy DQN model: {e}")
    
    # 5. Hyper-Heuristic Agent (if exists)
    if os.path.exists(args.hh_model):
        try:
            from src.agents import HyperHeuristicAgent
            
            hh_agent = HyperHeuristicAgent.load(
                args.hh_model,
                initial_bankroll=args.initial_bankroll
            )
            hh_agent.epsilon = 0.0
            
            agents.append(AgentWrapper(
                "Hyper-Heuristic", "hh",
                agent=hh_agent,
                initial_bankroll=args.initial_bankroll
            ))
            print(f"✅ Loaded HH model: {args.hh_model}")
        except Exception as e:
            print(f"⚠️ Could not load HH model: {e}")
    
    return agents


def run_episode(agent: AgentWrapper, env, max_steps: int) -> Dict[str, Any]:
    """Run a single episode for an agent."""
    obs, info = env.reset()
    agent.reset()
    
    history = list(obs['history']) if isinstance(obs, dict) else []
    total_reward = 0.0
    wins = 0
    losses = 0
    
    for step in range(max_steps):
        action = agent.select_action(obs, history)
        
        obs, reward, term, trunc, info = env.step(action)
        outcome = info.get('winning_number', np.random.randint(0, 37))
        
        agent.update(outcome, action, reward)
        
        history.append(outcome)
        if len(history) > 20:
            history = history[-20:]
        
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
        'bankrupt': agent.bankroll <= 0,
    }


def run_comparison(agents: List[AgentWrapper], args) -> Dict[str, List[Dict]]:
    """Run comparison across all agents."""
    env = RouletteEnv(initial_bankroll=args.initial_bankroll)
    
    results = {agent.name: [] for agent in agents}
    
    for ep in range(1, args.episodes + 1):
        if ep % 20 == 0:
            print(f"  Running episode {ep}/{args.episodes}...")
        
        for agent in agents:
            result = run_episode(agent, env, args.max_steps)
            results[agent.name].append(result)
    
    return results


def print_results(results: Dict[str, List[Dict]], args):
    """Print comparison results in a nice table."""
    print("\n" + "=" * 90)
    print("📊 AGENT COMPARISON RESULTS")
    print("=" * 90)
    print(f"Episodes: {args.episodes} | Max Steps: {args.max_steps} | Initial Bankroll: ${args.initial_bankroll}")
    print("-" * 90)
    
    # Header
    print(f"{'Agent':<20} | {'Avg Reward':>12} | {'Avg Bankroll':>14} | "
          f"{'Win Rate':>10} | {'Bankrupt':>10} | {'Profitable':>10}")
    print("-" * 90)
    
    # Calculate stats for each agent
    agent_stats = []
    
    for agent_name, agent_results in results.items():
        rewards = [r['reward'] for r in agent_results]
        bankrolls = [r['final_bankroll'] for r in agent_results]
        total_wins = sum(r['wins'] for r in agent_results)
        total_losses = sum(r['losses'] for r in agent_results)
        bankrupts = sum(1 for r in agent_results if r['bankrupt'])
        profitable = sum(1 for r in agent_results if r['reward'] > 0)
        
        stats = {
            'name': agent_name,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_bankroll': np.mean(bankrolls),
            'win_rate': total_wins / max(1, total_wins + total_losses) * 100,
            'bankrupt_rate': bankrupts / len(agent_results) * 100,
            'profitable_rate': profitable / len(agent_results) * 100,
        }
        agent_stats.append(stats)
        
        print(f"{agent_name:<20} | {stats['avg_reward']:>+12.2f} | "
              f"${stats['avg_bankroll']:>13.2f} | {stats['win_rate']:>9.1f}% | "
              f"{stats['bankrupt_rate']:>9.1f}% | {stats['profitable_rate']:>9.1f}%")
    
    print("-" * 90)
    
    # Find best agent
    best_by_reward = max(agent_stats, key=lambda x: x['avg_reward'])
    best_by_bankroll = max(agent_stats, key=lambda x: x['avg_bankroll'])
    best_by_survival = min(agent_stats, key=lambda x: x['bankrupt_rate'])
    
    print("\n🏆 BEST PERFORMERS:")
    print(f"  📈 Highest Avg Reward:    {best_by_reward['name']} ({best_by_reward['avg_reward']:+.2f})")
    print(f"  💰 Highest Avg Bankroll:  {best_by_bankroll['name']} (${best_by_bankroll['avg_bankroll']:.2f})")
    print(f"  🛡️ Lowest Bankruptcy:     {best_by_survival['name']} ({best_by_survival['bankrupt_rate']:.1f}%)")
    
    # Compare vs Random baseline
    random_stats = next((s for s in agent_stats if s['name'] == 'Random'), None)
    if random_stats:
        print("\n📊 COMPARISON VS RANDOM BASELINE:")
        print("-" * 60)
        for stats in agent_stats:
            if stats['name'] != 'Random':
                diff = stats['avg_reward'] - random_stats['avg_reward']
                emoji = "✅" if diff > 0 else "❌" if diff < 0 else "➖"
                print(f"  {stats['name']:<20}: {diff:+.2f} {emoji}")
    
    print("\n" + "=" * 90)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("🎰 RL Roulette - Agent Comparison")
    print("=" * 70)
    
    # Create agents
    print("\n📦 Loading agents...")
    agents = create_agents(args)
    print(f"   Loaded {len(agents)} agents")
    
    # Run comparison
    print(f"\n🏃 Running comparison ({args.episodes} episodes per agent)...")
    results = run_comparison(agents, args)
    
    # Print results
    print_results(results, args)


if __name__ == '__main__':
    main()
