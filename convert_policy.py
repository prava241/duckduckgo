"""
Convert Trained MAPPO Policy back to Strategy Dictionary Format

This script shows how to:
1. Load a trained MAPPO policy
2. Convert it to your strategy dictionary format
3. Use it in your double oracle algorithm
"""

import torch
import numpy as np
from typing import Dict
from collections import defaultdict

from game import Game, Strategy, Player, Action, player_to_Player
from mappo_poker import PokerInfosetEncoder, PokerGameTreeEnv, train_mappo_best_response


def policy_network_to_strategy_dict(
    policy_network,
    game: Game,
    team: str,
    encoder: PokerInfosetEncoder,
    device: str = "cpu",
    num_samples: int = 100
) -> Strategy:
    """
    Convert a trained MAPPO policy network to strategy dictionary format
    
    Args:
        policy_network: Trained TorchRL policy
        game: The game tree
        team: "13" or "24"
        encoder: The observation encoder
        device: torch device
        num_samples: Number of samples to estimate probabilities (for stochastic policies)
    
    Returns:
        Strategy object in your original format
    """
    
    team_players = [player_to_Player(f"P{p}") for p in team]
    strategy_dict = {player: {} for player in team_players}
    
    print(f"Converting policy network to strategy dict for team {team}...")
    
    # For each player in the team
    for player in team_players:
        player_infosets = game.infosets[player]
        
        print(f"  Processing {len(player_infosets)} infosets for player {player}...")
        
        for infoset, infoset_info in player_infosets.items():
            available_actions = infoset_info["actions"]
            
            # Encode the infoset as observation
            obs = encoder.infoset_to_observation(infoset, player)
            obs_batch = obs.unsqueeze(0).to(device)  # Add batch dimension
            
            # For multi-agent, we need to create a full observation tensor
            # with placeholders for other agents
            n_agents = len(team_players)
            agent_idx = team_players.index(player)
            
            full_obs = torch.zeros(1, n_agents, encoder.obs_dim, device=device)
            full_obs[0, agent_idx] = obs_batch[0]
            
            # Get action probabilities from policy
            with torch.no_grad():
                # Create a minimal tensordict for the policy
                from tensordict import TensorDict
                
                td = TensorDict({
                    "agents": TensorDict({
                        "observation": full_obs
                    }, batch_size=[1, n_agents])
                }, batch_size=[1])
                
                # Run policy
                policy_output = policy_network(td)
                
                # Extract action probabilities for this agent
                if "agents" in policy_output and "action" in policy_output["agents"]:
                    # If policy outputs actions directly, sample multiple times
                    action_counts = defaultdict(int)
                    for _ in range(num_samples):
                        policy_out_sample = policy_network(td)
                        action = policy_out_sample[("agents", "action")][0, agent_idx].item()
                        action_counts[Action(action)] += 1
                    
                    # Convert counts to probabilities
                    action_probs = {
                        action: count / num_samples 
                        for action, count in action_counts.items()
                    }
                
                elif "agents" in policy_output and "logits" in policy_output["agents"]:
                    # If policy outputs logits, convert to probabilities
                    logits = policy_output[("agents", "logits")][0, agent_idx]
                    probs = torch.softmax(logits, dim=0)
                    
                    action_probs = {
                        Action(i): probs[i].item()
                        for i in range(3)  # Assuming 3 actions
                    }
                else:
                    # Fallback: uniform distribution
                    action_probs = {
                        action: 1.0 / len(available_actions)
                        for action in available_actions
                    }
            
            # Filter to only available actions and renormalize
            filtered_probs = {
                action: action_probs.get(action, 0.0)
                for action in available_actions
            }
            
            # Renormalize
            total = sum(filtered_probs.values())
            if total > 0:
                filtered_probs = {
                    action: prob / total
                    for action, prob in filtered_probs.items()
                }
            else:
                # Uniform fallback
                filtered_probs = {
                    action: 1.0 / len(available_actions)
                    for action in available_actions
                }
            
            strategy_dict[player][infoset] = filtered_probs
    
    # Create Strategy object
    strategy = Strategy(
        game=game,
        team=team,
        strategies=[(strategy_dict, 1.0)]  # Single pure strategy with probability 1
    )
    strategy.calculate_contribs()
    
    print(f"Strategy conversion complete!")
    return strategy


def evaluate_policy_vs_strategy(
    policy_network,
    opponent_strategy: Strategy,
    game: Game,
    learning_team: str,
    encoder: PokerInfosetEncoder,
    device: str = "cpu",
    num_episodes: int = 100
) -> float:
    """
    Evaluate a trained policy against an opponent strategy
    
    Returns:
        Average reward over num_episodes
    """
    
    print(f"\nEvaluating policy for team {learning_team}...")
    
    # Create environment
    env = PokerGameTreeEnv(
        game=game,
        opponent_strategy=opponent_strategy,
        learning_team=learning_team,
        encoder=encoder,
        device=device,
        batch_size=1
    )
    
    total_reward = 0.0
    completed_episodes = 0
    
    for episode in range(num_episodes):
        td = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                td = policy_network(td)
            
            # Step environment
            td = env.step(td)
            
            # Check if done
            done = td["done"].item()
            
            if done:
                # Get episode reward
                if ("agents", "reward") in td.keys(True):
                    reward = td[("agents", "reward")].mean().item()
                    episode_reward += reward
        
        total_reward += episode_reward
        completed_episodes += 1
    
    avg_reward = total_reward / completed_episodes if completed_episodes > 0 else 0.0
    print(f"Average reward over {completed_episodes} episodes: {avg_reward:.4f}")
    
    return avg_reward


def integrate_with_double_oracle(
    game: Game,
    initial_opponent_strategy: Strategy,
    learning_team: str = "13",
    num_do_iterations: int = 5,
    mappo_iters: int = 50,
    device: str = "cpu"
):
    """
    Example: Integrate MAPPO with your double oracle algorithm
    
    Instead of using LP-based best response, use MAPPO to learn best response
    """
    
    print("=" * 80)
    print("INTEGRATING MAPPO WITH DOUBLE ORACLE")
    print("=" * 80)
    
    encoder = PokerInfosetEncoder(game)
    populations = {"13": [], "24": []}
    
    # Initialize with random strategies
    for team in ["13", "24"]:
        init_strat = Strategy(game, team)
        init_strat.random_strategy()
        populations[team].append(init_strat)
    
    opponent_team = "24" if learning_team == "13" else "13"
    
    for iteration in range(num_do_iterations):
        print(f"\n{'=' * 80}")
        print(f"DOUBLE ORACLE ITERATION {iteration + 1}/{num_do_iterations}")
        print(f"{'=' * 80}")
        
        # Get current opponent strategy (could be mixture from Nash equilibrium)
        opponent_strategy = populations[opponent_team][-1]
        
        # Train MAPPO best response
        print(f"\nTraining MAPPO best response for team {learning_team}...")
        trained_policy, rewards, env = train_mappo_best_response(
            game=game,
            opponent_strategy=opponent_strategy,
            learning_team=learning_team,
            device=device,
            frames_per_batch=2000,
            n_iters=mappo_iters,
            num_epochs=20,
            minibatch_size=200,
            use_mappo=True,
        )
        
        # Convert trained policy to strategy dictionary
        print(f"\nConverting trained policy to strategy dictionary...")
        new_strategy = policy_network_to_strategy_dict(
            policy_network=trained_policy,
            game=game,
            team=learning_team,
            encoder=encoder,
            device=device,
            num_samples=100
        )
        
        # Add to population
        populations[learning_team].append(new_strategy)
        
        # Evaluate
        print(f"\nEvaluating new strategy...")
        avg_reward = evaluate_policy_vs_strategy(
            policy_network=trained_policy,
            opponent_strategy=opponent_strategy,
            game=game,
            learning_team=learning_team,
            encoder=encoder,
            device=device,
            num_episodes=50
        )
        
        print(f"\nIteration {iteration + 1} complete. Average reward: {avg_reward:.4f}")
        
        # In a full implementation, you would:
        # 1. Compute utilities for all strategy pairs
        # 2. Solve for Nash equilibrium mixture
        # 3. Use Nash mixture as opponent for next iteration
        # 4. Check for convergence
    
    print("\n" + "=" * 80)
    print("DOUBLE ORACLE WITH MAPPO COMPLETE")
    print("=" * 80)
    
    return populations


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Loading game...")
    game = Game()
    print(f"Game loaded with {len(game.leaves)} leaf nodes\n")
    
    # Example 1: Train MAPPO and convert to strategy dict
    print("=" * 80)
    print("EXAMPLE 1: Train MAPPO and convert to strategy dictionary")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Create opponent strategy
    opponent_strategy = Strategy(game, "24")
    opponent_strategy.random_strategy()
    
    # Train MAPPO
    encoder = PokerInfosetEncoder(game)
    trained_policy, rewards, env = train_mappo_best_response(
        game=game,
        opponent_strategy=opponent_strategy,
        learning_team="13",
        device=device,
        frames_per_batch=2000,
        n_iters=30,
        num_epochs=15,
        minibatch_size=200,
    )
    
    # Convert to strategy dict
    learned_strategy = policy_network_to_strategy_dict(
        policy_network=trained_policy,
        game=game,
        team="13",
        encoder=encoder,
        device=device,
        num_samples=100
    )
    
    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    print(f"Strategy has {len(learned_strategy.strategies)} pure strategies")
    print(f"Sequence form shape: {learned_strategy.seq_form.shape}")
    
    # Example 2: Integrate with double oracle
    # Uncomment to run full integration
    print("\n\n")
    integrate_with_double_oracle(
        game=game,
        initial_opponent_strategy=opponent_strategy,
        learning_team="13",
        num_do_iterations=3,
        mappo_iters=30,
        device=device
    )