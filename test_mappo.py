"""
Test Script for MAPPO Poker Implementation

Run this to verify that everything is working correctly
"""

import torch
import numpy as np
from game import Game, Strategy, Player, Action, str_player
from mappo_poker import (
    PokerInfosetEncoder,
    OpponentPolicyLookup,
    PokerGameTreeEnv,
    train_mappo_best_response
)
from convert_policy import policy_network_to_strategy_dict


def test_encoder():
    """Test that the observation encoder works"""
    print("=" * 80)
    print("TEST 1: Observation Encoder")
    print("=" * 80)
    
    game = Game()
    encoder = PokerInfosetEncoder(game)
    
    # First, let's examine some actual infosets from the game
    print("\nExamining actual infosets from the game:")
    for player in [Player.One, Player.Two, Player.Three, Player.Four]:
        player_infosets = game.infosets[player]
        print(f"\n{str_player(player)} has {len(player_infosets)} infosets")
        # Show first 5 infosets as examples
        for i, infoset in enumerate(list(player_infosets.keys())[:5]):
            print(f"  Example {i+1}: '{infoset}'")
    
    # Test encoding a simple infoset
    # Pick an actual infoset from the game
    sample_player = Player.Two
    player_infosets = game.infosets[sample_player]
    if len(player_infosets) > 0:
        infoset = list(player_infosets.keys())[0]
        print(f"\nTesting encoding with actual infoset:")
        print(f"  Infoset: '{infoset}'")
        print(f"  Player: {sample_player}")
        
        obs = encoder.infoset_to_observation(infoset, sample_player)
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dim: {encoder.obs_dim}")
        assert obs.shape[0] == encoder.obs_dim, "Observation dimension mismatch!"
    
    # Test global state encoding with actual node
    print(f"\nTesting global state encoding:")
    # Get a sample node that's not root
    sample_nodes = [name for name in game.nodes.keys() if len(name) >= 4 and '/' in name]
    if len(sample_nodes) > 0:
        node_name = sample_nodes[0]
        node = game.nodes[node_name]
        print(f"  Node name: '{node_name}'")
        global_state = encoder.encode_global_state(node_name, node)
        print(f"  Global state shape: {global_state.shape}")
        print(f"  Global state dim: {encoder.global_state_dim}")
        assert global_state.shape[0] == encoder.global_state_dim, "Global state dimension mismatch!"
    
    print("\n✅ Encoder test passed!\n")
    return encoder, game


def test_opponent_policy():
    """Test that opponent policy lookup works"""
    print("=" * 80)
    print("TEST 2: Opponent Policy Lookup")
    print("=" * 80)
    
    game = Game()
    encoder = PokerInfosetEncoder(game)
    
    # Create a simple opponent strategy
    opponent_strategy = Strategy(game, "24")
    opponent_strategy.random_strategy()
    
    # Create opponent policy
    opponent_policy = OpponentPolicyLookup(
        game, opponent_strategy, encoder, device="cpu"
    )
    
    # Test forward pass
    n_agents = 2
    observations = torch.randn(n_agents, encoder.obs_dim)
    
    action_probs = opponent_policy.forward(observations)
    print(f"Input observations shape: {observations.shape}")
    print(f"Output action probs shape: {action_probs.shape}")
    print(f"Action probs sum per agent: {action_probs.sum(dim=1)}")
    
    assert action_probs.shape == (n_agents, 3), "Action probs shape incorrect!"
    assert torch.allclose(action_probs.sum(dim=1), torch.ones(n_agents)), "Probs don't sum to 1!"
    
    # Test action sampling
    actions = opponent_policy.get_actions(observations)
    print(f"Sampled actions: {actions}")
    assert actions.shape == (n_agents,), "Actions shape incorrect!"
    
    print("\n✅ Opponent policy test passed!\n")
    return opponent_policy, game, encoder


def test_environment():
    """Test that the environment works"""
    print("=" * 80)
    print("TEST 3: Environment")
    print("=" * 80)
    
    game = Game()
    encoder = PokerInfosetEncoder(game)
    
    # Create opponent strategy
    opponent_strategy = Strategy(game, "24")
    opponent_strategy.random_strategy()
    
    # Create environment
    env = PokerGameTreeEnv(
        game=game,
        opponent_strategy=opponent_strategy,
        learning_team="13",
        encoder=encoder,
        device="cpu",
        batch_size=1
    )
    
    print(f"Environment created successfully")
    print(f"Learning team: {env.learning_team}")
    print(f"Number of learning agents: {env.n_learning_agents}")
    print(f"Observation spec: {env.observation_spec}")
    print(f"Action spec: {env.action_spec}")
    
    # Test reset
    print("\nTesting reset...")
    td = env.reset()
    print(f"Reset output keys: {td.keys()}")
    print(f"Observation shape: {td[('agents', 'observation')].shape}")
    print(f"Global state shape: {td['global_state'].shape}")
    
    # Test step
    print("\nTesting step...")
    from tensordict import TensorDict
    
    # Create random actions
    actions = torch.randint(0, 3, (1, env.n_learning_agents))
    td[("agents", "action")] = actions
    print(f"Actions: {actions}")
    
    td_next = env.step(td)
    print(f"Step output keys: {td_next.keys()}")
    
    # In TorchRL, rewards are in the root level, not nested
    # Check what's actually in the tensordict
    if "reward" in td_next.keys():
        print(f"Reward shape: {td_next['reward'].shape}")
        print(f"Reward value: {td_next['reward']}")
    elif ("agents", "reward") in td_next.keys(True):
        print(f"Reward shape: {td_next[('agents', 'reward')].shape}")
        print(f"Reward value: {td_next[('agents', 'reward')]}")
    else:
        print(f"Available keys: {td_next.keys(True)}")
        # The reward might be in 'next'
        if "next" in td_next.keys() and "agents" in td_next["next"].keys():
            if "reward" in td_next["next"]["agents"].keys():
                print(f"Reward in next: {td_next['next']['agents']['reward']}")
    
    print(f"Done: {td_next['done']}")
    
    # Test a few more steps
    print("\nTesting episode rollout...")
    td = env.reset()
    steps = 0
    total_reward = 0
    
    while steps < 10:  # Max 10 steps for test
        actions = torch.randint(0, 3, (1, env.n_learning_agents))
        td[("agents", "action")] = actions
        td = env.step(td)
        
        # Extract reward from the proper location
        # TorchRL puts new observations/rewards in 'next'
        reward = 0
        if "next" in td.keys():
            if ("agents", "reward") in td["next"].keys(True):
                reward = td["next"][("agents", "reward")].sum().item()
            elif "reward" in td["next"].keys():
                reward = td["next"]["reward"].sum().item()
        elif ("agents", "reward") in td.keys(True):
            reward = td[("agents", "reward")].sum().item()
        elif "reward" in td.keys():
            reward = td["reward"].sum().item()
        
        total_reward += reward
        done = td["done"].item() if "done" in td.keys() else td["next"]["done"].item()
        
        steps += 1
        print(f"  Step {steps}: reward={reward:.2f}, done={done}")
        
        if done:
            print(f"  Episode finished in {steps} steps. Total reward: {total_reward:.2f}")
            break
    
    if not done:
        print(f"  Reached max test steps ({steps}). Total reward so far: {total_reward:.2f}")
    
    print("\n✅ Environment test passed!\n")
    return env


def test_training_mini():
    """Test a minimal training run"""
    print("=" * 80)
    print("TEST 4: Mini Training Run")
    print("=" * 80)
    
    game = Game()
    
    # Create opponent strategy
    opponent_strategy = Strategy(game, "24")
    opponent_strategy.random_strategy()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\nStarting mini training (5 iterations)...")
    trained_policy, rewards, env = train_mappo_best_response(
        game=game,
        opponent_strategy=opponent_strategy,
        learning_team="13",
        device=device,
        frames_per_batch=1000,  # Very small for testing
        n_iters=5,               # Just 5 iterations
        num_epochs=5,            # Fewer epochs
        minibatch_size=100,
        lr=3e-4,
        use_mappo=True,
    )
    
    print(f"\nTraining complete!")
    print(f"Rewards over iterations: {rewards}")
    
    if len(rewards) > 0:
        print(f"Initial reward: {rewards[0]:.4f}")
        print(f"Final reward: {rewards[-1]:.4f}")
        improvement = rewards[-1] - rewards[0]
        print(f"Improvement: {improvement:.4f}")
    
    print("\n✅ Training test passed!\n")
    return trained_policy, game, env


def test_conversion():
    """Test converting policy to strategy dict"""
    print("=" * 80)
    print("TEST 5: Policy Conversion")
    print("=" * 80)
    
    # Train a quick policy
    game = Game()
    opponent_strategy = Strategy(game, "24")
    opponent_strategy.random_strategy()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Training quick policy for conversion test...")
    trained_policy, _, env = train_mappo_best_response(
        game=game,
        opponent_strategy=opponent_strategy,
        learning_team="13",
        device=device,
        frames_per_batch=500,
        n_iters=3,
        num_epochs=3,
        minibatch_size=50,
    )
    
    # Convert to strategy dict
    print("\nConverting policy to strategy dictionary...")
    from convert_policy import policy_network_to_strategy_dict
    
    encoder = PokerInfosetEncoder(game)
    learned_strategy = policy_network_to_strategy_dict(
        policy_network=trained_policy,
        game=game,
        team="13",
        encoder=encoder,
        device=device,
        num_samples=50  # Fewer samples for testing
    )
    
    print(f"\nConversion complete!")
    print(f"Strategy format: {type(learned_strategy)}")
    print(f"Team: {learned_strategy.team_name}")
    print(f"Number of pure strategies: {len(learned_strategy.strategies)}")
    print(f"Sequence form shape: {learned_strategy.seq_form.shape}")
    
    # Check strategy dict structure
    strat_dict = learned_strategy.strategies[0][0]
    print(f"\nStrategy dict keys (players): {list(strat_dict.keys())}")
    
    sample_player = list(strat_dict.keys())[0]
    print(f"Sample player: {sample_player}")
    print(f"Number of infosets for player: {len(strat_dict[sample_player])}")
    
    # Check an infoset
    sample_infoset = list(strat_dict[sample_player].keys())[0]
    print(f"\nSample infoset: {sample_infoset}")
    print(f"Action probabilities: {strat_dict[sample_player][sample_infoset]}")
    
    # Verify probabilities sum to 1
    prob_sum = sum(strat_dict[sample_player][sample_infoset].values())
    print(f"Probability sum: {prob_sum:.6f}")
    assert abs(prob_sum - 1.0) < 0.01, "Probabilities don't sum to 1!"
    
    print("\n✅ Conversion test passed!\n")
    return learned_strategy


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS")
    print("=" * 80 + "\n")
    
    try:
        # Test 1: Encoder
        encoder, game = test_encoder()
        
        # Test 2: Opponent Policy
        opponent_policy, game, encoder = test_opponent_policy()
        
        # Test 3: Environment
        env = test_environment()
        
        # Test 4: Training (mini)
        trained_policy, game, env = test_training_mini()
        
        # Test 5: Conversion
        learned_strategy = test_conversion()
        
        # All tests passed
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED! ✅")
        print("=" * 80)
        print("\nYour MAPPO implementation is working correctly!")
        print("You can now:")
        print("  1. Run full training with more iterations")
        print("  2. Integrate with your double oracle algorithm")
        print("  3. Experiment with different hyperparameters")
        print("\nSee README_MAPPO.md for detailed usage instructions.")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error and fix any issues.")
        print("=" * 80 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()