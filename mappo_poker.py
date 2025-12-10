"""
Complete MAPPO Implementation for 4-Player Team Poker Game
Converts game tree and strategy dictionary to RL environment and trains best response
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

# TorchRL imports
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.envs import RewardSum, TransformedEnv
from tqdm import tqdm
from matplotlib import pyplot as plt

# Your game imports
from game import Game, Strategy, Player, Action, Node, LeafNode, player_to_Player, str_player, action_to_Action

# ============================================================================
# PART 1: OBSERVATION ENCODER
# ============================================================================

class PokerInfosetEncoder:
    """Encodes poker infosets as observation vectors"""
    
    def __init__(self, game: Game):
        self.game = game
        
        # Card encoding: J=0, Q=1, K=2, None=-1
        self.card_to_idx = {"J": 0, "Q": 1, "K": 2}
        
        # Action encoding for history
        self.action_to_idx = {Action.Call: 0, Action.Fold: 1, Action.Raise: 2}
        
        # Maximum history length (estimate based on game structure)
        self.max_history_len = 20
        
        # Observation dimensions
        self.private_card_dim = 3  # One-hot encoding of J/Q/K
        self.community_card_dim = 3  # One-hot encoding of community card (or zeros)
        self.history_dim = self.max_history_len * 5  # (player_id, action) pairs, flattened
        self.meta_dim = 2  # (turn_number, player_position)
        
        self.obs_dim = (self.private_card_dim + self.community_card_dim + 
                       self.history_dim + self.meta_dim)
        
        # Global state for centralized critic
        self.global_state_dim = 4 * self.private_card_dim + self.community_card_dim + self.history_dim + 4
    
    def encode_card(self, card: str) -> torch.Tensor:
        """One-hot encode a card"""
        vec = torch.zeros(3)
        if card in self.card_to_idx:
            vec[self.card_to_idx[card]] = 1.0
        return vec
    
    def encode_history(self, history: List[Tuple[Player, str, Action]]) -> torch.Tensor:
        """Encode action history"""
        vec = torch.zeros(self.max_history_len * 5)
        
        for i, (player, infoset, action) in enumerate(history[:self.max_history_len]):
            if player == Player.Chance:
                # Encode chance node
                vec[i * 5] = 4  # Special marker for chance
                vec[i * 5 + 4] = 1  # Flag for chance
            else:
                # Encode player action
                vec[i * 5 + player] = 1.0  # One-hot player
                if action is not None:
                    vec[i * 5 + 4] = self.action_to_idx[action]
        
        return vec
    
    def encode_observation(self, node_name: str, player_id: int, node: Node) -> torch.Tensor:
        """
        Encode a node's state as an observation for a specific player
        Args:
            node_name: Full node string (e.g., "JQQK/P1:C/P2:R")
            player_id: Player ID (0-3)
            node: The actual node object
        """
        # Extract private card for this player
        # The first 4 characters of a full node are the cards for players 0,1,2,3
        if len(node_name) >= 4:
            # Check if we're at the initial card deal or later
            card_section = node_name.split('/')[0]  # Get part before any actions
            if len(card_section) >= player_id + 1:
                private_card = card_section[player_id]
                private_card_vec = self.encode_card(private_card)
            else:
                private_card_vec = torch.zeros(3)
        else:
            private_card_vec = torch.zeros(3)
        
        # Extract community card if revealed
        community_card_vec = torch.zeros(3)
        if "C:" in node_name:
            # Find the community card reveal
            parts = node_name.split('/')
            for part in parts:
                if part.startswith("C:"):
                    community_card = part[2] if len(part) > 2 else None
                    if community_card:
                        community_card_vec = self.encode_card(community_card)
                    break
        
        # Encode history
        history_vec = self.encode_history(node.history)
        
        # Meta information
        turn_number = len([h for h in node.history if h[0] != Player.Chance])
        meta_vec = torch.tensor([turn_number / 20.0, player_id / 4.0])
        
        # Concatenate all features
        obs = torch.cat([private_card_vec, community_card_vec, history_vec, meta_vec])
        
        return obs
    
    def encode_global_state(self, node_name: str, node: Node) -> torch.Tensor:
        """Encode full global state for centralized critic"""
        # All private cards (first 4 characters if available)
        all_cards = []
        card_section = node_name.split('/')[0] if '/' in node_name else node_name
        
        if len(card_section) >= 4:
            for i in range(4):
                all_cards.append(self.encode_card(card_section[i]))
        else:
            all_cards = [torch.zeros(3) for _ in range(4)]
        
        # Community card
        community_card_vec = torch.zeros(3)
        if "C:" in node_name:
            parts = node_name.split('/')
            for part in parts:
                if part.startswith("C:"):
                    community_card = part[2] if len(part) > 2 else None
                    if community_card:
                        community_card_vec = self.encode_card(community_card)
                    break
        
        # History
        history_vec = self.encode_history(node.history)
        
        # Meta info (turn, all player positions)
        turn_number = len([h for h in node.history if h[0] != Player.Chance])
        meta_vec = torch.tensor([turn_number / 20.0, 0.25, 0.5, 0.75])  # Player positions
        
        global_state = torch.cat(all_cards + [community_card_vec, history_vec, meta_vec])
        return global_state
    
    def infoset_to_observation(self, infoset: str, player: Player) -> torch.Tensor:
        """
        Convert an infoset string to observation
        This is used for the opponent policy lookup
        
        Infoset format examples:
        - "Q???" - Just the player's card (Player 1), others unknown
        - "?J??/P1:C" - Player 2's card is J, Player 1 called
        - "??Q?/P1:C/P2:R" - Player 3's card, with actions
        - "???K/P1:C/P2:R/P3:C" - Player 4's card
        - "Q???/P1:C/P2:R/C:K" - With community card revealed
        """
        
        # Split by '/' to get parts
        parts = infoset.split('/')
        
        # First part contains cards (e.g., "?J??")
        # The player's card is at position player (0-3)
        card_string = parts[0] if len(parts) > 0 else ""
        
        # Extract player's private card
        if len(card_string) > int(player) and card_string[int(player)] != '?':
            private_card = card_string[int(player)]
            player_card_vec = self.encode_card(private_card)
        else:
            # If we see '?', we don't know this card (shouldn't happen for this player's infoset)
            player_card_vec = torch.zeros(3)
        
        # Extract community card if present (format: "C:K")
        community_card_vec = torch.zeros(3)
        history = []
        
        for part in parts[1:]:  # Skip first part (cards)
            if ':' not in part:
                continue
                
            player_str, action_str = part.split(':', 1)  # Use maxsplit=1
            
            # Check if this is a community card reveal (player_str == "C")
            if player_str == "C":
                # This is the community card
                community_card_vec = self.encode_card(action_str)
                # Add a chance node to history
                history.append((Player.Chance, None, None))
            else:
                # This is a player action
                try:
                    p = player_to_Player(player_str)
                    a = action_to_Action(action_str)
                    history.append((p, None, a))
                except (KeyError, ValueError):
                    # Skip invalid entries
                    continue
        
        history_vec = self.encode_history(history)
        
        # Meta info
        turn_number = len(history)
        meta_vec = torch.tensor([turn_number / 20.0, int(player) / 4.0])
        
        obs = torch.cat([player_card_vec, community_card_vec, history_vec, meta_vec])
        return obs


# ============================================================================
# PART 2: OPPONENT POLICY (Converts Strategy Dict to Neural Network)
# ============================================================================

class OpponentPolicyLookup(nn.Module):
    """
    Exact lookup policy that uses the opponent's strategy dictionary
    No approximation - directly looks up probabilities
    """
    
    def __init__(self, game: Game, strategy: Strategy, encoder: PokerInfosetEncoder, device: str = "cpu"):
        super().__init__()
        self.game = game
        self.strategy = strategy
        self.encoder = encoder
        self.device = device
        
        # Extract the pure strategy (assuming strategy has single strategy with prob 1)
        if len(strategy.strategies) > 0:
            self.strategy_dict = strategy.strategies[0][0]  # Get first (and likely only) strategy
        else:
            raise ValueError("Strategy has no strategies defined")
        
        self.team = strategy.team
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (n_agents, obs_dim) for the opponent team
        Returns:
            action_probs: (n_agents, n_actions) probability distribution over actions
        """
        n_agents = observations.shape[0]
        action_probs = []
        
        for agent_idx in range(n_agents):
            player = self.team[agent_idx]
            obs = observations[agent_idx]
            
            # Find which infoset this observation corresponds to
            # This requires searching through the game's infosets
            action_prob = self._get_action_probs_for_obs(obs, player)
            action_probs.append(action_prob)
        
        return torch.stack(action_probs)
    
    def _get_action_probs_for_obs(self, obs: torch.Tensor, player: Player) -> torch.Tensor:
        """
        Find the matching infoset and return action probabilities
        """
        # Get all infosets for this player
        player_infosets = self.game.infosets[player]
        
        best_match = None
        best_match_infoset = None
        min_dist = float('inf')
        
        # Find closest matching infoset (should be exact match in theory)
        for infoset in player_infosets.keys():
            infoset_obs = self.encoder.infoset_to_observation(infoset, player)
            dist = torch.norm(obs - infoset_obs)
            if dist < min_dist:
                min_dist = dist
                best_match_infoset = infoset
        
        # Get action probabilities from strategy dict
        if best_match_infoset and player in self.strategy_dict:
            action_dict = self.strategy_dict[player].get(best_match_infoset, {})
            
            # Convert to tensor (assuming actions are 0, 1, 2)
            action_prob = torch.zeros(3, device=self.device)
            for action, prob in action_dict.items():
                action_prob[int(action)] = prob
            
            # Normalize in case of numerical issues
            if action_prob.sum() > 0:
                action_prob = action_prob / action_prob.sum()
            else:
                # Uniform if no valid actions
                action_prob = torch.ones(3, device=self.device) / 3.0
        else:
            # Uniform default
            action_prob = torch.ones(3, device=self.device) / 3.0
        
        return action_prob
    
    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions from the policy"""
        action_probs = self.forward(observations)
        actions = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        return actions


# ============================================================================
# PART 3: GAME TREE ENVIRONMENT
# ============================================================================

class PokerGameTreeEnv(EnvBase):
    """
    TorchRL environment that wraps the 4-player poker game tree
    Trains team 13 (or 24) against a fixed opponent team strategy
    """
    
    def __init__(
        self,
        game: Game,
        opponent_strategy: Strategy,
        learning_team: str = "13",  # "13" or "24"
        encoder: PokerInfosetEncoder = None,
        device: str = "cpu",
        batch_size: int = 1
    ):
        # Initialize parent class FIRST
        super().__init__(device=device, batch_size=torch.Size([batch_size]))
        
        self.game = game
        self.opponent_strategy = opponent_strategy
        self.learning_team = learning_team
        self.opponent_team = "24" if learning_team == "13" else "13"
        
        # Set up teams
        self.learning_players = [player_to_Player(f"P{p}") for p in learning_team]
        self.opponent_players = [player_to_Player(f"P{p}") for p in self.opponent_team]
        
        self.n_learning_agents = len(self.learning_players)
        self.n_opponent_agents = len(self.opponent_players)
        
        # Encoder
        self.encoder = encoder if encoder else PokerInfosetEncoder(game)
        
        # Create opponent policy
        self.opponent_policy = OpponentPolicyLookup(
            game, opponent_strategy, self.encoder, device
        )
        
        # Current game state
        self.current_node = None
        self.current_node_name = None
        self.episode_steps = 0
        self.max_episode_steps = 50  # Safety limit
        
        self._make_specs()
    
    def _make_specs(self):
        """Define observation, action, reward, and done specs"""
        
        # Observation spec
        self.observation_spec = CompositeSpec(
            agents=CompositeSpec(
                observation=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, self.n_learning_agents, self.encoder.obs_dim),
                    device=self.device
                ),
                shape=(*self.batch_size, self.n_learning_agents)
            ),
            global_state=UnboundedContinuousTensorSpec(
                shape=(*self.batch_size, self.encoder.global_state_dim),
                device=self.device
            ),
            shape=self.batch_size
        )
        
        # Action spec - discrete actions (Call=0, Fold=1, Raise=2)
        self.action_spec = CompositeSpec(
            agents=CompositeSpec(
                action=DiscreteTensorSpec(
                    n=3,  # 3 possible actions
                    shape=(*self.batch_size, self.n_learning_agents),
                    device=self.device
                ),
                shape=(*self.batch_size, self.n_learning_agents)
            ),
            shape=self.batch_size
        )
        
        # Reward spec
        self.reward_spec = CompositeSpec(
            agents=CompositeSpec(
                reward=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, self.n_learning_agents, 1),
                    device=self.device
                ),
                shape=(*self.batch_size, self.n_learning_agents)
            ),
            shape=self.batch_size
        )
        
        # Done spec
        self.done_spec = CompositeSpec(
            done=BoundedTensorSpec(
                low=0, high=1, 
                shape=(*self.batch_size, 1), 
                dtype=torch.bool, 
                device=self.device
            ),
            terminated=BoundedTensorSpec(
                low=0, high=1, 
                shape=(*self.batch_size, 1), 
                dtype=torch.bool, 
                device=self.device
            ),
            shape=self.batch_size
        )
    
    def _reset(self, tensordict=None, **kwargs):
        """Reset to root of game tree"""
        # Sample a random initial deal
        initial_hands = list(self.game.nodes[""].chance_actions.keys())
        chosen_hand = initial_hands[torch.randint(0, len(initial_hands), (1,)).item()]
        
        self.current_node_name = chosen_hand
        self.current_node = self.game.nodes[chosen_hand]
        self.episode_steps = 0
        
        # Get observations
        observations = self._get_observations()
        global_state = self._get_global_state()
        
        done = torch.zeros(*self.batch_size, 1, device=self.device, dtype=torch.bool)
        
        td = TensorDict({
            "agents": TensorDict({
                "observation": observations,
            }, batch_size=(*self.batch_size, self.n_learning_agents)),
            "global_state": global_state,
            "done": done,
            "terminated": done.clone(),
        }, batch_size=self.batch_size)
        
        return td
    
    def _step(self, tensordict: TensorDict):
        """Execute one step in the game tree"""
        self.episode_steps += 1
        
        # Check if we're already at a terminal node
        if isinstance(self.current_node, LeafNode):
            # Already at terminal, return final state
            terminal_rewards = self._get_terminal_rewards(self.current_node)
            next_observations = self._get_observations()
            global_state = self._get_global_state()
            done_tensor = torch.tensor([[True]], device=self.device, dtype=torch.bool)
            
            td = TensorDict({
                "agents": TensorDict({
                    "observation": next_observations,
                    "reward": terminal_rewards,
                }, batch_size=(*self.batch_size, self.n_learning_agents)),
                "global_state": global_state,
                "done": done_tensor,
                "terminated": done_tensor.clone(),
            }, batch_size=self.batch_size)
            
            return td
        
        # Process all player actions until we hit a terminal or need more learning agent actions
        done = False
        terminal_rewards = None
        
        while not done and self.episode_steps < self.max_episode_steps:
            current_player = self.current_node.player
            
            if current_player == Player.Chance:
                # Handle chance node
                self._handle_chance_node()
                # Check if we landed on a terminal after chance
                if isinstance(self.current_node, LeafNode):
                    done = True
                    terminal_rewards = self._get_terminal_rewards(self.current_node)
                    break
                continue
            
            # Determine if current player is learning or opponent
            if current_player in self.learning_players:
                # Get action from tensordict
                agent_idx = self.learning_players.index(current_player)
                # Handle batch dimension: tensordict[("agents", "action")] has shape [batch, n_agents]
                action_tensor = tensordict[("agents", "action")]
                if action_tensor.dim() > 1:
                    # Extract action for this agent from first batch element
                    action = action_tensor[0, agent_idx].item()
                else:
                    action = action_tensor[agent_idx].item()
                action = Action(action)
                
                # Execute action
                if action in self.current_node.actions:
                    next_node_or_leaf = self.current_node.actions[action]
                    
                    if isinstance(next_node_or_leaf, LeafNode):
                        # Terminal node
                        done = True
                        terminal_rewards = self._get_terminal_rewards(next_node_or_leaf)
                        self.current_node = next_node_or_leaf
                        self.current_node_name = next_node_or_leaf.name
                    else:
                        # Continue to next node
                        self.current_node = next_node_or_leaf
                        self.current_node_name = next_node_or_leaf.name
                else:
                    # Invalid action - treat as fold
                    action = Action.Fold
                    next_node_or_leaf = self.current_node.actions.get(
                        action, list(self.current_node.actions.values())[0]
                    )
                    if isinstance(next_node_or_leaf, LeafNode):
                        done = True
                        terminal_rewards = self._get_terminal_rewards(next_node_or_leaf)
                        self.current_node = next_node_or_leaf
                        self.current_node_name = next_node_or_leaf.name
                    else:
                        self.current_node = next_node_or_leaf
                        self.current_node_name = next_node_or_leaf.name
                
                # Break to return control after learning agent acts
                break
            
            else:
                # Opponent player - use fixed policy
                agent_idx = self.opponent_players.index(current_player)
                opponent_obs = self._get_opponent_observations()
                
                with torch.no_grad():
                    opponent_actions = self.opponent_policy.get_actions(opponent_obs)
                    action = opponent_actions[agent_idx].item()
                    action = Action(action)
                
                # Execute opponent action
                if action in self.current_node.actions:
                    next_node_or_leaf = self.current_node.actions[action]
                    
                    if isinstance(next_node_or_leaf, LeafNode):
                        done = True
                        terminal_rewards = self._get_terminal_rewards(next_node_or_leaf)
                        self.current_node = next_node_or_leaf
                        self.current_node_name = next_node_or_leaf.name
                    else:
                        self.current_node = next_node_or_leaf
                        self.current_node_name = next_node_or_leaf.name
                else:
                    # Invalid action
                    action = Action.Fold
                    next_node_or_leaf = self.current_node.actions.get(
                        action, list(self.current_node.actions.values())[0]
                    )
                    if isinstance(next_node_or_leaf, LeafNode):
                        done = True
                        terminal_rewards = self._get_terminal_rewards(next_node_or_leaf)
                        self.current_node = next_node_or_leaf
                        self.current_node_name = next_node_or_leaf.name
                    else:
                        self.current_node = next_node_or_leaf
                        self.current_node_name = next_node_or_leaf.name
        
        # Safety check for max steps
        if self.episode_steps >= self.max_episode_steps:
            done = True
            terminal_rewards = torch.zeros(*self.batch_size, self.n_learning_agents, 1, device=self.device)
        
        # Get new observations
        next_observations = self._get_observations()
        global_state = self._get_global_state()
        
        # Rewards
        if terminal_rewards is not None:
            rewards = terminal_rewards
        else:
            rewards = torch.zeros(*self.batch_size, self.n_learning_agents, 1, device=self.device)
        
        done_tensor = torch.tensor([[done]], device=self.device, dtype=torch.bool)
        
        td = TensorDict({
            "agents": TensorDict({
                "observation": next_observations,
                "reward": rewards,
            }, batch_size=(*self.batch_size, self.n_learning_agents)),
            "global_state": global_state,
            "done": done_tensor,
            "terminated": done_tensor.clone(),
        }, batch_size=self.batch_size)
        
        return td
    
    def _handle_chance_node(self):
        """Sample outcome from chance node"""
        outcomes = list(self.current_node.chance_actions.keys())
        
        # Sample uniformly (or use actual probabilities if available)
        chosen_idx = torch.randint(0, len(outcomes), (1,)).item()
        chosen_outcome = outcomes[chosen_idx]
        
        next_node = self.current_node.chance_actions[chosen_outcome]
        self.current_node = next_node
        self.current_node_name = next_node.name
    
    def _get_observations(self) -> torch.Tensor:
        """Get observations for learning team agents"""
        observations = []
        
        for player in self.learning_players:
            # Handle both Node and LeafNode
            if isinstance(self.current_node, LeafNode):
                # At terminal, use the terminal node's info
                obs = self.encoder.encode_observation(
                    self.current_node_name, int(player), self.current_node
                )
            else:
                obs = self.encoder.encode_observation(
                    self.current_node_name, int(player), self.current_node
                )
            observations.append(obs)
        
        obs_tensor = torch.stack(observations).unsqueeze(0)  # Add batch dim
        return obs_tensor.to(self.device)
    
    def _get_opponent_observations(self) -> torch.Tensor:
        """Get observations for opponent team agents"""
        observations = []
        
        for player in self.opponent_players:
            # Handle both Node and LeafNode
            if isinstance(self.current_node, LeafNode):
                obs = self.encoder.encode_observation(
                    self.current_node_name, int(player), self.current_node
                )
            else:
                obs = self.encoder.encode_observation(
                    self.current_node_name, int(player), self.current_node
                )
            observations.append(obs)
        
        return torch.stack(observations).to(self.device)
    
    def _get_global_state(self) -> torch.Tensor:
        """Get global state for centralized critic"""
        # Handle both Node and LeafNode
        global_state = self.encoder.encode_global_state(
            self.current_node_name, self.current_node
        )
        return global_state.unsqueeze(0).to(self.device)  # Add batch dim
    
    def _get_terminal_rewards(self, leaf_node: LeafNode) -> torch.Tensor:
        """Extract rewards for learning team from leaf node"""
        payoffs = leaf_node.payoff
        team_payoff = payoffs[self.learning_team]
        
        # Distribute equally among team members (or customize)
        individual_reward = team_payoff / self.n_learning_agents
        
        # Shape: [batch_size, n_agents, 1]
        rewards = torch.full(
            (*self.batch_size, self.n_learning_agents, 1),
            individual_reward,
            device=self.device,
            dtype=torch.float32
        )
        
        return rewards
    
    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)


# ============================================================================
# PART 4: MAPPO TRAINING
# ============================================================================

def train_mappo_best_response(
    game: Game,
    opponent_strategy: Strategy,
    learning_team: str = "13",
    device: str = "cpu",
    # Hyperparameters
    frames_per_batch: int = 6000,
    n_iters: int = 100,
    num_epochs: int = 30,
    minibatch_size: int = 400,
    lr: float = 3e-4,
    max_grad_norm: float = 1.0,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    lmbda: float = 0.9,
    entropy_eps: float = 1e-4,
    share_parameters_policy: bool = True,
    share_parameters_critic: bool = True,
    use_mappo: bool = True  # True for MAPPO, False for IPPO
):
    """
    Train MAPPO to find best response against fixed opponent strategy
    
    Args:
        game: The poker game tree
        opponent_strategy: Fixed opponent team strategy
        learning_team: "13" or "24" - which team to train
        device: "cpu" or "cuda"
        Other args: MAPPO hyperparameters
    
    Returns:
        trained_policy: The learned policy network
        episode_rewards: List of episode rewards during training
    """
    
    print(f"Training team {learning_team} against team {opponent_strategy.team_name}")
    print(f"Using {'MAPPO' if use_mappo else 'IPPO'} with parameter sharing: {share_parameters_policy}")
    
    # Create encoder
    encoder = PokerInfosetEncoder(game)
    
    # Create environment - use single environment, not vectorized
    # TorchRL's collector will handle parallelization
    env = PokerGameTreeEnv(
        game=game,
        opponent_strategy=opponent_strategy,
        learning_team=learning_team,
        encoder=encoder,
        device=device,
        batch_size=1  # Single environment
    )
    
    # Add reward transform
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[("agents", "reward")], out_keys=[("agents", "episode_reward")])
    )
    
    n_agents = env.n_learning_agents
    obs_dim = encoder.obs_dim
    n_actions = 3  # Call, Fold, Raise
    
    print(f"Environment created with {n_agents} learning agents")
    print(f"Observation dim: {obs_dim}, Action dim: {n_actions}")
    
    # ========================================================================
    # Create Policy Network (Decentralized Execution)
    # ========================================================================
    
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=n_actions,  # Discrete actions, output logits
            n_agents=n_agents,
            centralised=False,  # Decentralized policy
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
    )
    
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )
    
    # Create categorical policy for discrete actions
    from torchrl.modules import MaskedCategorical
    
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )
    
    # ========================================================================
    # Create Critic Network (Centralized or Decentralized)
    # ========================================================================
    
    if use_mappo:
        # MAPPO: Centralized critic uses global state
        # For centralized critic, we use a regular MLP (not MultiAgentMLP)
        # because global_state doesn't have the agent dimension
        
        class CentralizedCritic(torch.nn.Module):
            def __init__(self, global_state_dim, n_agents, hidden_dim=256):
                super().__init__()
                self.n_agents = n_agents
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(global_state_dim, hidden_dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_dim, n_agents),
                )
            
            def forward(self, global_state):
                # global_state: [batch, global_dim]
                values = self.net(global_state)  # [batch, n_agents]
                # Reshape to [batch, n_agents, 1]
                return values.unsqueeze(-1)
        
        critic_net = CentralizedCritic(encoder.global_state_dim, n_agents).to(device)
        
        critic = TensorDictModule(
            module=critic_net,
            in_keys=["global_state"],  # Use global state
            out_keys=[("agents", "state_value")],
        )
    else:
        # IPPO: Decentralized critic uses local observations
        critic_net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralised=False,  # Decentralized critic
            share_params=share_parameters_critic,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )
        
        critic = TensorDictModule(
            module=critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )
    
    print("Policy and critic networks created")
    
    # ========================================================================
    # Create Collector
    # ========================================================================
    
    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * n_iters,
    )
    
    print("Collector created")
    
    # ========================================================================
    # Create Replay Buffer
    # ========================================================================
    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )
    
    # ========================================================================
    # Create Loss Function
    # ========================================================================
    
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,  # Important for multi-agent
    )
    
    loss_module.set_keys(
        reward=("agents", "reward"),
        action=("agents", "action"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    GAE = loss_module.value_estimator
    
    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
    
    print("Loss module and optimizer created")
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    
    episode_reward_mean_list = []
    pbar = tqdm(total=n_iters, desc="Training MAPPO Best Response")
    
    for iteration, tensordict_data in enumerate(collector):
        # Expand done/terminated to match reward shape
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", "agents", "reward"))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", "agents", "reward"))),
        )
        
        # Compute GAE
        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        
        # Flatten and add to replay buffer
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)
        
        # Training epochs
        for epoch in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                
                optim.step()
                optim.zero_grad()
        
        # Update collector policy
        collector.update_policy_weights_()
        
        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        if done.any():
            episode_rewards = tensordict_data.get(("next", "agents", "episode_reward"))[done]
            if len(episode_rewards) > 0:
                episode_reward_mean = episode_rewards.mean().item()
                episode_reward_mean_list.append(episode_reward_mean)
                pbar.set_description(
                    f"Iter {iteration+1}/{n_iters} | Reward: {episode_reward_mean:.3f}",
                    refresh=False
                )
        
        pbar.update(1)
    
    pbar.close()
    
    # ========================================================================
    # Extract Learned Strategy
    # ========================================================================
    
    print("\nTraining complete!")
    print(f"Final average reward: {episode_reward_mean_list[-1] if episode_reward_mean_list else 'N/A'}")
    
    # Plot results
    if episode_reward_mean_list:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_reward_mean_list)
        plt.xlabel("Training Iteration")
        plt.ylabel("Mean Episode Reward")
        plt.title(f"MAPPO Training - Team {learning_team} vs Team {opponent_strategy.team_name}")
        plt.grid(True)
        plt.savefig(f"/mnt/user-data/outputs/mappo_training_team{learning_team}.png")
        plt.close()
        print("Training plot saved")
    
    return policy, episode_reward_mean_list, env


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Loading game...")
    game = Game()
    print(f"Game loaded: {len(game.leaves)} leaf nodes")
    
    # Create a random opponent strategy for team 24
    print("\nCreating opponent strategy...")
    opponent_strategy = Strategy(game, "24")
    opponent_strategy.random_strategy()
    print("Opponent strategy created")
    
    # Train MAPPO best response for team 13
    print("\nStarting MAPPO training...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trained_policy, rewards, env = train_mappo_best_response(
        game=game,
        opponent_strategy=opponent_strategy,
        learning_team="13",
        device=device,
        frames_per_batch=3000,  # Reduced for faster testing
        n_iters=50,  # Reduced for testing
        num_epochs=20,
        minibatch_size=200,
        lr=3e-4,
        use_mappo=True,  # Set to False for IPPO
        share_parameters_policy=True,
        share_parameters_critic=True,
    )
    
    print("\nDone!")