from itertools import permutations
import numpy as np
import zipfile
from typing import *
from collections import defaultdict
from enum import Enum, unique
from dataclasses import dataclass
from nash_eq import RestrictedGame

@unique
class Player(Enum):
    One = 0
    Two = 1
    Three = 2
    Four = 3
    Chance = 4

players = [Player.One, Player.Two, Player.Three, Player.Four]
player_to_Player = lambda player : {'P1': Player.One, 'P2': Player.Two, 'P3': Player.Three, 'P4': Player.Four, 'C': Player.Chance}[player]

@unique
class Action(Enum):
    Call = 1
    Fold = 2
    Raise = 3

actions_to_Actions = lambda actions : [{'C': Action.Call, 'F': Action.Fold, 'R': Action.Raise}[ch] for ch in actions]

class Node:
    def __init__(self, name, parent, player, actions):
        self.name: str = name
        self.parent: Node = parent
        self.player: Player = player_to_Player(player)
        self.actions: List[Action] = actions_to_Actions(actions)

class Infoset:
    def __init__(self, player, nodes, actions):
        self.nodes : Dict[Node, float] = {}
        for node in nodes:
            self.nodes[node] = 1/len(nodes)
        self.player: Player = player_to_Player(player)
        self.actions: List[Action] = actions_to_Actions(actions)
    
    def update_probs(self, new_probs):
        self.nodes = dict.copy(new_probs)

class Game:
    def __init__(self):
        nodes : Dict[str, Node] = {}
        infosets : Dict[Player, Dict[str, Infoset]] = defaultdict(dict)

        def create_all_nodes(infoset, player, actions):
            available_cards = {"J" : 2, "Q" : 2, "K" : 2}
            players = {"P1" : 0, "P2" : 1, "P3" : 2, "P4" : 3}
            p = players[player]
            cc = infoset.find("C:")
            if cc != -1:
                available_cards[infoset[cc + 2]] -= 1
            available_cards[infoset[p]] -= 1

            indices = [i for i in range(4) if i != (p)]

            deck = []
            for card, count in available_cards.items():
                deck.extend([card] * count)
            
            infoset_nodes = set()
            hands = set(permutations(deck, 3))
            for hand in hands:
                node_str = (infoset[:indices[0]] + hand[0] + 
                    infoset[indices[0] + 1 : indices[1]] + hand[1] + 
                    infoset[indices[1] + 1 : indices[2]] + hand[2] + 
                    infoset[indices[2] + 1 :])
                parent = '/'.join(node_str.split('/')[:-1]) if '/' in node_str else ""

                # if the parent is not in nodes (this could happen since the node after player 4 plays before the community card comes out)
                # assert that the last element in split is a chance action
                if parent not in nodes:
                    assert ('/' in node_str and (node_str.split('/')[-1][:2] == "C:")), node_str
                    parent2 = '/'.join(parent.split('/')[:-1]) if '/' in parent else ""
                    parent_node = Node(parent, nodes[parent2], "C", {})
                    nodes[parent] = parent_node

                if node_str not in nodes:
                    node = Node(node_str, nodes[parent], player, actions)
                    nodes[node_str] = node
                infoset_nodes.add(nodes[node_str])
            
            return infoset_nodes
                
        infoset_lines = {}

        for player in range(1, 5):
            with open(f"player_{player}_infosets.txt", "r") as infos:
                lines = infos.readlines()
                for line in lines:
                    parts = line.split()
                    infoset_lines[parts[1]] = ("P" + str(player), parts[2])
        
        nodes[""] = Node("", None, "C", {}) # TODO: add actions
        for infoset in sorted(infoset_lines.keys(), key = len):
            player = infoset_lines[infoset][0]
            actions = infoset_lines[infoset][1]
            infoset_nodes = create_all_nodes(infoset, player, actions)
            infosets[player_to_Player(player)][infoset] = Infoset(player, infoset_nodes, actions)

        self.nodes : Dict[str, Node] = nodes
        self.infosets : Dict[Player, Dict[str, Infoset]] = infosets
        self.root = ""

class Strategy:

    def __init__(self, team, game): 
        self.team: Tuple[Player] = team
        self.strategy: Dict[Player, Dict[Infoset, Dict[Action, float]]] = {}
        self.game: Game = game

    def uniform_strategy(self):
        for player in self.team:
            self.strategy[player] = {infoset : {a : 1/len(infoset.actions) for a in infoset.actions} for infoset in self.game.infosets[player].values()}
        
    # def load_strategy(self, filenames):
    # def save_strategy(self, filenames):
    # def best_response():

@dataclass(frozen=True)
class PureStrategy:
    strategy: Dict[str, Action]

class TeamDoubleOracle:

    def __init__(self, team: Tuple[Player], tolerance = 1.):
        self.populations: Dict[Player, List[PureStrategy]] = {}
        self.team: Tuple[Player]= team
        self.tolerance: float = tolerance
        self.utilities: Dict[Tuple[int, int, int, int], float] = {}

    def iterate(self, trial: int) -> Tuple[bool, List[float], List[float]]:
        team_mixture, opponent_mixture = self.get_mixtures()
        team_best_response, opponent_best_response = opponent_mixture.best_response(), team_mixture.best_response()
        
        best_responses = [None, None, None, None]
        if self.team == (Player.One, Player.Three):
            best_responses[0], best_responses[2] = team_best_response
            best_responses[1], best_responses[3] = opponent_best_response
        else:
            best_responses[0], best_responses[2] = opponent_best_response
            best_responses[1], best_responses[3] = team_best_response

        for player, best_response in zip(players, best_responses):
            self.populations[player].append(best_response)
        self.update_utilities(trial)
        
        eq_utility = self.utility(team_mixture, opponent_mixture)
        team_br_utility = self.utility(team_best_response, opponent_mixture)
        opponent_br_utility = self.utility(team_mixture, opponent_best_response)

        converged = abs(eq_utility - team_br_utility) < self.tolerance and abs(team_br_utility - opponent_br_utility) < self.tolerance
        return converged, team_mixture, opponent_mixture
    
    def update_utilities(self, trial: int):

        def pure_strategies_average_utility(p1_strat, p2_strat, p3_strat, p4_strat):
            u1, u2, u3, u4 = 0., 0., 0., 0.

            deck = ["J", "J", "Q", "Q", "K", "K"]
            deals = set(permutations(deck, 5))
    
            for p1c, p2c, p3c, p4c, cc in deals:
                

        def update_utilities_for_player(player: Player):
            other_players = players[:]
            other_players.remove(player)

            player_new_strat = self.populations[player][-1]
            for i, i_strat in enumerate(self.populations[Player.Two]):
                for j, j_strat in enumerate(self.populations[Player.Three]):
                    for k, k_strat in enumerate(self.populations[Player.Four]):
                        strats = [None, None, None, None]
                        strats[other_players[0]] = i_strat
                        strats[other_players[1]] = j_strat
                        strats[other_players[2]] = k_strat
                        strats[player] = player_new_strat

                        indices = [0,0,0,0]
                        strats[other_players[0]] = i
                        strats[other_players[1]] = j
                        strats[other_players[2]] = k
                        strats[player] = trial

                        self.utilities[tuple(indices)] = pure_strategies_average_utility(*strats)
        
        for player in players:
            update_utilities_for_player(player)
        return
        
    def get_mixtures(self):
        pass

    def mixture_to_tensors(self, mixture: List[float], team: Tuple[Player, Player]):
        strategies_1 = self.populations[team[0]]
        strategies_2 = self.populations[team[1]]

        n = len(mixture)
        assert len(strategies_1) == n
        assert len(strategies_2) == n

        infosets = list(strategies_1[0].mapping.keys())
        m = len(infosets)

        out_1 = np.zeros((m, 3), dtype=np.float64)
        out_2 = np.zeros((m, 3), dtype=np.float64)

        mix = np.asarray(mixture, dtype=np.float64)

        for mix_weight, strat1, strat2 in zip(mix, strategies_1, strategies_2):
            for i, infoset in enumerate(infosets):
                a1 = strat1.strategy[infoset].value - 1
                a2 = strat2.strategy[infoset].value - 1
                out_1[i, a1] += mix_weight
                out_2[i, a2] += mix_weight

        def normalize(arr: np.ndarray) -> np.ndarray:
            row_sums = arr.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            return arr / row_sums

        return normalize(out_1), normalize(out_2)

    def train(self, trials = 1000):
        self.utilities = [[0] * trials] * trials

        team_mixture, opponent_mixture = None, None
        for trial in range(trials):
            converged, team_mixture, opponent_mixture = self.iterate(trial)
            if converged:
                print(f"Converged at trial: {trial}")
                return team_mixture, opponent_mixture
            if trial % 100:
                print(f"Progress: {(100 * trial) / trials}%")
            
        return team_mixture, opponent_mixture
            

if __name__ == "__main__":
    game = Game()
    print(len(game.nodes), len(game.infosets))
    strategy = Strategy((Player.One, Player.Three), game)
    strategy.uniform_strategy()
    print(len(strategy.strategy), len(strategy.strategy[Player.One]))