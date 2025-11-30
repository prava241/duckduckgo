from __future__ import annotations
from itertools import permutations
from typing import *
from enum import IntEnum, unique
from dataclasses import dataclass
import pickle
import numpy as np
import highspy

@unique
class Player(IntEnum):
    One = 0
    Two = 1
    Three = 2
    Four = 3
    Chance = 4

players = [Player.One, Player.Two, Player.Three, Player.Four]
player_to_Player = lambda player : {'P1': Player.One, 'P2': Player.Two, 'P3': Player.Three, 'P4': Player.Four, 'C': Player.Chance}[player]
str_player = lambda player : ['P1', 'P2', 'P3', 'P4', 'C'][player]

@unique
class Action(IntEnum):
    Call = 0
    Fold = 1
    Raise = 2

actions_to_Actions = lambda actions : [{'C': Action.Call, 'F': Action.Fold, 'R': Action.Raise}[ch] for ch in actions]
action_to_Action = lambda action : {'C': Action.Call, 'F': Action.Fold, 'R': Action.Raise}[action]
str_action = lambda action : ['C', 'F', 'R'][action]

class Node:
    def __init__(self, name, parent, player, actions, chance_actions, infoset):
        self.name : str = name
        self.parent : Node = parent
        self.player : Player = player
        self.actions : Dict[Action, Node | LeafNode] = actions
        self.chance_actions : Dict[str, Node | LeafNode | None] = chance_actions
        self.action_payoffs : Dict[Action, Dict[str, float]] = {}
        self.infoset : str = infoset
        self.history : List[Tuple[Player, str, Action]] = (self.parent.history + 
                        ([(Player.Chance, None, None)] if self.parent.player == Player.Chance
                        else [(self.parent.player, self.parent.infoset, action_to_Action(self.name[-1]))])) if self.parent else []

class LeafNode:
    def compute_payoff(self, card_values):
        pot = [1, 1, 1, 1]
        folds = set()
        max_bet = 1
        raise_amt = 2
        for player, _, action in self.history:
            if player == Player.Chance:
                raise_amt = 4
            else:
                if action == Action.Call:
                    pot[player] = max_bet
                elif action == Action.Raise:
                    max_bet += raise_amt
                    pot[player] = max_bet
                else:
                    folds.add(player)
        best, best_players = 0, []
        for p in players:
            if p not in folds:
                if card_values[self.name[p]] > best:
                    best_players = [p]
                elif card_values[self.name[p]] == best:
                    best_players += [p]
        pot_total = sum(pot)
        win = pot_total/len(best_players)
        team_payoffs = {"13" : sum([win for p in (0, 2) if p in best_players]) - sum([pot[p] for p in (0, 2)]), 
                        "24" : sum([win for p in (1, 3) if p in best_players]) - sum([pot[p] for p in (1, 3)])}
        return team_payoffs
    
    def __init__(self, parent, name):
        self.name : str = name
        self.parent : Node = parent
        self.history : List[Tuple[Player, str, Action]] = self.parent.history + (
            [(Player.Chance, None, None)] if self.parent.player == Player.Chance
            else [(self.parent.player, self.parent.infoset, action_to_Action(self.name[-1]))])
        s = name[:4]
        self.chance : float = 1 if ("J" not in s) or ("Q" not in s) or ("K" not in s) else 2
        cards = {"K" : 3, "Q" : 2, "J" : 1}
        cc = name.find("C:")
        if cc != -1:
            cc = name[cc+2]
            if cc in s:
                self.chance *= 2
            cards[cc] = 4
        else:
            self.chance *= 2

        self.payoff : Dict[str, float] = self.compute_payoff(cards)

class Game:
    def __init__(self):
        nodes : Dict[str, Node] = {}
        infosets : Dict[Player, Dict[str, Dict]] = {p : {} for p in players}

        def draw_cards(available_cards = {"J" : 2, "Q" : 2, "K" : 2}, num_cards = 4):
            deck = []
            for card, count in available_cards.items():
                deck.extend([card] * count)
            hands = set(permutations(deck, num_cards))
            return hands
        
        def create_all_nodes(infoset : str, player : Player, actions):
            available_cards = {"J" : 2, "Q" : 2, "K" : 2}
            cc = infoset.find("C:")
            if cc != -1:
                available_cards[infoset[cc + 2]] -= 1
            available_cards[infoset[player]] -= 1

            indices = [i for i in range(4) if i != player]

            hands = draw_cards(available_cards, 3)
            
            infoset_nodes = set()
            for hand in hands:
                node_str = (infoset[:indices[0]] + hand[0] + 
                    infoset[indices[0] + 1 : indices[1]] + hand[1] + 
                    infoset[indices[1] + 1 : indices[2]] + hand[2] + 
                    infoset[indices[2] + 1 :])
                parent = '/'.join(node_str.split('/')[:-1]) if '/' in node_str else ""

                # if the parent is not in nodes (this could happen since the node after player 4 plays before the community card comes out)
                # assert that the last element in split is a chance action
                if parent not in nodes:
                    parent2 = '/'.join(parent.split('/')[:-1]) if '/' in parent else ""
                    parent_node = Node(parent, nodes[parent2], Player.Chance, {}, {node_str[-1] : None}, "")
                    nodes[parent] = parent_node
                elif nodes[parent].player == Player.Chance:
                    nodes[parent].chance_actions[node_str[-1]] = None

                if node_str not in nodes:
                    node = Node(node_str, nodes[parent], player, actions, {}, infoset)
                    nodes[node_str] = node
                
                else:
                    print("node", node_str, "in multiple infosets")
                
                infoset_nodes.add(nodes[node_str])
            
            return infoset_nodes
                
        infoset_lines = {}

        for player in range(1, 5):
            with open(f"player_{player}_infosets.txt", "r") as infos:
                lines = infos.readlines()
                for line in lines:
                    parts = line.split()
                    infoset_lines[parts[1]] = ("P" + str(player), parts[2], int(parts[0]))
        
        nodes[""] = Node("", None, Player.Chance, {}, {}, "")
        for infoset in sorted(infoset_lines.keys(), key = len):
            player = player_to_Player(infoset_lines[infoset][0])
            actions = actions_to_Actions(infoset_lines[infoset][1])
            infosets[player][infoset] = {"nodes" : create_all_nodes(infoset, player, {a : None for a in actions}), 
                                         "actions" : actions,
                                         "line" : infoset_lines[infoset][2]}

        self.nodes = nodes
        self.infosets = infosets
        self.root = ""
        self.leaves = []

        for node_str, node in self.nodes.items():
            if node_str == "":
                node.chance_actions = {"".join(hand) : nodes["".join(hand)] for hand in draw_cards(num_cards=4)}
            elif node.player == Player.Chance:
                node.chance_actions = {a : nodes[node.name + "/C:" + a] for a in node.chance_actions}
            else:
                actions_dict = {}
                for action in node.actions:
                    child : str = node_str + "/" + str_player(node.player) + ":" + str_action(action)
                    if child not in self.nodes:
                        leaf = LeafNode(node, child)
                        self.leaves += [leaf]
                        actions_dict[action] = leaf
                        node.action_payoffs[action] = leaf.payoff
                    else:
                        actions_dict[action] = nodes[child]
                node.actions = actions_dict
        
        self.payoffs_13 = np.array([leaf.payoff["13"] for leaf in self.leaves])
        self.payoffs_24 = self.payoffs_13 * (-1)
        self.chances = np.array([leaf.chance for leaf in self.leaves])
        self.chance_payoffs_13 = self.chances * self.payoffs_13
        self.chance_payoffs_24 = self.chance_payoffs_13 * -1

    # def save_game(self, filename="game.pkl"):
    #     with open(filename, 'wb') as f:
    #         pickle.dump(self, f)

    def construct_constraints(self, team=(Player.One, Player.Three)):
        p1, p2 = team
        player_infosets_1 = self.infosets[p1]
        player_infosets_2 = self.infosets[p2]

        x1 = [(k, a) for k, v in player_infosets_1.items() for a in v["actions"]]
        x_ind = {x : i for i, x in enumerate(x1)}
        m = len(x1)
        x2 = [(k, a) for k, v in player_infosets_2.items() for a in v["actions"]]
        for i, x in enumerate(x2):
            x_ind[x] = i + m
        n = len(x2)
        
        def last_actions(leaf_node):
            last_action_1 = [(i, a) for (p, i, a) in leaf_node.history if p == p1][-1]
            last_action_2 = [(i, a) for (p, i, a) in leaf_node.history if p == p2][-1]
            return (last_action_1, last_action_2)
        
        leaf_to_y = [last_actions(leaf) for leaf in self.leaves]
        y = list(set(leaf_to_y))
        k = len(y)
        y_ind = {y : i + m + n for i, y in enumerate(y)}
        leaf_to_y_ind = [y_ind[action] for action in leaf_to_y]

        constraints = []
        row_bounds = []
        col_bounds = [(0, 1)] * (m + n + k)

        row = 0
        for p in team:
            for infoset, v in self.infosets[p].items():
                constraints += [(row, x_ind[(infoset, action)], 1) for action in v["actions"]]
                last_player_index = infoset.rfind(str_player(p))
                if last_player_index == -1:
                    row_bounds += [(1, 1)]
                else:
                    parent = (infoset[:last_player_index-1], action_to_Action(infoset[last_player_index + 3]))
                    constraints += [(row, x_ind[parent], -1)]
                    row_bounds += [(0, 0)]
            row += 1
        
        for x1_val, x2_val in y:
            i = y_ind[(x1_val, x2_val)]
            constraints += [(row, x_ind[x1_val], 1), (row, i, -1)]
            row_bounds += [(0, 1)]
            row += 1
            constraints += [(row, x_ind[x2_val], 1), (row, i, -1)]
            row_bounds += [(0, 1)]
            row += 1

        return ((x1, x2, y), (x_ind, y_ind), leaf_to_y_ind, (constraints, row_bounds, col_bounds))
        

class Strategy:
    def __init__(self, game, team : str, strategies: List[Tuple[Dict, float]]=[], seq_form=np.array([]), all_contribs=np.array([])):
        self.game = game
        self.team_name = team
        self.team = [player_to_Player("P" + p) for p in team]
        self.opp_team = [p for p in players if p not in self.team]
        self.strategies = strategies
        self.seq_form = seq_form
        self.all_contribs = all_contribs
        if len(strategies) > 0 and len(seq_form) == 0:
            self.calculate_contribs()

    def calculate_contribs(self):
        contribs = {}
        for strategy, strategy_prob in self.strategies:
            visited = set()
            to_visit = [(self.game.nodes[""], strategy_prob)]
            while to_visit:
                next_node, prob = to_visit.pop()
                if next_node in visited:
                    continue
                if not next_node.history:
                    to_visit += [(child, prob) for child in next_node.chance_actions.values()]
                else:
                    player, infoset, action = next_node.history[-1]
                    prob *= (strategy[player][infoset][action] 
                                if player in self.team else 1)
                    if isinstance(next_node, LeafNode):
                        if next_node in contribs:
                            contribs[next_node] += prob
                        else:
                            contribs[next_node] = prob
                    else:
                        to_visit += [(child, prob) for child in next_node.actions.values()]
                        to_visit += [(child, prob) for child in next_node.chance_actions.values()]
        self.seq_form = np.array([contribs[leaf] for leaf in self.game.leaves])
        self.all_contribs = ((self.game.chance_payoffs_24 * self.seq_form) 
                             if self.team_name == "13" 
                             else (self.game.chance_payoffs_13 * self.seq_form))
    
    def uniform_strategy(self):
        strategy = {}
        for p in self.team:
            strategy[p] = {i : {a : 1/len(v["actions"]) for a in v["actions"]} for i, v in self.game.infosets[p].items()}
        self.strategies = [(strategy, 1)]
        self.calculate_contribs()

    def random_strategy(self):
        strategy = {}
        for p in self.team:
            strategy[p] = {i : {a : 0 for a in v["actions"]} for i, v in self.game.infosets[p].items()}
            for i, d in strategy[p].items():
                a = np.random.choice(list(d.keys()))
                strategy[p][i][a] = 1
        self.strategies = [(strategy, 1)]
        self.calculate_contribs()
    
    # def load_strategy(self, filenames):
    # def save_strategy(self, filenames):
    
    def best_response(self, h, num_vars, leaf_to_y_ind) -> Strategy:
        target_coeffs = [0] * num_vars
        # TODO: should i add the constant to leaf payoffs or to target_coeffs?
        for i, var in enumerate(leaf_to_y_ind):
            target_coeffs[var] += self.all_contribs[i]

        # TODO: update h with target_coeffs, then run it, then parse output
        
        other_team_name = "24" if self.team_name == "13" else "13"
        return Strategy(game, other_team_name)



if __name__ == "__main__":
    game = Game()
    constraints_13 = game.construct_constraints((Player.One, Player.Three))
    print(len(constraints_13), constraints_13[0][0])
    constraints_24 = game.construct_constraints((Player.Two, Player.Four))
    strategy = Strategy(game, "24")
    strategy.uniform_strategy()
    strategy.best_response(constraints_13)