from itertools import permutations
# import numpy as np
import zipfile
from typing import *
from collections import defaultdict
from enum import IntEnum, Enum, unique
from dataclasses import dataclass
# from nash_eq import RestrictedGame

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
        self.chance_actions : Dict[str, Node | LeafNode] = chance_actions
        self.action_payoffs : Dict[Action, Dict[str, float]] = {}
        self.infoset : str = infoset
        self.history : List[Tuple[Player, str, Action]] = (self.parent.history + 
                        ([(Player.Chance, None, None)] if self.parent.player == "C"
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
            [(Player.Chance, None, None)] if self.parent.player == "C"
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
                    assert ('/' in node_str and (node_str.split('/')[-1][:2] == "C:")), node_str
                    parent2 = '/'.join(parent.split('/')[:-1]) if '/' in parent else ""
                    parent_node = Node(parent, nodes[parent2], "C", {}, {node_str[-1] : None}, "")
                    nodes[parent] = parent_node

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
                    infoset_lines[parts[1]] = ("P" + str(player), parts[2])
        
        nodes[""] = Node("", None, "C", {}, {}, "") # TODO: add actions
        for infoset in sorted(infoset_lines.keys(), key = len):
            player = player_to_Player(infoset_lines[infoset][0])
            actions = actions_to_Actions(infoset_lines[infoset][1])
            infosets[player][infoset] = {"nodes" : create_all_nodes(infoset, player, {a : None for a in actions}), 
                                         "actions" : actions}

        self.nodes = nodes
        self.infosets = infosets
        self.root = ""
        self.leaves = set()

        for node_str, node in self.nodes.items():
            if node_str == "":
                node.chance_actions = {"".join(hand) : nodes["".join(hand)] for hand in draw_cards(num_cards=4)}
            elif node.player == "C":
                node.chance_actions = {a : nodes[node.name + "/C:" + a] for a in node.chance_actions}
            else:
                actions_dict = {}
                for action in node.actions:
                    child : str = node_str + "/" + str_player(node.player) + ":" + str_action(action)
                    if child not in self.nodes:
                        leaf = LeafNode(node, child)
                        self.leaves.add(leaf)
                        actions_dict[action] = leaf
                        node.action_payoffs[action] = leaf.payoff
                    else:
                        actions_dict[action] = nodes[child]
                node.actions = actions_dict

    def save_game(self, file):
        # save the game to json
        pass

class Strategy:
    def __init__(self, team):
        self.game = Game()
        self.team = [player_to_Player("P" + p) for p in team]
        self.opp_team = [p for p in players if p not in self.team]
        self.strategy = {player : {} for player in self.team}
        self.opp_strategy = {player : {} for player in self.opp_team}
    
    def uniform_strategy(self, players):
        for p in players:
            self.strategy[p] = {i : {a : 1/len(v["actions"]) for a in v["actions"]} for i, v in self.game.infosets[p].items()}
    
    # def load_strategy(self, filenames):
    # def save_strategy(self, filenames):
    
    def pure_br(self):
        def leaf_node_contributions(my_team=True, chance=False):
            contribs = {}
            visited = set()
            to_visit = [(self.game.nodes[""], 1)]
            team = self.team if my_team else self.opp_team

            while to_visit:
                next_node, prob = to_visit.pop()
                if next_node in visited:
                    continue
                if not next_node.history: # this only happens at the root node
                    to_visit += [(child, prob) for child in next_node.chance_actions.values()]
                else:
                    player, infoset, action = next_node.history[-1]
                    prob *= (self.strategy[player][infoset][action] 
                                if player in team else 1)
                    if isinstance(next_node, LeafNode):
                        contribs[next_node] = prob
                    else:
                        to_visit += [(child, prob) for child in next_node.actions.values()]
                        to_visit += [(child, prob) for child in next_node.chance_actions.values()]
            print(len(contribs), len(self.game.leaves))
            for leaf in self.game.leaves:
                if leaf not in contribs:
                    print(leaf.name)
            if chance:
                return {leaf : contribs[leaf] * leaf.chance for leaf in contribs}
            return contribs
        
        def construct_ilp(other_contributions):
            p1, p2 = self.opp_team[0], self.opp_team[1]
            player_infosets_1 = self.game.infosets[p1]
            player_infosets_2 = self.game.infosets[p2]

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
            
            leaf_to_y = {leaf : last_actions(leaf) for leaf in self.game.leaves}
            y = list(set(leaf_to_y.values()))
            k = len(y)
            y_ind = {y : i + m + n for i, y in enumerate(y)}
            leaf_to_y_ind = {leaf : y_ind[v] for leaf, v in leaf_to_y.items()}

            constraints = []
            row_bounds = []
            col_bounds = [(0, 1)] * (m + n + k)

            row = 0
            for p in self.opp_team:
                for infoset, v in self.game.infosets[p].items():
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
                row += 1
                constraints += [(row, x_ind[x2_val], 1), (row, i, -1)]
                row += 1

            target_coeffs = [0] * (m + n + k)
            for leaf, var in leaf_to_y_ind.items():
                target_coeffs[var] += other_contributions[leaf]

            return (x1, x2, y), (x_ind, y_ind), (constraints, row_bounds, col_bounds), target_coeffs
        
        other_contributions = leaf_node_contributions(my_team=True, chance=True)
        other_team = "13" if Player.One in self.opp_team else "24"
        other_contributions = {leaf : other_contributions[leaf] * leaf.payoff[other_team] for leaf in self.game.leaves}

        variables, var_indices, constraints, target = construct_ilp(other_contributions)
        print(len(var_indices), len(constraints))



if __name__ == "__main__":
    strategy = Strategy("13")
    strategy.uniform_strategy([Player.One, Player.Three])
    strategy.pure_br()