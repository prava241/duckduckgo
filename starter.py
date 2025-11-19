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

'''
class Strategy:
    team_strategy = {} # dict from player to dict from infoset to actions and probs

    def __init__(self, team, game): 
        self.team = team # "13" or "24"
        self.strategy = {"P" + p : {} for p in team}
        self.game = game
    
    def uniform_strategy(self, players):
        for p in players:
            self.strategy[p] = {i : {a : 1/len(i["actions"]) for a in i["actions"]} for i in self.game.infosets[p]}
    
    # def load_strategy(self, filenames):
    # def save_strategy(self, filenames):

    def best_response(self):
        my_team = {"13" : ["P2", "P4"], "24" : ["P1", "P3"]}[self.team]
        swap_team = {"13" : "24", "24" : "13"}[self.team]
        opp_team = {"24" : ["P2", "P4"], "13" : ["P1", "P3"]}[self.team]

        # TODO: is there an issue with actions being greyed out here?
        def leaf_node_contributions(players):
            contribs = {}
            visited = set()
            to_visit = [(self.game.nodes[""], 1)]
            while to_visit:
                next_node, prob = to_visit.pop()
                if next_node not in visited:
                    if not next_node.history:
                        to_visit += [(child, prob) for child in next_node.actions]
                    else:
                        infoset, action = next_node.history[-1]
                        prob *= (self.strategy[infoset.player][infoset][action] 
                                 if infoset.player in players else 1)
                        if isinstance(next_node, LeafNode):
                            contribs[next_node] = prob
                        else:
                            to_visit += [(child, prob) for child in next_node.actions]
            if "C" in players:
                for leaf in contribs:
                    contribs[leaf] *= leaf.chance
            return contribs

        def create_constraints(player):
            # (row, col, coeff) for pbounds
            # (lower, upper) for row in rows [(0, 0)]
            # (lower, upper) for col in cols [(0.00001, 0)]
            player_infosets = self.game.infosets[player]

            variables = [(k, a) for k, v in player_infosets.items() for a in v["actions"]]
            vars_to_indices = {var : i for i, var in enumerate(variables)}
            vars_to_indices[""] = -1

            constraints = []
            row_bounds = []
            col_bounds = [(0.00001, 0) for _ in variables]

            # for each infoset, find the parent (infoset, action) and add a new tuple of (parent, infoset and all actions)
            row = 0
            for infoset, value in player_infosets:
                last_player_index = infoset.rfind(player)
                constraints += [(row, vars_to_indices[(infoset, a)], 1) for a in value["actions"]]
                if last_player_index == -1:
                    row_bounds += [(1, 1)]
                else:
                    parent = (infoset[:last_player_index-1], infoset[last_player_index + 3])
                    row_bounds += [(0, 0)]
                    constraints += [(row, vars_to_indices[parent], -1)]
                # if the parent index is -1, the vars on the RHS must add to 1, otherwise must add to the var on the LHS
            
            leaf_to_var = {}
            for leaf in opp_team_contributions:
                i = leaf.name.rfind(player)
                var = (leaf.name[:i-1], leaf.name[i+3])
                leaf_to_var[leaf] = vars_to_indices[var]

            return variables, constraints, row_bounds, col_bounds, leaf_to_var
        
        def create_target(variables, leaf_to_var, other_contributions):
            coeffs = [0 for _ in variables]
            for leaf, var in leaf_to_var.items():
                coeffs[var] += other_contributions[leaf]
            return coeffs

        def single_player_best_response(player):
            other_contributions = {leaf : opp_team_contributions[leaf] * teammate_contributions[leaf] 
                                   for leaf in teammate_contributions}
            # return target, 

        self.uniform_strategy(my_team)
        opp_team_contributions = leaf_node_contributions(["C"] + opp_team)
        opp_team_contributions = {leaf : opp_team_contributions * leaf.payoff[swap_team] for leaf in self.game.leaves}
        teammate_contributions = leaf_node_contributions([my_team[1]])
        constraints = [create_constraints(player) for player in my_team]

        old_payoff, payoff = -1000, -500
        threshold = 0.5
        while payoff - old_payoff > threshold:
            old_payoff = payoff
            single_player_best_response(my_team[0], my_team[1])
            payoff = single_player_best_response(my_team[1], my_team[0])
'''





if __name__ == "__main__":
    game = Game()
    print(len(game.nodes), len(game.infosets[Player.One]))
        
    # strategy = Strategy("13", game)
    # strategy.uniform_strategy()
    # print(len(strategy.strategy), len(strategy.strategy["P1"]))