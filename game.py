from __future__ import annotations
from itertools import permutations
from typing import *
from enum import IntEnum, unique
from dataclasses import dataclass
import pickle
import numpy as np
from scipy.sparse import coo_matrix
import highspy
from collections import defaultdict

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

        constraints = defaultdict(list)
        row_bounds = []
        col_bounds = [(0, 1)] * (m + n + k)

        row = 0
        for p in team:
            for infoset, v in self.infosets[p].items():
                for action in v["actions"]:
                    constraints[row].append((x_ind[(infoset, action)], 1))
                last_player_index = infoset.rfind(str_player(p))
                if last_player_index == -1:
                    row_bounds += [(1, 1)]
                else:
                    parent = (infoset[:last_player_index-1], action_to_Action(infoset[last_player_index + 3]))
                    constraints[row].append((x_ind[parent], -1))
                    row_bounds += [(0, 0)]
                row += 1
        
        for x1_val, x2_val in y:
            i = y_ind[(x1_val, x2_val)]
            constraints[row] = [(x_ind[x1_val], 1), (i, -1)]
            row_bounds += [(0, 1)]
            row += 1
            constraints[row] = [(x_ind[x2_val], 1), (i, -1)]
            row_bounds += [(0, 1)]
            row += 1
            constraints[row] = [(i, 1), (x_ind[x1_val], -1), (x_ind[x2_val], -1)]
            row_bounds += [(-1, 1)]

        return ((x1, x2, y), (x_ind, y_ind), leaf_to_y_ind, (constraints, row_bounds, col_bounds))
    
    def construct_constraints_lp(self, team=(Player.One, Player.Three)):
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

        X = m + n
        N = 1 + X
        
        y_ij = np.arange(N * N).reshape(N, N)
        print("created y_ij")
        
        def last_actions(leaf_node):
            last_action_1 = [(i, a) for (p, i, a) in leaf_node.history if p == p1][-1]
            last_action_2 = [(i, a) for (p, i, a) in leaf_node.history if p == p2][-1]
            return y_ij[x_ind[last_action_1] + 1, x_ind[last_action_2] + 1]
        
        leaf_to_ind = [last_actions(leaf) for leaf in self.leaves]
        print("hello")

        original_constraints = defaultdict(list)
        orig_row_bounds = []
        # col_bounds = [(0, 1)] * (m + n + k)

        orig_row = 0
        for p in team:
            for infoset, v in self.infosets[p].items():
                for action in v["actions"]:
                    original_constraints[orig_row].append((x_ind[(infoset, action)], 1))
                last_player_index = infoset.rfind(str_player(p))
                if last_player_index == -1:
                    orig_row_bounds += [1]
                else:
                    parent = (infoset[:last_player_index-1], action_to_Action(infoset[last_player_index + 3]))
                    original_constraints[orig_row].append((x_ind[parent], -1))
                    orig_row_bounds += [0]
                orig_row += 1
        print("created original constraints")

        row = 0

        # Flow Constraints:
        # There are (1 + m + n)^2 Y-variables. There are row original constraints.
        # There are 2(m + n)(row + m + n) new flow constraints.
        # For each original constraint, add the new constraint for all cols
        # For each col/col diff, add the x <= x_0

        constraints = defaultdict(list)
        row_bounds = []
        for orig_row, bound in enumerate(orig_row_bounds):
            assert (bound in (0, 1)), "original row bound not in \{0, 1\}"
            if bound == 0:
                for matrix_col in range(1, N):
                    constraints[row] = [(y_ij[ind + 1, matrix_col], weight) 
                                        for ind, weight in original_constraints[orig_row]]
                    row += 1
                    constraints[row] = ([(y_ij[ind + 1, matrix_col], weight) 
                                        for ind, weight in original_constraints[orig_row]]
                                        + [(y_ij[ind + 1, 0], -weight) 
                                        for ind, weight in original_constraints[orig_row]])
                    row += 1
            else:
                for matrix_col in range(1, N):
                    constraints[row] = ([(y_ij[ind + 1, matrix_col], weight) 
                                        for ind, weight in original_constraints[orig_row]]
                                        + [(y_ij[0, matrix_col], -1)])
                    row += 1
                    constraints[row] = ([(y_ij[ind + 1, matrix_col], -weight) 
                                        for ind, weight in original_constraints[orig_row]]
                                        + [(y_ij[ind + 1, 0], weight) 
                                        for ind, weight in original_constraints[orig_row]]
                                        + [(y_ij[0, matrix_col], 1), (0, -1)])
                    row += 1
            print(orig_row)
        assert (row == 2*X*(orig_row + X)), "incorrect number of flow constraints"
        row_bounds += [(0, 0) for _ in range(row)]
        print(f"finished flow constraints part 1 {row}")
        for matrix_col in range(1, N):
            # x <= x_0 -> x_0 - x >= 0
            for matrix_row in range(1, N):
                constraints[row] = [(y_ij[0, matrix_col], 1),
                                    (y_ij[matrix_row, matrix_col], -1)]
                row += 1
                constraints[row] = [(0, 1), (y_ij[matrix_row, 0], -1),
                                    (y_ij[0, matrix_col], -1), 
                                    (y_ij[matrix_row, matrix_col], 1)]
                row += 1
        assert (row == 2*X*(orig_row + X) + 2*(N**2)), "incorrect number of flow constraints"
        row_bounds += [(0, 2) for _ in range(2*(N**2))]
        print(f"finished flow constraints part 2 {row}")

        # Symmetry Constraints:
        for matrix_row in range(1, N):
            for matrix_col in range(matrix_row, N):
                constraints[row] = [(y_ij[matrix_row, matrix_col], 1),
                                    (y_ij[matrix_col, matrix_row], -1)]
                row += 1
        row_bounds += [(0, 0) for _ in range(X*N//2)]
        print(f"finished symmetry constraints {row}")

        # Diagonal Constraints:
        for matrix_row in range(1, N):
            constraints[row] = [(y_ij[matrix_row, 0], 1),
                                (y_ij[matrix_row, matrix_row], -1)]
            row += 1
        row_bounds += [(0, 0) for _ in range(X)]
        constraints[row] = [(0, 1)]
        row_bounds += [(1, 1)]
        print(f"finished diagonal constraints {row}")
        
        print(f"{row} rows, {N**2} cols")

        return (x1, x2), x_ind, leaf_to_ind, (constraints, row_bounds)

    def construct_constraints_lp_fast(self, team=(Player.One, Player.Three)):
        p1, p2 = team
        player_infosets_1 = self.infosets[p1]
        player_infosets_2 = self.infosets[p2]

        # -----------------------------
        # 1. Create x variables
        # -----------------------------
        x1 = [(k, a) for k, v in player_infosets_1.items() for a in v["actions"]]
        x2 = [(k, a) for k, v in player_infosets_2.items() for a in v["actions"]]
        x_ind = {x: i for i, x in enumerate(x1)}
        offset = len(x1)
        for i, x in enumerate(x2):
            x_ind[x] = i + offset

        m, n = len(x1), len(x2)
        X = m + n
        N = 1 + X  # including x_0

        # Precompute y_ij indices in a N x N matrix
        Y_index = np.arange(N * N).reshape(N, N)
        def y_ij(i, j):
            return Y_index[i, j]
        print("created Y_index")

        # -----------------------------
        # 2. leaf_to_ind
        # -----------------------------
        def last_actions(leaf_node):
            last_action_1 = [(i, a) for (p, i, a) in leaf_node.history if p == p1][-1]
            last_action_2 = [(i, a) for (p, i, a) in leaf_node.history if p == p2][-1]
            return y_ij(x_ind[last_action_1] + 1, x_ind[last_action_2] + 1)

        leaf_to_ind = [last_actions(leaf) for leaf in self.leaves]

        # -----------------------------
        # 3. original constraints
        # -----------------------------
        original_constraints = []
        orig_row_bounds = []

        for p in team:
            for infoset, v in self.infosets[p].items():
                row_entries = []
                for action in v["actions"]:
                    row_entries.append((x_ind[(infoset, action)], 1))
                last_player_index = infoset.rfind(str_player(p))
                if last_player_index == -1:
                    orig_row_bounds.append(1)
                else:
                    parent = (infoset[:last_player_index-1],
                            action_to_Action(infoset[last_player_index + 3]))
                    row_entries.append((x_ind[parent], -1))
                    orig_row_bounds.append(0)
                original_constraints.append(row_entries)

        num_orig_rows = len(orig_row_bounds)
        print("created original constraints")

        # -----------------------------
        # 4. Estimate number of rows
        # -----------------------------
        num_flow_rows_part1 = 2 * X * (num_orig_rows + X)
        num_flow_rows_part2 = 2 * (N ** 2)
        num_symmetry_rows = X * N // 2
        num_diag_rows = X + 1  # plus the final x0 row
        total_rows = num_flow_rows_part1 + num_flow_rows_part2 + num_symmetry_rows + num_diag_rows

        rows, cols, data = [], [], []
        row_bounds = []

        row = 0

        # -----------------------------
        # 5. Flow constraints - part 1
        # -----------------------------
        for orig_row, bound in enumerate(orig_row_bounds):
            row_entries = np.array(original_constraints[orig_row], dtype=int)
            indices, weights = row_entries[:, 0], row_entries[:, 1]
            if bound == 0:
                for matrix_col in range(1, N):
                    # first constraint
                    rows.extend([row]*len(indices))
                    cols.extend([y_ij(ind+1, matrix_col) for ind in indices])
                    data.extend(weights.tolist())
                    row_bounds.append((0, 0))
                    row += 1

                    # second constraint
                    rows.extend([row]*len(indices)*2)
                    cols.extend([y_ij(ind+1, matrix_col) for ind in indices] +
                                [y_ij(ind+1, 0) for ind in indices])
                    data.extend(weights.tolist() + [-w for w in weights.tolist()])
                    row_bounds.append((0, 0))
                    row += 1
            else:
                for matrix_col in range(1, N):
                    # first constraint
                    rows.extend([row]*len(indices) + [row])
                    cols.extend([y_ij(ind+1, matrix_col) for ind in indices] + [y_ij(0, matrix_col)])
                    data.extend(weights.tolist() + [-1])
                    row_bounds.append((0, 0))
                    row += 1

                    # second constraint
                    rows.extend([row]*(len(indices)*2 + 2))
                    cols.extend([y_ij(ind+1, matrix_col) for ind in indices] +
                                [y_ij(ind+1, 0) for ind in indices] +
                                [y_ij(0, matrix_col), y_ij(0, 0)])
                    data.extend([-w for w in weights.tolist()] +
                                weights.tolist() +
                                [1, -1])
                    row_bounds.append((0, 0))
                    row += 1

        print(f"finished flow constraints part 1 {row}")

        # -----------------------------
        # 6. Flow constraints - part 2 (x <= x0)
        # -----------------------------
        for matrix_col in range(1, N):
            for matrix_row in range(1, N):
                # x0 - x >= 0
                rows.extend([row, row])
                cols.extend([y_ij(0, matrix_col), y_ij(matrix_row, matrix_col)])
                data.extend([1, -1])
                row_bounds.append((0, 2))
                row += 1

                # second variant
                rows.extend([row]*4)
                cols.extend([0, y_ij(matrix_row, 0), y_ij(0, matrix_col), y_ij(matrix_row, matrix_col)])
                data.extend([1, -1, -1, 1])
                row_bounds.append((0, 2))
                row += 1

        print(f"finished flow constraints part 2 {row}")

        # -----------------------------
        # 7. Symmetry constraints
        # -----------------------------
        r, c = np.triu_indices(N, 1)
        for i, j in zip(r, c):
            if i == 0:  # skip x0 row/col
                continue
            rows.extend([row, row])
            cols.extend([y_ij(i, j), y_ij(j, i)])
            data.extend([1, -1])
            row_bounds.append((0, 0))
            row += 1

        print(f"finished symmetry constraints {row}")

        # -----------------------------
        # 8. Diagonal constraints
        # -----------------------------
        for matrix_row in range(1, N):
            rows.extend([row, row])
            cols.extend([y_ij(matrix_row, 0), y_ij(matrix_row, matrix_row)])
            data.extend([1, -1])
            row_bounds.append((0, 0))
            row += 1

        # final x0 constraint
        rows.append(row)
        cols.append(0)
        data.append(1)
        row_bounds.append((1, 1))
        row += 1

        print(f"finished diagonal constraints {row}")
        print(f"{row} rows, {N**2} cols")

        # Build sparse representation
        constraints = coo_matrix((data, (rows, cols)), shape=(row, N*N))

        return (x1, x2), x_ind, leaf_to_ind, (constraints, row_bounds) 

    def construct_constraints_lp_fast_vectorized(self, team=(Player.One, Player.Three)):
        p1, p2 = team
        player_infosets_1 = self.infosets[p1]
        player_infosets_2 = self.infosets[p2]

        # -----------------------------
        # 1. x variables
        # -----------------------------
        x1 = [(k, a) for k, v in player_infosets_1.items() for a in v["actions"]]
        x2 = [(k, a) for k, v in player_infosets_2.items() for a in v["actions"]]
        x_ind = {x: i for i, x in enumerate(x1)}
        offset = len(x1)
        for i, x in enumerate(x2):
            x_ind[x] = i + offset

        m, n = len(x1), len(x2)
        X = m + n
        N = 1 + X  # including x_0

        # Precompute Y indices
        Y_index = np.arange(N * N).reshape(N, N)
        def y_ij(i, j):
            return Y_index[i, j]
        print("Y_index")

        # -----------------------------
        # 2. leaf_to_ind
        # -----------------------------
        def last_actions(leaf_node):
            last_action_1 = [(i, a) for (p, i, a) in leaf_node.history if p == p1][-1]
            last_action_2 = [(i, a) for (p, i, a) in leaf_node.history if p == p2][-1]
            return y_ij(x_ind[last_action_1] + 1, x_ind[last_action_2] + 1)

        leaf_to_ind = [last_actions(leaf) for leaf in self.leaves]

        # -----------------------------
        # 3. original constraints
        # -----------------------------
        original_constraints = []
        orig_row_bounds = []

        for p in team:
            for infoset, v in self.infosets[p].items():
                row_entries = []
                for action in v["actions"]:
                    row_entries.append((x_ind[(infoset, action)], 1))
                last_player_index = infoset.rfind(str_player(p))
                if last_player_index == -1:
                    orig_row_bounds.append(1)
                else:
                    parent = (infoset[:last_player_index-1],
                            action_to_Action(infoset[last_player_index + 3]))
                    row_entries.append((x_ind[parent], -1))
                    orig_row_bounds.append(0)
                original_constraints.append(row_entries)

        num_orig_rows = len(orig_row_bounds)
        print("created original constraints")

        # -----------------------------
        # 4. Prepare for sparse matrix building
        # -----------------------------
        row_ptrs, col_ptrs, data_ptrs = [], [], []
        row_bounds = []

        row_counter = 0

        # -----------------------------
        # 5. Flow constraints - vectorized
        # -----------------------------
        # Precompute Y indices for all rows and columns
        y_rows = np.arange(N).reshape(-1, 1)  # broadcastable row indices
        y_cols = np.arange(N).reshape(1, -1)  # broadcastable col indices

        # For each original constraint
        for orig_row, bound in enumerate(orig_row_bounds):
            entries = np.array(original_constraints[orig_row], dtype=int)
            inds, weights = entries[:,0], entries[:,1]

            # Broadcast over matrix_col
            matrix_cols = np.arange(1, N)
            if bound == 0:
                # Constraint 1: y_ij * weight
                rr, cc = np.meshgrid(matrix_cols, np.arange(len(inds)), indexing='ij')
                row_ptrs.extend(list(row_counter + rr.flatten()))
                col_ptrs.extend([y_ij(inds[c]+1, matrix_cols[r]) for r in range(len(matrix_cols)) for c in range(len(inds))])
                data_ptrs.extend(np.tile(weights, len(matrix_cols)))
                row_bounds.extend([(0,0)] * (len(matrix_cols)))
                row_counter += len(matrix_cols)

                # Constraint 2: y_ij * weight + y_i0 * -weight
                row_ptrs.extend(list(row_counter + rr.flatten()))
                col_ptrs.extend([y_ij(inds[c]+1, matrix_cols[r]) for r in range(len(matrix_cols)) for c in range(len(inds))] +
                                [y_ij(inds[c]+1, 0) for r in range(len(matrix_cols)) for c in range(len(inds))])
                data_ptrs.extend(np.tile(weights, len(matrix_cols)).tolist() +
                                (-np.tile(weights, len(matrix_cols))).tolist())
                row_bounds.extend([(0,0)] * (len(matrix_cols)))
                row_counter += len(matrix_cols)

            else:
                # Constraint 1: y_ij * weight + y_0j * -1
                rr, cc = np.meshgrid(matrix_cols, np.arange(len(inds)), indexing='ij')
                row_ptrs.extend(list(row_counter + rr.flatten()) + [row_counter + i for i in range(len(matrix_cols))])
                col_ptrs.extend([y_ij(inds[c]+1, matrix_cols[r]) for r in range(len(matrix_cols)) for c in range(len(inds))] +
                                [y_ij(0, mc) for mc in matrix_cols])
                data_ptrs.extend(np.tile(weights, len(matrix_cols)).tolist() + [-1]*len(matrix_cols))
                row_bounds.extend([(0,0)]*len(matrix_cols))
                row_counter += len(matrix_cols)

                # Constraint 2
                row_ptrs.extend(list(row_counter + rr.flatten())*2 + [row_counter + i for i in range(2*len(matrix_cols))])
                col_ptrs.extend([y_ij(inds[c]+1, matrix_cols[r]) for r in range(len(matrix_cols)) for c in range(len(inds))] +
                                [y_ij(inds[c]+1, 0) for r in range(len(matrix_cols)) for c in range(len(inds))] +
                                [y_ij(0, mc) for mc in matrix_cols] + [y_ij(0,0) for mc in matrix_cols])
                data_ptrs.extend((-np.tile(weights, len(matrix_cols))).tolist() +
                                np.tile(weights, len(matrix_cols)).tolist() +
                                [1]*len(matrix_cols) + [-1]*len(matrix_cols))
                row_bounds.extend([(0,0)]*len(matrix_cols))
                row_counter += len(matrix_cols)

        print(f"finished vectorized flow constraints part 1, row_counter={row_counter}")

        # -----------------------------
        # 6. Flow constraints - part 2 (x <= x0)
        # -----------------------------
        mc = np.arange(1, N)
        mr = np.arange(1, N)
        mr_grid, mc_grid = np.meshgrid(mr, mc, indexing='ij')

        # Constraint 1: x0 - x >= 0
        row_ptrs.extend((row_counter + np.arange(mr_grid.size)).tolist())
        col_ptrs.extend([y_ij(0, c) for c in mc_grid.flatten()] + [y_ij(r, c) for r,c in zip(mr_grid.flatten(), mc_grid.flatten())])
        data_ptrs.extend([1]*mr_grid.size + [-1]*mr_grid.size)
        row_bounds.extend([(0,2)]*mr_grid.size)
        row_counter += mr_grid.size

        # Constraint 2: variant
        row_ptrs.extend((row_counter + np.arange(mr_grid.size)).tolist())
        col_ptrs.extend([0]*mr_grid.size + [y_ij(r,0) for r in mr_grid.flatten()] + [y_ij(0,c) for c in mc_grid.flatten()] + 
                        [y_ij(r,c) for r,c in zip(mr_grid.flatten(), mc_grid.flatten())])
        data_ptrs.extend([1]*mr_grid.size + [-1]*mr_grid.size + [-1]*mr_grid.size + [1]*mr_grid.size)
        row_bounds.extend([(0,2)]*mr_grid.size)
        row_counter += mr_grid.size

        print(f"finished vectorized flow constraints part 2, row_counter={row_counter}")

        # -----------------------------
        # 7. Symmetry constraints
        # -----------------------------
        r, c = np.triu_indices(N, 1)
        mask = r>0
        r, c = r[mask], c[mask]
        row_ptrs.extend(list(range(row_counter, row_counter + len(r)))*2)
        col_ptrs.extend([y_ij(i,j) for i,j in zip(r,c)] + [y_ij(j,i) for i,j in zip(r,c)])
        data_ptrs.extend([1]*len(r) + [-1]*len(r))
        row_bounds.extend([(0,0)]*len(r))
        row_counter += len(r)

        print(f"finished symmetry constraints, row_counter={row_counter}")

        # -----------------------------
        # 8. Diagonal constraints
        # -----------------------------
        for matrix_row in range(1, N):
            row_ptrs.extend([row_counter]*2)
            col_ptrs.extend([y_ij(matrix_row, 0), y_ij(matrix_row, matrix_row)])
            data_ptrs.extend([1, -1])
            row_bounds.append((0,0))
            row_counter += 1

        # final x0 row
        row_ptrs.append(row_counter)
        col_ptrs.append(0)
        data_ptrs.append(1)
        row_bounds.append((1,1))
        row_counter += 1

        print(f"finished diagonal constraints, total rows={row_counter}, cols={N*N}")

        # -----------------------------
        # 9. Build sparse matrix
        # -----------------------------
        constraints = coo_matrix((data_ptrs, (row_ptrs, col_ptrs)), shape=(row_counter, N*N))

        return (x1, x2), x_ind, leaf_to_ind, (constraints, row_bounds)

    '''
    def construct_constraints_lp2(self, h, team=(Player.One, Player.Three)):
        p1, p2 = team
        player_infosets_1 = self.infosets[p1]
        player_infosets_2 = self.infosets[p2]

        x1 = [(k, a) for k, v in player_infosets_1.items() for a in v["actions"]]
        x_ind = {x : i + 1 for i, x in enumerate(x1)}
        m = len(x1)
        x2 = [(k, a) for k, v in player_infosets_2.items() for a in v["actions"]]
        for i, x in enumerate(x2):
            x_ind[x] = i + m + 1
        n = len(x2)

        X = m + n
        N = 1 + X
        
        y_ij = np.arange(N * N).reshape(N, N)
        print("created y_ij")
        
        def last_actions(leaf_node):
            last_action_1 = [(i, a) for (p, i, a) in leaf_node.history if p == p1][-1]
            last_action_2 = [(i, a) for (p, i, a) in leaf_node.history if p == p2][-1]
            return y_ij[x_ind[last_action_1], x_ind[last_action_2]]
        
        leaf_to_ind = [last_actions(leaf) for leaf in self.leaves]
        print("hello")

        original_constraints = defaultdict(list)
        # orig_row_bounds = []
        # col_bounds = [(0, 1)] * (m + n + k)

        orig_row = 0
        for p in team:
            for infoset, v in self.infosets[p].items():
                for action in v["actions"]:
                    original_constraints[orig_row].append((x_ind[(infoset, action)], 1))
                last_player_index = infoset.rfind(str_player(p))
                if last_player_index == -1:
                    original_constraints[orig_row].append((0, -1))
                else:
                    parent = (infoset[:last_player_index-1], action_to_Action(infoset[last_player_index + 3]))
                    original_constraints[orig_row].append((x_ind[parent], -1))
                orig_row += 1
        print("created original constraints")

        # Flow Constraints:
        # There are (1 + m + n)^2 Y-variables. There are row original constraints.
        # There are 2(m + n)(row + m + n) new flow constraints.
        # For each original constraint, add the new constraint for all cols
        # For each col/col diff, add the x <= x_0

        constraints = defaultdict(list)
        row_bounds = []
        for row in range(orig_row):
            # Ax = 0
            # h.addRows(2X, lower=0*2X, upper=0*2X, num_nz=3*X*s, 
            # start = [0, s, ..., (X-1)s] + [Xs, (X+2)s, ... (2X-2)s], 
            # index = , value)
            # ig we'll do col and diff_col together
            inds = [i for i, _ in constraints[row]]
            weights = [w for _, w in constraints[row]]
            s = len(inds)

            lower = np.zeros(2*X, dtype=np.float64)
            upper = np.zeros(2*X, dtype=np.float64)

            start = np.empty(2*X, dtype=np.float64)
            rangeX = np.arange(X)
            start[:X] = rangeX * s
            start[X:] = (X + 2*rangeX) * s
            # X*s (for each col) + X*2s (for each diff_col)
            for matrix_col in range(1, N):
                constraints[row] = [(y_ij[ind + 1, matrix_col], weight) 
                                    for ind, weight in original_constraints[orig_row]]
                constraints[row] = ([(y_ij[ind + 1, matrix_col], weight) 
                                    for ind, weight in original_constraints[orig_row]]
                                    + [(y_ij[ind + 1, 0], -weight) 
                                    for ind, weight in original_constraints[orig_row]])
            print(orig_row)
        assert (row == 2*X*(orig_row + X)), "incorrect number of flow constraints"
        row_bounds += [(0, 0) for _ in range(row)]
        print(f"finished flow constraints part 1 {row}")
        for matrix_col in range(1, N):
            # x <= x_0 -> x_0 - x >= 0
            for matrix_row in range(1, N):
                constraints[row] = [(y_ij[0, matrix_col], 1),
                                    (y_ij[matrix_row, matrix_col], -1)]
                row += 1
                constraints[row] = [(0, 1), (y_ij[matrix_row, 0], -1),
                                    (y_ij[0, matrix_col], -1), 
                                    (y_ij[matrix_row, matrix_col], 1)]
                row += 1
        assert (row == 2*X*(orig_row + X) + 2*(N**2)), "incorrect number of flow constraints"
        row_bounds += [(0, 2) for _ in range(2*(N**2))]
        print(f"finished flow constraints part 2 {row}")

        # Symmetry Constraints:
        for matrix_row in range(1, N):
            for matrix_col in range(matrix_row, N):
                constraints[row] = [(y_ij[matrix_row, matrix_col], 1),
                                    (y_ij[matrix_col, matrix_row], -1)]
                row += 1
        row_bounds += [(0, 0) for _ in range(X*N//2)]
        print(f"finished symmetry constraints {row}")

        # Diagonal Constraints:
        for matrix_row in range(1, N):
            constraints[row] = [(y_ij[matrix_row, 0], 1),
                                (y_ij[matrix_row, matrix_row], -1)]
            row += 1
        row_bounds += [(0, 0) for _ in range(X)]
        constraints[row] = [(0, 1)]
        row_bounds += [(1, 1)]
        print(f"finished diagonal constraints {row}")
        
        print(f"{row} rows, {N**2} cols")

        return (x1, x2), x_ind, leaf_to_ind, (constraints, row_bounds)
    '''

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
    
    def best_response(self, h, x1, x2, y, num_vars, leaf_to_y_ind) -> Strategy:
        target_coeffs = [0] * num_vars
        for i, var in enumerate(leaf_to_y_ind):
            target_coeffs[var] += self.all_contribs[i]

        m = min(target_coeffs)
        if m <= 0:
            target_coeffs = [t - m + 0.1 for t in target_coeffs]

        for i, t in enumerate(target_coeffs):
            h.changeColCost(i, t)
        
        print(f"Running best response against team {self.team} strategy")
        h.maximize()

        solution = h.getSolution()
        col_value = list(solution.col_value)
        value = [col_value[icol]
                for icol in range(num_vars)]
        
        strategy = {}
        player_strategy_1 = defaultdict(dict)
        for i, (infoset, action) in enumerate(x1):
            player_strategy_1[infoset][action] = value[i]
        strategy[self.opp_team[0]] = player_strategy_1
        
        player_strategy_2 = defaultdict(dict)
        for i, (infoset, action) in enumerate(x2):
            player_strategy_2[infoset][action] = value[i + len(x1)]
        strategy[self.opp_team[1]] = player_strategy_2

        seq_form = np.array([value[leaf_to_y_ind[i]] for i in range(len(self.game.leaves))])
        all_contribs = ((self.game.chance_payoffs_13 * seq_form) 
                        if self.team_name == "13" 
                        else (self.game.chance_payoffs_24 * seq_form))
        
        other_team_name = "24" if self.team_name == "13" else "13"
        return Strategy(self.game,
                        other_team_name,
                        strategies=[(strategy, 1)],
                        seq_form=seq_form,
                        all_contribs=all_contribs
                        )

    # def best_response(self, h, x1, x2, leaf_to_ind) -> Strategy:
    #     N = len(x1) + len(x2) + 1
    #     target_coeffs = [0] * (N**2)
    #     for i, ind in enumerate(leaf_to_ind):
    #         target_coeffs[ind] += self.all_contribs[i]

    #     # m = min(target_coeffs)
    #     # if m <= 0:
    #     #     target_coeffs = [t - m + 0.1 for t in target_coeffs]
    #     # I think since we have McCormick envelope, no need to be maximizing.

    #     for i, t in enumerate(target_coeffs):
    #         h.changeColCost(i, t)
        
    #     print(f"Running best response against team {self.team} strategy")
    #     h.maximize()

    #     solution = h.getSolution()
    #     col_value = list(solution.col_value)
    #     value = [col_value[icol]
    #             for icol in range(N**2)]
        
    #     strategy = {}
    #     player_strategy_1 = defaultdict(dict)
    #     for i, (infoset, action) in enumerate(x1):
    #         player_strategy_1[infoset][action] = value[i + 1]
    #     strategy[self.opp_team[0]] = player_strategy_1
        
    #     player_strategy_2 = defaultdict(dict)
    #     for i, (infoset, action) in enumerate(x2):
    #         player_strategy_2[infoset][action] = value[i + len(x1) + 1]
    #     strategy[self.opp_team[1]] = player_strategy_2

    #     seq_form = np.array([value[leaf_to_ind[i]] for i in range(len(self.game.leaves))])
    #     all_contribs = ((self.game.chance_payoffs_13 * seq_form) 
    #                     if self.team_name == "13" 
    #                     else (self.game.chance_payoffs_24 * seq_form))
        
    #     other_team_name = "24" if self.team_name == "13" else "13"
    #     return Strategy(self.game,
    #                     other_team_name,
    #                     strategies=[(strategy, 1)],
    #                     seq_form=seq_form,
    #                     all_contribs=all_contribs
    #                     )



if __name__ == "__main__":
    game = Game()
    print(sum([leaf.chance for leaf in game.leaves]))
    print(len(game.leaves))
    # constraints_13 = game.construct_constraints((Player.One, Player.Three))
    # constraints_24 = game.construct_constraints((Player.Two, Player.Four))
    # strategy = Strategy(game, "24")
    # strategy.random_strategy()
    # print(strategy.seq_form)