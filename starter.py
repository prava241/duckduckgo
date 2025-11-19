from itertools import permutations

class Node:
    def __init__(self, name, parent, player, actions):
        self.name = name
        self.parent = parent
        self.player = player
        self.actions = actions
        self.action_payoffs = {}
        self.infoset = None
        self.history = (self.parent.history + 
                        [] if self.parent.player == "C"
                        else [(self.parent.infoset, self.name[-1])])

class LeafNode:
    def __init__(self, parent, name):
        self.name = name
        def compute_payoff(node_str):
            pot = [1, 1, 1, 1]
            folds = set()
            rounds = node_str.split("C:")
            for r in range(len(rounds)):
                raises = rounds[r].split("R")
                for r in raises[1:]:
                    for p in range(4):
                        if "P" + str(p+1) + ":F" in r:
                            folds.add(p)
                        for player in pot:
                            if player not in folds:
                                pot[player] += 2*(r+1)
            cards = {"K" : 3, "Q" : 2, "J" : 1}
            cc = node_str.index("C:")
            if cc != -1:
                cards[node_str[cc + 2]] = 4
            best = 0
            best_players = [-1]
            for i in range(4):
                if i not in folds:
                    if cards[node_str[i]] > best:
                        best_players = [i]
                    elif cards[node_str[i]] == best:
                        best_players += [i]
            
            pot = sum(pot.values())
            win = pot/len(best_players)
            team_payoffs = {"13" : sum([win for p in (0, 2) if p in best_players]) - sum([pot[p] for p in (0, 2)]), 
                            "24" : sum([win for p in (1, 3) if p in best_players]) - sum([pot[p] for p in (1, 3)])}
            return team_payoffs
        self.parent = parent
        self.payoff = compute_payoff(name)
        self.history = (self.parent.history + 
                        [] if self.parent.player == "C"
                        else [(self.parent.infoset, self.name[-1])])
        # note that chance is scaled, so the payoffs will be scaled
        s = name[:4]
        self.chance = 1 if ("J" not in s) or ("Q" not in s) or ("K" not in s) else 2
        cc = name.find("C:")
        if cc != -1:
            cc = name[cc+2]
            if cc in s:
                self.chance *= 2
        
# class Infoset:
#     def __init__(self, player, nodes, actions):
#         self.nodes = {}
#         for node in nodes:
#             self.nodes[node] = 1/len(nodes)
#         self.player = player
#         self.actions = actions
    
#     def update_probs(self, new_probs):
#         self.nodes = dict.copy(new_probs)

class Game:
    def __init__(self):
        nodes = {} # string to node
        infosets = {p : {} for p in ["P1", "P2", "P3", "P4"]} # player to infoset to nodes in infoset

        def draw_cards(available_cards = {"J" : 2, "Q" : 2, "K" : 2}, num_cards = 4):
            deck = []
            for card, count in available_cards.items():
                deck.extend([card] * count)
            hands = set(permutations(deck, 3))
            return hands
        
        def create_all_nodes(infoset, player, actions):
            available_cards = {"J" : 2, "Q" : 2, "K" : 2}
            players = {"P1" : 0, "P2" : 1, "P3" : 2, "P4" : 3}
            p = players[player]
            cc = infoset.find("C:")
            if cc != -1:
                available_cards[infoset[cc + 2]] -= 1
            available_cards[infoset[p]] -= 1

            indices = [i for i in range(4) if i != (p)]

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
                    parent_node = Node(parent, nodes[parent2], "C", {}, None)
                    nodes[parent] = parent_node

                if node_str not in nodes:
                    node = Node(node_str, nodes[parent], player, actions, infoset)
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
        
        nodes[""] = Node("", None, "C", set(draw_cards(num_cards=3)), None) # TODO: add actions
        for infoset in sorted(infoset_lines.keys(), key = len):
            player = infoset_lines[infoset][0]
            actions = infoset_lines[infoset][1]
            infosets[player][infoset] = {"nodes" : create_all_nodes(infoset, player, actions), 
                                         "actions" : actions}

        self.nodes = nodes
        self.infosets = infosets
        self.root = ""
        self.leaves = set()

        for node_str, node in self.nodes.items():
            actions_dict = {}
            for action in node.actions:
                child = (node_str + "/" + node.player + ":" + action 
                         if node_str != "" 
                         else action)
                if child not in self.nodes:
                    leaf = LeafNode(node, child)
                    self.leaves.add(leaf)
                    actions_dict[action] = leaf
                    node.action_payoffs[action] = leaf.payoff
                else:
                    actions_dict[action] = nodes[child]
            node.actions = actions_dict

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






if __name__ == "__main__":
    game = Game()
    print(len(game.nodes), len(game.infosets))
    strategy = Strategy("13", game)
    strategy.uniform_strategy()
    print(len(strategy.strategy), len(strategy.strategy["P1"]))