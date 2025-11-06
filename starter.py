from itertools import permutations

class Node:
    def __init__(self, name, parent, player, actions):
        self.name = name
        self.parent = parent
        self.player = player
        self.actions = actions

    
class Infoset:
    def __init__(self, player, nodes, actions):
        self.nodes = {}
        for node in nodes:
            self.nodes[node] = 1/len(nodes)
        self.player = player
        self.actions = actions
    
    def update_probs(self, new_probs):
        self.nodes = dict.copy(new_probs)

class Game:
    def __init__(self):
        nodes = {} # string to node
        infosets = {} # string to infoset
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
            infosets[infoset] = Infoset(player, infoset_nodes, actions)

        self.nodes = nodes
        self.infosets = infosets
        self.root = ""

class Strategy:
    team_strategy = {} # dict from player to dict from infoset to actions and probs

    def __init__(self, team, game): 
        self.team = team # "13" or "24"
        self.strategy = {"P" + p : {} for p in team}
        self.game = game
    
    def uniform_strategy(self):
        for player in self.team:
            p = "P" + player
            self.strategy[p] = {infoset : {a : 1/len(infoset.actions) for a in infoset.actions} for infoset in self.game.infosets.values() if infoset.player == p}
    
    # def load_strategy(self, filenames):
    # def save_strategy(self, filenames):

    # def best_response():

if __name__ == "__main__":
    game = Game()
    print(len(game.nodes), len(game.infosets))
    strategy = Strategy("13", game)
    strategy.uniform_strategy()
    print(len(strategy.strategy), len(strategy.strategy["P1"]))