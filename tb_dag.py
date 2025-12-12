from itertools import permutations, product
import numpy as np
from typing import *
from dataclasses import dataclass
from game import *
import zipfile

class TBNode:
    def __init__(self, label: Tuple[str, ...], parent, prescription, active):
        self.label = label
        self.parent = parent
        self.active = active
        self.prescriptions = []
        self.children = []
        if parent:
            parent.prescriptions.append(prescription)
            parent.children.append(self)
        else:
            print(parent)

class TBDag:
    def __init__(self, team = [Player.One, Player.Three], game = Game()):
        self.active_labels : Set[Tuple[str, ...]] = set()
        self.inactive_labels : Set[Tuple[str, ...]] = set()
        self.game = game
        self.team = team
        self.root_node = TBNode(("",), None, None, True)
        self.leaves = []
        self.make_active_node(self.root_node, [self.game.nodes[""]])

    def make_active_node(self, node, B: List[Union[Node, LeafNode]]):
        # for each og node in B, if it's active (in a team infoset)
        # cartesian product of all of these actions
        # for each og node in B, if it's inactive, do all possible children of 
        # need some sort of mapping from prescription to child

        # prescription: infoset to action
        if len(B) == 1 and isinstance(B[0], LeafNode):
            self.leaves.append(node)
            return
        active_nodes = [b for b in B if isinstance(b, Node) and b.player in self.team]
        active_infosets = list(set([(b.infoset, b.player) for b in active_nodes]))
        action_choices = [self.game.infosets[y][x]["actions"] for x, y in active_infosets]
        active_infosets = [x for x, _ in active_infosets]
        
        all_prescriptions = [
            dict(zip(active_infosets, action_choice))
            for action_choice in product(*action_choices)
        ]

        inactive_children = ([child for b in B if isinstance(b, Node) 
                             and b.player not in self.team
                             for child in b.actions.values()] + 
                            [child for b in B if isinstance(b, Node) 
                             and b.player not in self.team
                             for child in b.chance_actions.values()])
        
        for prescription in all_prescriptions:
            active_children = [b.actions[prescription[b.infoset]] for b in active_nodes]
            children = active_children + inactive_children
            child_label : Tuple[str, ...] = tuple(sorted([child.name for child in children]))
            child_node = TBNode(child_label, node, prescription, False)
            self.make_inactive_node(child_node, children)
            self.inactive_labels.add(child_label)
            
        # for each prescription, make a child with that prescription


    def make_inactive_node(self, node, O: List[Union[Node, LeafNode]]):
        def neighboring(item, item2):
            if item[0] == item2[0]:
                return True
            if item[1] == item2[1]:
                return True
        
        def find(x, parent):
            # Path compression optional but nice
            while parent[x] != x:
                x = parent[x]
            return x

        children = []
        opp_components = defaultdict(list)
        team_cc = defaultdict(list)
        # each component is a public history and a pair of sets of cards
        # start with a dictionary
        for o in O:
            if isinstance(o, LeafNode):
                children.append([o])
            elif o.player not in self.team:
                # union find
                c1, c2 = o.name[self.team[0]], o.name[self.team[1]]
                history = o.name[5:]
                # children.append([o])
                opp_components[history].append((c1, c2, o.name))
            else:
                team_cc[o.infoset] += [o]

        opp_child_groups = []

        for history, items in opp_components.items():
            parent = {}
            for item in items:
                neighbors = [item2 for item2 in parent if neighboring(item, item2)]
                if len(neighbors) == 0:
                    parent[item] = item
                else:
                    parent[item] = parent[neighbors[0]]
                for m in neighbors[1:]:
                    parent[m] = parent[neighbors[0]]
            components = defaultdict(list)
            for node2 in parent:
                root = find(node2, parent)
                components[root].append(node2)
            for root in components:
                opp_child_groups.append([self.game.nodes[x] for _, _, x in components[root]])

        children += list(team_cc.values())
        children += opp_child_groups
        for child in children:
            child_label : Tuple[str, ...] = tuple(sorted([c.name for c in child]))
            if child_label not in self.active_labels:
                # print(len(self.active_labels))
                child_node = TBNode(child_label, node, {}, True)
                self.make_active_node(child_node, child)
                self.active_labels.add(child_label)

if __name__ == "__main__":
    tbdag = TBDag()
    print(len(tbdag.active_labels))
    i = 0
    to_visit = [tbdag.root_node]
    while i < 10:
        next = to_visit.pop()
        print(next.label)
        print(f"player: {set(tbdag.game.nodes[l].player for l in next.label)}")
        print(f"active: {next.active}")
        print(f"number of children: {len(next.children)}")
        if next.active:
            print(f"prescription: {next.prescriptions}")
        print("")
        to_visit += next.children
        i+=1

    print(len(tbdag.leaves))