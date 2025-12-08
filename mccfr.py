import re
import numpy as np
import zipfile
from collections import defaultdict
import time

TRIALS = 1000000
REPORT_PER = max(1, TRIALS // 100)
SAVE_PER = 100000
EPSILON = .25
GAMMA = .999

class Node:
    def __init__(self, path):
        self.path = path
        self.type = None
        self.player = None
        self.actions = None
        self.payoffs = None
        self.chance_outcomes = None

def parse_tree_file():
    nodes = {}
    history_to_infoset = {}
    with open("leduc_tree.txt", "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("infoset"):
                parts = line.split()
                if len(parts) < 3:
                    continue
                key = parts[1]
                if "nodes" in parts:
                    idx = parts.index("nodes")
                    for h in parts[idx+1:]:
                        history_to_infoset[h] = key
                continue
            m = re.match(r"node\s+([^ ]+)\s+(.*)$", line)
            if not m:
                continue
            path, rest = m.group(1), m.group(2)
            node = Node(path)
            if rest.startswith("chance actions"):
                node.type = "chance"
                toks = rest[14:].strip().split()
                outcomes = {}
                for tok in toks:
                    if "=" in tok:
                        o, p = tok.split("=", 1)
                        try:
                            outcomes[o] = float(p)
                        except:
                            pass
                node.chance_outcomes = outcomes
                nodes[path] = node
                continue
            m2 = re.search(r"player\s+([1-4])\s+actions\s+(.*)$", rest)
            if m2:
                node.type = "decision"
                node.player = int(m2.group(1))
                node.actions = m2.group(2).strip().split()
                nodes[path] = node
                continue
            m3 = re.search(r"leaf payoffs\s+(.*)$", rest)
            if m3:
                node.type = "terminal"
                pay_map = {}
                for tok in m3.group(1).strip().split():
                    if "=" in tok:
                        pl, v = tok.split("=", 1)
                        try:
                            pay_map[int(pl)] = float(v)
                        except:
                            pass
                node.payoffs = pay_map
                nodes[path] = node
    return nodes, history_to_infoset

def load_infoset_action_files():
    infoset_actions = {}
    infoset_to_player = {}
    for p in range(1, 5):
        with open(f"player_{p}_infosets.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    infoset_key = parts[1]
                    full_key = (p, infoset_key)
                    infoset_actions[full_key] = list(parts[2])
                    infoset_to_player[full_key] = p

    return infoset_actions, infoset_to_player

class MCCFRTrainer:
    def __init__(self, nodes, history_to_infoset, infoset_actions, infoset_to_player, epsilon=EPSILON):
        self.start_time = time.time()
        self.nodes = nodes
        self.history_to_infoset = history_to_infoset
        self.infoset_actions = infoset_actions
        self.infoset_to_player = infoset_to_player
        self.epsilon = epsilon
        self.regret = {0: defaultdict(lambda: defaultdict(float)),
                       1: defaultdict(lambda: defaultdict(float))}
        self.strategy_sum = {0: defaultdict(lambda: defaultdict(float)),
                             1: defaultdict(lambda: defaultdict(float))}
        self.iteration = 0
        self.player_to_team = {1: 0, 2: 1, 3: 0, 4: 1}
        self.team_players = {0: [1, 3], 1: [2, 4]}
        self.root = min(nodes.keys(), key=len)
        self.terminal_values = {}
        for path, node in nodes.items():
            if node.type == "terminal":
                team_utils = {0: 0.0, 1: 0.0}
                for p, v in node.payoffs.items():
                    team_utils[self.player_to_team[p]] += v
                self.terminal_values[path] = team_utils[0] - team_utils[1]
        self.chance_data = {}
        for path, node in nodes.items():
            if node.type == "chance":
                outcomes = list(node.chance_outcomes.keys())
                probs = np.array([node.chance_outcomes[o] for o in outcomes], dtype=np.float64)
                s = probs.sum()
                if s > 0:
                    probs /= s
                else:
                    probs = np.ones(len(probs)) / len(probs)
                self.chance_data[path] = (outcomes, probs)

    def get_strategy(self, key, acts, team):
        pos = [max(0.0, self.regret[team][key].get(a, 0.0)) for a in acts]
        s = sum(pos)
        if s > 0:
            return {a: pos[i]/s for i, a in enumerate(acts)}
        else:
            return {a: 1.0/len(acts) for a in acts}
    
    def get_sampling_strategy(self, key, acts, team):
        base = self.get_strategy(key, acts, team)
        epsilon = max(EPSILON / 5, EPSILON * (1.0 - self.iteration / float(TRIALS)))
        uniform_mass = epsilon / len(acts)
        return {a: (1 - epsilon) * base[a] + uniform_mass for a in acts}

    def avg_strategy(self, I, team, player):
        key = (player, I)
        acts = self.infoset_actions.get(key)
        if acts is None:
            return {}
        s = sum(self.strategy_sum[team][key].get(a, 0.0) for a in acts)
        if s > 0:
            return {a: self.strategy_sum[team][key].get(a, 0.0)/s for a in acts}
        return {a: 1.0/len(acts) for a in acts}

    def external_sampling_cfr(self, history, reach_team0, reach_team1, traversing_team):
        node = self.nodes.get(history)
        if node is None:
            return 0.0
        if node.type == "terminal":
            util = self.terminal_values[history]
            if traversing_team == 0:
                return util
            else:
                return -util
        if node.type == "chance":
            outcomes, probs = self.chance_data[history]
            o = np.random.choice(outcomes, p=probs)
            child = history.rstrip("/") + "/C:" + o
            return self.external_sampling_cfr(child, reach_team0, reach_team1, traversing_team)
        player = node.player
        player_team = self.player_to_team[player]
        I = self.history_to_infoset.get(history)
        if I is None:
            return 0.0
        key = (player, I)
        acts = self.infoset_actions.get(key) or node.actions
        sigma = self.get_strategy(key, acts, player_team)
        sigma_sample = self.get_sampling_strategy(key, acts, player_team)
        base = history.rstrip("/") + f"/P{player}:"
        
        if player_team == traversing_team:
            action_utils = {}
            for a in acts:
                child = base + a
                new_reach_team0 = reach_team0.copy()
                new_reach_team1 = reach_team1.copy()
                if player_team == 0:
                    new_reach_team0[player] *= sigma_sample[a]
                else:
                    new_reach_team1[player] *= sigma_sample[a]
                action_utils[a] = self.external_sampling_cfr(child, new_reach_team0, new_reach_team1, traversing_team)
            
            node_util = sum(sigma_sample[a] * action_utils[a] for a in acts)
            
            opp_reach = 1.0
            for p in self.team_players[1 - traversing_team]:
                opp_reach *= (reach_team1[p] if traversing_team == 0 else reach_team0[p])
            
            for a in acts:
                regret = action_utils[a] - node_util
                self.regret[player_team][key][a] = (self.regret[player_team][key][a] * GAMMA) + (opp_reach * regret)
            
            teammate_reach = 1.0
            for p in self.team_players[traversing_team]:
                if p != player:
                    teammate_reach *= (reach_team0[p] if traversing_team == 0 else reach_team1[p])
            
            for a in acts:
                self.strategy_sum[player_team][key][a] += teammate_reach * sigma[a]
            
            return node_util
        else:
            probs = np.array([sigma_sample[a] for a in acts], dtype=np.float64)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                probs = np.ones(len(acts)) / len(acts)
            else:
                probs /= probs_sum
            chosen = np.random.choice(acts, p=probs)
            child = base + chosen
            
            new_reach_team0 = reach_team0.copy()
            new_reach_team1 = reach_team1.copy()
            if player_team == 0:
                new_reach_team0[player] *= sigma_sample[chosen]
            else:
                new_reach_team1[player] *= sigma_sample[chosen]
            
            return self.external_sampling_cfr(child, new_reach_team0, new_reach_team1, traversing_team)

    def train(self, iterations):
        for t in range(1, iterations + 1):
            self.iteration = t
            for team in [0, 1]:
                reach_team0 = {1: 1.0, 3: 1.0}
                reach_team1 = {2: 1.0, 4: 1.0}
                self.external_sampling_cfr(self.root, reach_team0, reach_team1, team)
            if t % REPORT_PER == 0:
                minutes_from_start = (time.time() - self.start_time) / 60
                minutes_projected_left = ((float(iterations - t) / t) * minutes_from_start)
                print(f"Progress: {t}/{iterations} iterations, {minutes_from_start:.2f} minutes elapsed, {minutes_projected_left:.2f} minutes remaining")
            if t % SAVE_PER == 0:
                avg_team0 = {}
                avg_team1 = {}
                for key in self.infoset_actions:
                    player, I = key
                    team = self.player_to_team[player]
                    if team == 0:
                        avg_team0[key] = self.avg_strategy(I, 0, player)
                    else:
                        avg_team1[key] = self.avg_strategy(I, 1, player)
                save_team_strategies("13", avg_team0)
                save_team_strategies("24", avg_team1)
                print("Strategies for Team 13:")
                count = 0
                for key, v in avg_team0.items():
                    if key[0] == 1:
                        print(key[1], {a: round(p, 4) for a, p in v.items()})
                        count += 1
                        if count >= 10:
                            break
                print("Strategies for Team 24:")
                count = 0
                for key, v in avg_team1.items():
                    if key[0] == 2:
                        print(key[1], {a: round(p, 4) for a, p in v.items()})
                        count += 1
                        if count >= 10:
                            break


def save_team_strategies(team_str, avg):
    with zipfile.ZipFile(f"team{team_str}.zip", "w") as z:
        with z.open("meta-strategy.csv", "w") as f:
            f.write(b"0,1.0\n")
        for player_char in team_str:
            player = int(player_char)
            fname = f"player_{player}_infosets.txt"
            with open(fname, "r") as fin:
                lines = fin.readlines()
            T = np.zeros((len(lines), 3), dtype=np.float64)
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                I = parts[1]
                acts = parts[2]
                key = (player, I)
                strat = avg.get(key, {})
                aC = strat.get("C", 0.0) if "C" in acts else 0.0
                aF = strat.get("F", 0.0) if "F" in acts else 0.0
                aR = strat.get("R", 0.0) if "R" in acts else 0.0
                s = aC + aF + aR
                if s > 0:
                    aC /= s
                    aF /= s
                    aR /= s
                else:
                    n = len(acts)
                    aC = 1.0/n if "C" in acts else 0.0
                    aF = 1.0/n if "F" in acts else 0.0
                    aR = 1.0/n if "R" in acts else 0.0
                T[i, 0] = aC
                T[i, 1] = aF
                T[i, 2] = aR
            with z.open(f"strategy0-player{player}.npy", "w") as f:
                np.save(f, T)

if __name__ == "__main__":
    nodes, history_to_infoset = parse_tree_file()
    infoset_actions, infoset_to_player = load_infoset_action_files()
    trainer = MCCFRTrainer(nodes, history_to_infoset, infoset_actions, infoset_to_player, EPSILON)
    trainer.train(iterations=TRIALS)