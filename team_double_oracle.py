from itertools import permutations
import numpy as np
from typing import *
from dataclasses import dataclass
from game import *
import zipfile

class TeamDoubleOracle:
    def __init__(self, game, tolerance = 1.):
        self.populations: Dict[str, List[Strategy]] = {}
        self.tolerance: float = tolerance
        self.utilities: Dict[Tuple[int, int], float] = {}
        self.game = game

        constraints_13 = game.construct_constraints((Player.One, Player.Three))
        constraints_24 = game.construct_constraints((Player.Two, Player.Four))

        self.n_13 = sum([len(c) for c in constraints_13[0]])
        self.n_24 = sum([len(c) for c in constraints_24[0]])
        self.l_13 = constraints_13[2]
        self.l_24 = constraints_24[2]

        for team in ("13", "24"):
            c = constraints_13 if team == "13" else constraints_24
            (x1, x2, y), (x_ind, y_ind), leaf_to_y_ind, (constraints, row_bounds, col_bounds) = c
            h = highspy.Highs()
            num_vars = len(x1) + len(x2) + len(y)

            h.addCols(
                num_vars,
                np.array([0] * num_vars, dtype=np.float64),
                np.array([0] * num_vars, dtype=np.float64),
                np.array([1] * num_vars, dtype=np.float64),
                0,  
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )
            
            for row, (lb, ub) in enumerate(row_bounds):
                constraint = constraints[row]
                h.addRow(lb, ub, len(constraint), [c[0] for c in constraint], [c[1] for c in constraint])

            num_x = len(x1) + len(x2)
            indices = np.arange(num_x, dtype=np.int32)
            types = np.full(num_x, highspy.HighsVarType.kInteger, dtype=np.uint32)
            h.changeColsIntegrality(num_x, indices, types)
            h.setOptionValue("presolve", "on")
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)

            if team == "13":
                self.h_br_13 = h
            else:
                self.h_br_24 = h

    def compute_utility(self, strat13 : Strategy, strat24 : Strategy) -> float:
        payoff13 = np.sum(strat13.seq_form * strat24.all_contribs)
        return payoff13

    def iterate(self, trial: int) -> Tuple[bool, Strategy, Strategy]:
        strat_13, strat_24 = self.get_nash_strategies() # this is the nash equilibrium step
        br_24 = strat_13.best_response(self.h_br_24, self.n_24, self.l_24)
        br_13 = strat_24.best_response(self.h_br_13, self.n_13, self.l_13)
        self.populations["13"].append(br_13)
        self.populations["24"].append(br_24)

        self.update_utilities(trial)
        
        eq_utility = self.compute_utility(strat_13, strat_24)
        br_utility_13 = self.compute_utility(br_13, strat_24)
        br_utility_24 = self.compute_utility(strat_13, br_24)

        converged = abs(eq_utility - br_utility_13) < self.tolerance and abs(eq_utility - br_utility_24) < self.tolerance
        return converged, br_13, br_24
    
    def update_utilities(self, trial: int) -> None:
        for i in range(trial):
            self.utilities[(i, trial)] = self.compute_utility(self.populations["13"][i], self.populations["24"][trial])
            self.utilities[(trial, i)] = self.compute_utility(self.populations["13"][trial], self.populations["24"][i])
        self.utilities[(trial, trial)] = self.compute_utility(self.populations["13"][trial], self.populations["24"][trial])
        
    def get_nash_strategies(self) -> Tuple[Strategy, Strategy]:
        # TODO: get the actual Nash mixture
        mixtures = [("13", []), ("24", [])] # say this is two lists of probs

        nash_strategies = []
        for team, mix in mixtures:
            team_strategies = []
            team_seq_form = np.zeros(len(self.game.leaves))
            for i, prob in enumerate(mix):
                pure_strat = self.populations[team][i]
                team_strategies += [(pure_strat.strategies[0][0], prob)]
                team_seq_form += pure_strat.seq_form * prob
            team_all_contribs = team_seq_form * (self.game.chance_payoffs_24 if team == "13" else self.game.chance_payoffs_13)
            nash_strategies += [Strategy(self.game, team, team_strategies, team_seq_form, team_all_contribs)]
        return tuple(nash_strategies)

    def team_strategy_to_tensors(self, strategy: Strategy, team : str):
        with zipfile.ZipFile(f"team{team}.zip", "w") as zf:
            with zf.open("meta-strategy.csv", "w") as f1:
                for i, (pure_strat, prob) in enumerate(strategy.strategies):
                    f1.write(f"{i},{prob}\n".encode("utf-8"))
                    for p in team:
                        player = player_to_Player("P" + p)
                        player_infosets = self.game.infosets[player]
                        tensor = np.zeros((len(player_infosets), 3), dtype=np.float64)
                        for infoset, infoset_info in player_infosets.items():
                            infoset_line = infoset_info["line"]
                            for action, action_prob in pure_strat[player][infoset].items():
                                tensor[infoset_line, action] = action_prob
                        with zf.open(f"strategy{i}-player{p}.npy", "w") as f:
                            np.save(f, tensor)

    def train(self, num_rand = 3, trials = 1000) -> Tuple[Strategy, Strategy]:
        for team in ["13", "24"]:
            self.populations[team] = [Strategy(self.game, team) for _ in range(num_rand)]
            for strat in self.populations[team]:
                strat.random_strategy()
        self.utilities = {(i, j) : self.compute_utility(self.populations["13"][i], self.populations["24"][j]) 
                          for i in range(num_rand) for j in range(num_rand)}
        
        strat_13, strat_24 = self.populations["13"][0], self.populations["24"][0]
        
        for trial in range(trials):
            converged, strat_13, strat_24 = self.iterate(trial + num_rand)
            if converged:
                print(f"Converged at trial: {trial}")
                return strat_13, strat_24
            if (trial % 100) == 0:
                print(f"Progress: {(100 * trial) / trials}%")
            
        return strat_13, strat_24
            

if __name__ == "__main__":
    strategy = Strategy(Game(), "13")
    strategy.uniform_strategy()