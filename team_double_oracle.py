from itertools import permutations
import numpy as np
from typing import *
from dataclasses import dataclass
from game import *

class TeamDoubleOracle:
    def __init__(self, tolerance = 1., game_file="game.pkl"):
        self.populations: Dict[str, List[Strategy]] = {}
        self.tolerance: float = tolerance
        self.utilities: Dict[Tuple[int, int], float] = {}
        if game_file is None:
            self.game = Game()
        else:
            with open(game_file, 'rb') as f:
                self.game = pickle.load(f)

    def compute_utility(self, strat13 : Strategy, strat24 : Strategy) -> float:
        payoff13 = np.sum(strat13.seq_form * strat24.all_contribs)
        return payoff13

    def iterate(self, trial: int) -> Tuple[bool, Strategy, Strategy]:
        strat_13, strat_24 = self.get_nash_strategies() # this is the nash equilibrium step
        br_24, br_13 = strat_13.best_response(), strat_24.best_response()
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
                team_seq_form += pure_strat.seq_form * prob # this should be an np.add ?
            team_all_contribs = team_seq_form * (self.game.chance_payoffs_24 if team == "13" else self.game.chance_payoffs_13)
            nash_strategies += [Strategy(team, "game.pkl", team_strategies, team_seq_form, team_all_contribs)]
        return tuple(nash_strategies)

    # TODO will probably have to store the infoset ordering during Game construction
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

    def train(self, num_rand = 3, trials = 1000):
        for team in ["13", "24"]:
            self.populations[team] = [Strategy(team) for _ in range(num_rand)]
            for strat in self.populations[team]:
                strat.random_strategy()
        self.utilities = {(i, j) : self.compute_utility(self.populations["13"][i], self.populations["24"][j]) 
                          for i in range(num_rand) for j in range(num_rand)}
        
        # TODO: idk this is weird
        for trial in range(trials):
            converged, team_mixture, opponent_mixture = self.iterate(trial + num_rand)
            if converged:
                print(f"Converged at trial: {trial}")
                return team_mixture, opponent_mixture
            if (trial % 100) == 0:
                print(f"Progress: {(100 * trial) / trials}%")
            
        return team_mixture, opponent_mixture
            

if __name__ == "__main__":
    strategy = Strategy("13")
    strategy.uniform_strategy()