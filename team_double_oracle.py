from itertools import permutations
import numpy as np
from typing import *
from dataclasses import dataclass
from game import *

@dataclass(frozen=True)
class PureStrategy:
    strategy: Dict[str, Action]

class TeamDoubleOracle:

    def __init__(self, team: Tuple[Player], tolerance = 1.):
        self.populations: Dict[Player, List[PureStrategy]] = {}
        self.team: Tuple[Player]= team
        self.tolerance: float = tolerance
        self.utilities: Dict[Tuple[int, int, int, int], float] = {}

    def iterate(self, trial: int) -> Tuple[bool, List[float], List[float]]:
        team_mixture, opponent_mixture = self.get_mixtures()
        team_best_response, opponent_best_response = opponent_mixture.best_response(), team_mixture.best_response()
        
        best_responses = [None, None, None, None]
        if self.team == (Player.One, Player.Three):
            best_responses[0], best_responses[2] = team_best_response
            best_responses[1], best_responses[3] = opponent_best_response
        else:
            best_responses[0], best_responses[2] = opponent_best_response
            best_responses[1], best_responses[3] = team_best_response

        for player, best_response in zip(players, best_responses):
            self.populations[player].append(best_response)
        self.update_utilities(trial)
        
        eq_utility = self.utility(team_mixture, opponent_mixture)
        team_br_utility = self.utility(team_best_response, opponent_mixture)
        opponent_br_utility = self.utility(team_mixture, opponent_best_response)

        converged = abs(eq_utility - team_br_utility) < self.tolerance and abs(team_br_utility - opponent_br_utility) < self.tolerance
        return converged, team_mixture, opponent_mixture
    
    def update_utilities(self, trial: int):

        def pure_strategies_average_utility(p1_strat, p2_strat, p3_strat, p4_strat):
            # for each leaf node, the odds of being at this node are the product of the 
            # sequence form probs of all players and the chance probs
            # strategy representation 
            pass

        def update_utilities_for_player(player: Player):
            other_players = players[:]
            other_players.remove(player)

            player_new_strat = self.populations[player][-1]
            for i, i_strat in enumerate(self.populations[other_players[0]]):
                for j, j_strat in enumerate(self.populations[other_players[1]]):
                    for k, k_strat in enumerate(self.populations[other_players[2]]):
                        strats = [None, None, None, None]
                        strats[other_players[0]] = i_strat
                        strats[other_players[1]] = j_strat
                        strats[other_players[2]] = k_strat
                        strats[player] = player_new_strat

                        indices = [0,0,0,0]
                        strats[other_players[0]] = i
                        strats[other_players[1]] = j
                        strats[other_players[2]] = k
                        strats[player] = trial

                        self.utilities[tuple(indices)] = pure_strategies_average_utility(*strats)
        
        for player in players:
            update_utilities_for_player(player)
        return
        
    def get_mixtures(self):
        pass

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

    def train(self, trials = 1000):
        self.utilities = [[0] * trials] * trials

        team_mixture, opponent_mixture = None, None
        for trial in range(trials):
            converged, team_mixture, opponent_mixture = self.iterate(trial)
            if converged:
                print(f"Converged at trial: {trial}")
                return team_mixture, opponent_mixture
            if trial % 100:
                print(f"Progress: {(100 * trial) / trials}%")
            
        return team_mixture, opponent_mixture
            

if __name__ == "__main__":
    game = Game()
    print(len(game.nodes), len(game.infosets))
    strategy = Strategy((Player.One, Player.Three), game)
    strategy.uniform_strategy()
    print(len(strategy.strategy), len(strategy.strategy[Player.One]))