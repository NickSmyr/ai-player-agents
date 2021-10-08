import math
import random

from constants import *
from player_controller_hmm import PlayerControllerHMMAbstract

N_HIDDEN = 1
WARMUP_STEPS = 1


class FishAsPoint:
    def __init__(self, obs_count: list, species: int, fi: int):
        self.coords = obs_count
        self.species = species
        self.fi = fi

    def __sub__(self, other_point: 'FishAsPoint'):
        """
        Get vector distance.
        :param FishAsPoint other_point:
        """
        # Minkowski distance
        return math.pow(sum(math.pow(abs(ci - cj), 0.25) for ci, cj in zip(self.coords, other_point.coords)), 4)
        # Euclidean distance
        # return math.sqrt(sum((ci - cj) ** 2 for ci, cj in zip(self.coords, other_point.coords)))


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    """
    PlayerControllerHMM Class:
    Our controller using 1 HMM / fish type, updating its parameters from given observations.
    -------------------------------
    FISH MOVES (POSSIBLE EMISSIONS)
    -------------------------------
      [4]   [0]  [5]
      [2]    F   [3]
      [6]   [1]  [7]
    -------------------------------
    """

    def __init__(self):

        self.fish_points = [FishAsPoint(obs_count=[0] * N_EMISSIONS, species=-1, fi=fi) for fi in range(N_FISH)]
        self.explored_fishes = set()

        self.unexplored_fishes = None
        self.t = None
        super().__init__()

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.unexplored_fishes = list(range(N_FISH))

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        # Store observations
        for fi in range(N_FISH):
            fi_ot = observations[fi]
            # E_n = (n-1)/n * E_{n-1} + x_n/n, for n>=1
            self.fish_points[fi].coords[fi_ot] += 1

        if len(self.explored_fishes) == 0:
            return self.unexplored_fishes.pop(), random.randint(0, N_SPECIES - 1)

        # If we have enough data to start training, start the guessing procedure
        if step > N_STEPS - N_FISH and len(self.unexplored_fishes) > 0:
            min_dist, min_dist_fpi, fii = math.inf, None, None
            for fii, fi in enumerate(self.unexplored_fishes):
                # Compute point-wise distance
                for fj in self.explored_fishes:
                    dist_ij = self.fish_points[fj] - self.fish_points[fi]
                    if dist_ij < min_dist:
                        min_dist = dist_ij
                        min_dist_fpi = fj
            return self.unexplored_fishes.pop(fii), self.fish_points[min_dist_fpi].species

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        self.explored_fishes.add(fish_id)
        self.fish_points[fish_id].species = true_type
