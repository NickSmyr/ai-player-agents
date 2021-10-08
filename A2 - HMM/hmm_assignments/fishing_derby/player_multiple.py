import math
import random
from typing import Optional, Tuple

from constants import *
from hmm import HMM
from hmm_utils import argmax
from player_controller_hmm import PlayerControllerHMMAbstract

N_HIDDEN_MULTIPLIER = 15
N_MODELS_PER_SPECIES = 2
N_FISH_PER_SPECIES = N_FISH // N_SPECIES  # perfectly-balanced classes
WARMUP_STEPS = N_STEPS - N_FISH
N_MODELS = N_SPECIES
P_THRESHOLD = 1 / N_SPECIES


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
        self.unexplored_fishes = None
        self.active_species, self.active_species_list, self.done_species = None, None, None
        self.n_fishes_per_species = None
        self.species_models = None
        self.obs_seq = None
        self.t = None
        self.obs_counts = None
        self.step = None
        super().__init__()

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.species_models = [
            [HMM(N=1 + N_HIDDEN_MULTIPLIER * smi, K=N_EMISSIONS) for smi in range(N_MODELS_PER_SPECIES)]
            for _ in range(N_SPECIES)]
        self.unexplored_fishes = list(range(N_FISH))
        self.active_species, self.active_species_list, self.done_species = set(), list(), set()
        self.obs_seq = [[] for _ in range(N_FISH)]
        self.obs_counts = [0.] * N_EMISSIONS
        self.step = 0
        self.n_fishes_per_species = {species_index: 0 for species_index in range(N_SPECIES)}

    def pick_next_fish(self, policy: str = 'random', max_n_fish: int = 50) -> Tuple[int, Optional[int]]:
        if policy == 'random':
            return self.unexplored_fishes.pop(random.randint(0, len(self.unexplored_fishes) - 1)), None
        if policy == 'sequential':
            return self.unexplored_fishes.pop(), None
        if policy == 'max_all':
            # pick a fish from the unexplored ones STRATEGICALLY
            max_fii, max_fi_pred, max_mi = None, None, None
            max_fi_max_prob = -math.inf
            random.shuffle(self.unexplored_fishes)
            for fii in range(min(max_n_fish, len(self.unexplored_fishes))):
                fi = self.unexplored_fishes[fii]
                fi_probs = [max([self.species_models[s][smi].alpha_pass(self.obs_seq[fi])
                                 for smi in range(N_MODELS_PER_SPECIES)]) for s in self.active_species_list]
                max_prob, max_mi = argmax(fi_probs)
                if max_prob > 0.0 and max_prob > max_fi_max_prob:
                    max_fii = fii
                    max_fi_max_prob = max_prob
                    max_fi_pred = self.active_species_list[max_mi]
                if fii >= max_n_fish:
                    break
            if max_fii is not None:
                max_fi = self.unexplored_fishes.pop(max_fii)
                return max_fi, max_fi_pred
            return self.pick_next_fish('random')

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
            ot_fi = observations[fi]
            self.obs_seq[fi].append(ot_fi)
            self.obs_counts[ot_fi] += 1
        # if step in [80, 90, 100, 110, 120, 130, 140, 150, 170]:
        #     for mi in self.active_models:
        #         model = self.models[mi]
        #         # model.initialize(globalB=self.obs_counts)
        #         model.baum_welch(self.obs_seq[model.last_fish_id], max_iter=10, tol=1e-6, update_params=True)
        #     print('ALL MODELS RETRAINED!')
        #     print('  ALL MODELS RETRAINED!')
        #     print('    ALL MODELS RETRAINED!')
        self.step = step
        print(f'step={step}')
        if step == 170:
            print(f'self.globalB={[f"{bk:.3f}" for bk in self.obs_counts]}')
            # sys.exit(-1)
        # assert all(ot == self_ot for ot, self_ot in zip(observations, Matrix2d(self.obs_seq).get_col(self.t - 1)))
        # If we have enough data to start training, start the guessing procedure
        if step >= WARMUP_STEPS:
            if not self.active_species:
                return self.pick_next_fish('random')[0], random.randint(0, N_SPECIES - 1)
            #   - pick a fish from the unexplored ones
            fi, fi_pred = self.pick_next_fish(policy='max_all')
            if fi_pred is not None:
                return fi, fi_pred
            #   - pass its observation sequence through all the models and select the most active one
            fi_probs = [max([self.species_models[s][smi].alpha_pass(self.obs_seq[fi])
                             for smi in range(N_MODELS_PER_SPECIES)]) for s in self.active_species_list]
            # print(f'fi_probs={fi_probs}', file=stderr)
            return fi, self.active_species_list[argmax(fi_probs)[1]]

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
        # If all fishes of a species were found, remove respective models
        self.n_fishes_per_species[true_type] += 1
        if self.n_fishes_per_species == N_FISH_PER_SPECIES:
            self.active_species.remove(true_type)
            self.active_species_list = list(self.active_species)
            self.done_species.add(true_type)
            del self.species_models

        # Retrain model only if it made a wrong prediction
        if true_type not in self.active_species and true_type not in self.done_species:
            self.active_species.add(true_type)
            self.active_species_list = list(self.active_species)
            # Train all models of the returned species
            for model in self.species_models[true_type]:
                model.initialize(globalB=self.obs_counts, label=true_type)
                model.train(self.obs_seq[fish_id], max_iter=30, tol=1e-6)
                model.last_fish_id = fish_id
