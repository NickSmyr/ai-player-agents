import math
import random

from constants import *
from hmm import HMM
from hmm_utils import argmax
from player_controller_hmm import PlayerControllerHMMAbstract

N_HIDDEN = 1
WARMUP_STEPS = 75


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
        self.models = None
        self.unexplored_fishes = None
        self.obs_seq = None
        self.obs_count = None
        self.t = None
        self.last_model_index = None
        super().__init__()

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.models = []
        self.obs_seq = [[] for _ in range(N_FISH)]
        self.obs_count = [[0] * N_EMISSIONS for _ in range(N_FISH)]
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
            self.obs_seq[fi].append(fi_ot)
            self.obs_count[fi][fi_ot] += 1
        self.t = step

        # If we have enough data to start training, start the guessing procedure
        if step >= WARMUP_STEPS and len(self.unexplored_fishes) > 0:
            if self.models:
                max_fii, max_fi_pred, max_mi = None, None, None
                max_fi_max_prob = -math.inf
                #   - pick a fish from the unexplored ones STRATEGICALLY
                for fii, fi in enumerate(self.unexplored_fishes):
                    fi_probs = [model.alpha_pass(self.obs_seq[fi]) for model in self.models]
                    max_prob, max_mi = argmax(fi_probs)
                    if max_prob > 0.0 and max_prob > max_fi_max_prob:
                        max_fii = fii
                        max_fi_max_prob = max_prob
                        max_fi_pred = self.models[max_mi].label
                    if fii >= 20:
                        break
                if max_fii is not None:
                    max_fi = self.unexplored_fishes.pop(max_fii)
                    self.last_model_index = max_mi
                    return max_fi, max_fi_pred
            return self.unexplored_fishes.pop(random.randint(0, len(self.unexplored_fishes) - 1)), \
                   random.randint(0, N_SPECIES - 1)
        return None

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
        fish_obs_seq = self.obs_seq[fish_id]
        # If correction was wrong, train a new model to identify similar fishes
        if not correct:
            new_model = HMM(N=N_HIDDEN, K=N_EMISSIONS)
            new_model.initialize(self.obs_count[fish_id], label=true_type)
            new_model.train(fish_obs_seq, max_iter=30, p_tol=1e-6)
            self.models.append(new_model)
        # Prediction was correct: do nothing
        #   -> after warmup steps, retrain model every 10 steps
        elif (self.t - WARMUP_STEPS) % 10 == 0:
            # model.initialize(self.obs_count[fish_id], len(fish_obs_seq))
            # self.models[self.last_model_index].baum_welch(fish_obs_seq, max_iter=20, tol=1e-6, update_params=True)
            pass
