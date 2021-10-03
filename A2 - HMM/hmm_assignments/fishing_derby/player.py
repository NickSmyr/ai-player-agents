from constants import *
from hmm_assignments.fishing_derby.hmm import HMM
from hmm_utils import argmax
from player_controller_hmm import PlayerControllerHMMAbstract

N_HIDDEN = 1
WARMUP_STEPS = 75
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
        self.models = None
        self.unexplored_fishes = None
        self.active_models = None
        self.obs_seq = None
        self.t = None
        super().__init__()

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.models = [HMM(N=N_HIDDEN, K=N_EMISSIONS) for _ in range(N_MODELS)]
        self.unexplored_fishes = list(range(N_FISH))
        self.active_models = set()
        self.obs_seq = [[] for _ in range(N_FISH)]

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
            self.obs_seq[fi].append(observations[fi])
        # assert all(ot == self_ot for ot, self_ot in zip(observations, Matrix2d(self.obs_seq).get_col(self.t - 1)))
        # If we have enough data to start training, start the guessing procedure
        if step >= WARMUP_STEPS:
            #   - pick a fish from the unexplored ones randomly
            # fi = self.unexplored_fishes.pop(random.randint(0, len(self.unexplored_fishes) - 1))
            fi = self.unexplored_fishes.pop()
            #   - pass its observation sequence through all the models and select the most active one
            fi_probs = [model.alpha_pass_scaled(self.obs_seq[fi]) for model in self.models]
            # print(f'fi_probs={fi_probs}', file=stderr)
            return fi, argmax(fi_probs)[1]

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
        # Retrain model only if it made a wrong prediction
        if true_type not in self.active_models or not correct:
            self.active_models.add(true_type)
            self.models[true_type].baum_welch(self.obs_seq[fish_id], max_iter=30, tol=1e-8, update_params=True)
