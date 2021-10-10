import math
import random
import sys
from typing import Optional, Tuple, List

from constants import *
from hmm import HMM
from player_controller_hmm import PlayerControllerHMMAbstract

N_HIDDEN_HC = 15
N_HIDDEN_LC = 1
WARMUP_STEPS = 110
N_MODELS = N_SPECIES
NEXT_FISH_POLICY = 'sequential'
NEXT_FISH_POLICY_MAX = 20


class FishHMM:
    """
    FishHMM Class:
    Subclass of HMM to add functionality regarding fish updates and filtering.
    """

    def __init__(self, N_lc: int, N_hc: int, index: int):
        self.index = index
        self.active = False
        self.done = False
        self.n_found = 0
        self.lc_model = HMM(N=N_lc, K=N_EMISSIONS)
        self.hc_model = HMM(N=N_hc, K=N_EMISSIONS)

    def initialize(self, b, l=None):
        self.lc_model.initialize(b, l)
        self.hc_model.initialize(b, l)
        self.active = True

    def train(self, f: 'Fish', max_iter: int = 100, p_tol: float = 1e-6) -> None:
        # Initialize models
        self.initialize(f.beta)
        # Train low-capacity model
        self.lc_model.train(f.obs, max_iters=max_iter, p_tol=p_tol)
        # Train high-capacity model
        self.hc_model.train(f.obs, max_iters=max_iter, p_tol=p_tol)

    def infer(self, f: 'Fish') -> float:
        return max(self.lc_model.alpha_pass_scaled(observations=f.obs)[0],
                   self.hc_model.alpha_pass_scaled(observations=f.obs)[0])


class Fish:
    """
    Fish Class:
    Own implementation of a book-keeping struct to save statistics for each fish.
    """

    UNEXPLORED = 0
    EXPLORED = 1

    def __init__(self, index: int):
        """
        Fish class constructor.
        :param int index: fish index
        """
        self.index = index
        self._obs_seq = [-1] * N_STEPS
        self._type_probs = [-math.inf] * N_SPECIES
        self._t = 0
        self._state = Fish.UNEXPLORED
        self._species = None
        self._beta = [0] * N_EMISSIONS

    @property
    def obs(self) -> list:
        return self._obs_seq[:self._t]

    @obs.setter
    def obs(self, Ot: int) -> None:
        self._obs_seq[self._t] = Ot
        self._beta[Ot] += 1
        self._t += 1

    @property
    def probs(self) -> list:
        return self._type_probs

    def get_most_probable(self, models: List[FishHMM], return_prob: bool = False):
        max_prob, max_mi = -math.inf, None
        for m in models:
            if m.active:
                prob = m.infer(self)
                if prob > max_prob:
                    max_prob = prob
                    max_mi = m.index
        assert max_mi is not None
        if return_prob:
            return max_prob, max_mi
        return max_mi

    @property
    def species(self) -> int:
        return self._species

    @species.setter
    def species(self, si: int) -> None:
        self._species = si
        self._state = Fish.EXPLORED

    @property
    def beta(self) -> list:
        return [1. / self._t for _ in range(N_EMISSIONS)]
        # return [float(b) / self._t + 0.1 * random.random() for b in self._beta]


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
        self.fishes = None
        self.models = None
        self.unexplored_fis = None
        self.active_mis = None
        super().__init__()

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.fishes = [Fish(index=fi) for fi in range(N_FISH)]
        self.models = [FishHMM(N_hc=N_HIDDEN_HC, N_lc=N_HIDDEN_LC, index=mi) for mi in range(N_MODELS)]
        self.unexplored_fis = set(list(range(N_FISH)))
        self.active_mis = set()

    def pick_next_fish(self, policy: str = NEXT_FISH_POLICY) -> Tuple[int, Optional[int]]:
        """
        TODO
        :param policy:
        :return:
        """
        if policy == 'random':
            fo: Fish = self.fishes[random.randint(0, len(self.unexplored_fis) - 1)]
            return fo.index, fo.get_most_probable(self.models)
        if policy == 'sequential':
            fi = list(self.unexplored_fis)[0]
            fo: Fish = self.fishes[fi]
            return fo.index, fo.get_most_probable(self.models)
        if policy == 'max_all':
            # pick a fish from the unexplored ones STRATEGICALLY
            max_fi, max_fi_pred, max_fi_prob, counter = None, None, -math.inf, 0
            for fi in self.unexplored_fis:
                fo: Fish = self.fishes[fi]
                fi_prob, fi_mi = fo.get_most_probable(self.models, return_prob=True)
                if fi_prob > max_fi_prob:
                    max_fi = fi
                    max_fi_prob = fi_prob
                    max_fi_pred = fi_mi
                counter += 1
                if counter >= NEXT_FISH_POLICY_MAX:
                    break
            if max_fi is not None:
                assert not math.isinf(max_fi_pred) and max_fi in self.unexplored_fis
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
        for fi in self.unexplored_fis:
            fo: Fish = self.fishes[fi]
            fo.obs = observations[fi]
        # If we have enough data to start training, start the guessing procedure
        if step == WARMUP_STEPS:
            return random.randint(0, N_FISH - 1), random.randint(0, N_SPECIES - 1)
        # pick a fish from the unexplored ones using the pre-described policy
        elif step > WARMUP_STEPS:
            return self.pick_next_fish(policy=NEXT_FISH_POLICY)

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
        fo: Fish = self.fishes[fish_id]
        model: FishHMM = self.models[true_type]
        if model.done:
            return

        # Update fish
        fo.species = true_type
        self.unexplored_fis.remove(fish_id)
        print(f'[reveal] fi={fish_id} | true_type={true_type} | correct={correct}', file=sys.stderr)

        # Check if model is done
        if model.n_found == (N_FISH / N_SPECIES - 1):
            self.active_mis.remove(true_type)
            model.done = True
            model.active = False
            return

        # Train model using the fish's observation sequence
        if (true_type not in self.active_mis and self.models[true_type].active) or not correct:
            self.active_mis.add(true_type)
            model.n_found += 1
            model.train(f=fo, max_iter=40, p_tol=1e-6)
