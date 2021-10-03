import math
from sys import stderr

from hmm_assignments.fishing_derby.hmm import HMM
from hmm_utils import Matrix2d, Vector

AGGREGATE_OBSERVATIONS = False
MAX_OBS_SEQ_LEN = 100


class FishHMM(HMM):
    """
    FishHMM Class:
    Our implementation of a 1-st order Hidden Markov Model to PROGRESSIVELY predict fish species.
    """

    def __init__(self, N: int, K: int, model_index: int, initialization: str = 'random'):
        pi, A, B = None, None, None
        if 'cross' == initialization:
            # pi should be random
            pi = Vector.random(N, normalize=False)
            # A should say that most probably when you are in a state of going in one direction you will either
            # continue towards that direction or change and head to the opposite. So A, is I_N + antiI_N + rand.
            A = Matrix2d.random(N, N, row_stochastic=False)
            if N >= 4:
                A[0][0] += 1  # continue to the top
                A[0][3] += 1  # change to the bottom
                A[1][1] += 1  # continue to the left
                A[1][2] += 1  # change to the right
                A[2][2] += 1  # continue to the right
                A[2][1] += 1  # change to the left
                A[3][3] += 1  # continue to the bottom
                A[3][0] += 1  # change to the top
            A.normalize_rows()
            # Since we assumed that states are directions, at each state you will most likely observe moving at that
            # direction
            B = Matrix2d.random(N, K, row_stochastic=False)
            """
            -------------------------------
            FISH MOVES (POSSIBLE EMISSIONS)
            -------------------------------
              [4]   [0]  [5]
              [2]    F   [3]
              [6]   [1]  [7]
            -------------------------------
            """
            if N >= 4:
                # State 0: Going upwards
                B[0][0] += 1
                B[0][4] += 0.4
                B[0][5] += 0.4
                # State 3: Going downwards
                B[3][1] += 1
                B[3][6] += 0.4
                B[3][7] += 0.4
                # State 1: Going to the left
                B[1][2] += 1
                B[1][4] += 0.4
                B[1][6] += 0.4
                # State 2: Going to the right
                B[2][3] += 1
                B[2][5] += 0.4
                B[2][7] += 0.4
            B.normalize_rows()
        elif 'random' == initialization:
            pass
        super(FishHMM, self).__init__(N=N, K=K, pi=pi, A=A, B=B)
        self.oldA = Matrix2d(self.A.data.copy())
        self.oldB = Matrix2d(self.B.data.copy())
        self.oldPi = Vector(self.pi.data.copy())

        self.model_index = model_index
        self.done_fish_indices = []
        self.done = False

    @staticmethod
    def join_lists(l1: list, l2: list) -> list:
        """
        Joins two list by first aligning them on their closest common element.
        :param list l1: first list
        :param list l2: second list
        :return: the concatenated list
        """
        l = min(len(l1), max(l2))

        def get_crop_indices(_l1, _l2, _l, force: bool = False):
            for radius in range(1, (_l // 2 + 1) if force is False else _l):
                for r1 in range(-1, -radius - 1, -1):
                    if -r1 == radius:
                        for r2 in range(0, -r1):
                            if _l1[r1] == _l2[r2]:
                                return r1 + 1, r2
                    else:
                        r2 = radius - 1
                        if _l1[r1] == _l2[r2]:
                            return r1 + 1, r2
            return None, None

        # Find index
        #  - try l1 first
        which_first = 0
        crop_from_or_until_l1, crop_from_or_until_l2 = get_crop_indices(l1, l2, l)
        if crop_from_or_until_l1 is None:
            crop_from_or_until_l2, crop_from_or_until_l1 = get_crop_indices(l2, l1, l, force=True)
            if crop_from_or_until_l2 is None:
                # FAILED: No common element to join
                return l1 if len(l1) > len(l2) else l2
            which_first = 1

        # Join lists
        l1_cropped = l1.copy()[crop_from_or_until_l1 + 1:] if crop_from_or_until_l1 >= 0 else \
            l1.copy()[:crop_from_or_until_l1]
        l2_cropped = l2.copy()[crop_from_or_until_l2 + 1:] if crop_from_or_until_l2 >= 0 else \
            l2.copy()[:crop_from_or_until_l2]
        return l1_cropped + l2_cropped if which_first == 0 else l2_cropped + l1_cropped

    def train(self, entire_obs_seq: list, fish_indices: list, step: int):
        """
        Train model on the sequences of the given fishes.
        :param list entire_obs_seq: the entire data matrix of shape (N_FISH, N_STEPS)
        :param list fish_indices: indices of the fish whose type is this model's index
        :param int step: current time step
        :return:
        """
        # Train using Baum-Welch
        try:
            # Get aggregated + aligned observation sequence
            if AGGREGATE_OBSERVATIONS:
                aggreg_obs_seq = entire_obs_seq[fish_indices[0]][:step]
                for fi in range(1, len(fish_indices)):
                    aggreg_obs_seq = FishHMM.join_lists(
                        aggreg_obs_seq,
                        entire_obs_seq[fish_indices[fi]][:step]
                    )
                super().baum_welch(observations=aggreg_obs_seq[:MAX_OBS_SEQ_LEN], tol=1e-5, max_iter=40, T=step,
                                   update_params=True, lambda_mix=0.5)
            # Else, loop through them all
            else:
                aggr_gammas = [Vector([0.] * self.N)] * step
                aggr_digammas = [Matrix2d([[0.] * self.N] * self.N)] * (step - 1)
                lambda_mix = 1.0
                aggreg_obs_seq = None
                for fi in range(len(fish_indices)):
                    aggreg_obs_seq = entire_obs_seq[fish_indices[fi]][:step]
                    _, _, _, _, gammas, digammas = super().baum_welch(observations=aggreg_obs_seq, tol=1e-6,
                                                                      max_iter=30, T=step, update_params=True,
                                                                      lambda_mix=1.0 if fi == 0 else 0.5)
                    # for t in range(step - 1):
                    #     aggr_gammas[t] += gammas[t] * (1. / step)
                    #     aggr_digammas[t] += digammas[t] * (1. / step)
                    # aggr_gammas[step - 1] += gammas[step - 1] * (1. / step)
                # Reestimate from aggregated gammas & digammas
                # self.reestimate(observations=aggreg_obs_seq, gammas=aggr_gammas, digammas=aggr_digammas, T=step,
                #                 lambda_mix=0.9)
        except RuntimeError or ZeroDivisionError as e:
            print(f'[HMM {self.model_index}]!{type(e)}! {str(e)}', file=stderr)
            # raise e

    def infer(self, observations: list) -> float:
        """
        Predict likelihood of given observation sequence.
        :param list observations: sequence of observations
        :return: a float
        """
        try:
            ll, _ = super().beta_pass_scaled(observations)
            return ll
        except RuntimeError or ArithmeticError as e:
            print(f'[HMM {self.model_index}][infer] ArithmeticError: {str(e)}', file=stderr)
            return -math.inf

    def reset(self):
        self.done = False
        self.done_fish_indices.clear()
        # self.A = Matrix2d(self.oldA.data.copy())
        # self.B = Matrix2d(self.oldB.data.copy())
        # self.pi = Vector(self.oldPi.data.copy())
