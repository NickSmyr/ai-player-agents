import unittest

from hmm import HMM


class TestHMM(unittest.TestCase):
    def test_alpha_beta_gamma_passes(self):
        _hmm = HMM(N=4, K=3)
        _observations = [0, 1, 2, 1, 1, 1, 2, 0, 0]
        # Likelihood
        _ll_a, _ = _hmm.alpha_pass(_observations)
        _ll_b, _ = _hmm.beta_pass(_observations)
        # Approximate equality
        assert abs(_ll_a - _ll_b) < 1e-3, "Likelihoods don't match"

        _gammas, _digammas = _hmm.gamma_pass(_observations)
        _hmm.get_new_parameters(_observations, _gammas, _digammas)


if __name__ == '__main__':
    unittest.main()
