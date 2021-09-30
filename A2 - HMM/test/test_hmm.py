import unittest

from hmm import HMM


class TestHMM(unittest.TestCase):
    def setUp(self) -> None:
        self.hmm = HMM(N=4, K=3)
        self.observations = [0, 1, 2, 1, 1, 1, 2, 0, 0]

    def test_alpha_beta_passes(self):
        # Likelihood of observations
        ll_a, _, cs = self.hmm.alpha_pass(self.observations)
        ll_b, _ = self.hmm.beta_pass(self.observations, cs=cs)
        # Approximate equality
        self.assertAlmostEqual(ll_a, ll_b, places=6, msg="Likelihoods don't match")

    def test_gamma_pass(self):
        gammas, digammas = self.hmm.gamma_pass(self.observations)
        self.hmm.reestimate(self.observations, gammas, digammas)
        new_pi, new_A, new_B = self.hmm.pi, self.hmm.A, self.hmm.B,
        # Check new A matrix (shape, row stochastic)
        self.assertListEqual(list(new_A.shape), list(self.hmm.A.shape))
        for r in range(new_A.nrows):
            self.assertAlmostEqual(sum(new_A[r]), 1.0, places=14)
        # Check new B matrix (shape, row stochastic)
        self.assertListEqual(list(new_B.shape), list(self.hmm.B.shape))
        for r in range(new_B.nrows):
            self.assertAlmostEqual(sum(new_B[r]), 1.0, places=14)
        # Check new pi vector (shape, row stochastic)
        self.assertListEqual(list(new_pi.shape), list(self.hmm.pi.shape))
        self.assertAlmostEqual(new_pi.sum(), 1.0, places=14)

    def test_delta_pass(self):
        states_path, max_prob = self.hmm.delta_pass(self.observations)
        self.assertTrue(isinstance(states_path, list), msg=f'Invalid type for state_path returned '
                                                           f'(returned type={type(states_path)})')
        self.assertEqual(len(self.observations), len(states_path), msg='Number of states in returned states_path '
                                                                       'is wrong')

    def tearDown(self) -> None:
        del self.hmm.A
        del self.hmm.B
        del self.hmm.pi
