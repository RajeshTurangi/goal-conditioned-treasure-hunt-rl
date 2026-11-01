import unittest

import numpy as np

from goal_treasure.env import TreasureHuntEnv


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = TreasureHuntEnv(["#####", "#...#", "#...#", "#####"], max_steps=10)

    def test_reset_is_reproducible(self):
        first = self.env.reset(seed=7)
        second = self.env.reset(seed=7)
        self.assertEqual(first, second)

    def test_goal_terminates(self):
        self.env.reset(start=(1, 1), goal=(1, 2))
        result = self.env.step(3)
        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, 10.0)

    def test_vectors_have_expected_shapes(self):
        self.env.reset(seed=1)
        self.assertEqual(self.env.state_vector().shape, (self.env.state_size,))
        self.assertEqual(self.env.goal_vector().shape, (self.env.goal_size,))
        self.assertEqual(self.env.state_vector().dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

