import unittest

import numpy as np

from goal_treasure.exact import follow_policy, value_iteration


class ExactSolverTests(unittest.TestCase):
    def test_value_iteration_reaches_goal(self):
        obstacles = np.array(
            [[True, True, True, True], [True, False, False, True], [True, False, False, True], [True, True, True, True]],
            dtype=bool,
        )
        result = value_iteration(obstacles, tolerance=1e-9)
        path = follow_policy(result.policy, (1, 1), (2, 2), obstacles)
        self.assertEqual(path[-1], (2, 2))
        self.assertLessEqual(len(path) - 1, 4)
        self.assertLess(result.delta, 1e-9)


if __name__ == "__main__":
    unittest.main()

