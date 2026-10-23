import unittest
from math import comb
import numpy as np
from fixture_baselines import search_layouts, return_oracle
from itertools import product
from smartfixture_llm4teach.core.fixture_core import FixtureCore


class BaselineTests(unittest.TestCase):
    def test_exact_nine_and_twelve_counts(self):
        for mode, n in (("stride", 9), ("twelve", 12)):
            result = search_layouts(dict(anchor_mode=mode, grid_size=9, max_steps=20))
            self.assertEqual(result["simulator_calls"], 2 ** n)
            self.assertTrue(result["proven_optimum"])
            self.assertEqual(len(result["best_by_fixture_count"]), n + 1)

    def test_budget_horizon_and_gate(self):
        result = search_layouts(dict(anchor_mode="stride", grid_size=9,
            max_fixtures=4, max_steps=3, min_feedback_fixtures=2))
        self.assertEqual(result["simulator_calls"], comb(9, 2) + comb(9, 3))
        self.assertTrue(2 <= result["best"]["fixtures"] <= 3)

    def test_optimum_replays_in_actual_core(self):
        config = dict(anchor_mode="stride", grid_size=9, max_fixtures=3,
                      reward_mode="max_deformation_reduction")
        result = search_layouts(config)
        core = FixtureCore(**config)
        core.reset()
        for node in result["best"]["nodes"]:
            core.step(node)
        self.assertAlmostEqual(core.sim_max_deformation(), result["best"]["objective"])

    def test_search_bounds_and_reproducibility(self):
        config = dict(anchor_mode="stride", grid_size=9, max_fixtures=3)
        optimum = search_layouts(config)["best"]["objective"]
        for method in ("random", "greedy", "local", "beam", "bayes"):
            a = search_layouts(config, method=method, calls=60, seed=16)
            b = search_layouts(config, method=method, calls=60, seed=16)
            self.assertEqual(a["best"], b["best"])
            self.assertLessEqual(a["simulator_calls"], 60)
            self.assertGreaterEqual(a["best"]["objective"] + 1e-8, optimum)
            self.assertFalse(a["proven_optimum"])

    def test_grid100_exact_is_guarded(self):
        with self.assertRaises(ValueError):
            search_layouts(dict(max_fixtures=5))

    def test_return_oracle_matches_exhaustive_action_sequences(self):
        config = dict(anchor_mode="stride", grid_size=3, anchor_stride=2,
            max_steps=3, max_fixtures=2, reward_mode="normalized_rmse")
        oracle = return_oracle(config, gamma=0.9)
        core = FixtureCore(**config)
        best = -float("inf")
        for actions in product(range(core.action_catalog.action_dim), repeat=3):
            core.reset()
            total = 0
            for t, action in enumerate(actions):
                _, reward, _, _ = core.step(action)
                total += 0.9 ** t * reward
            best = max(best, total)
        self.assertAlmostEqual(oracle["discounted_return"], best)
        core.reset()
        replay = [core.step(action)[1] for action in oracle["actions"]]
        np.testing.assert_allclose(replay, oracle["rewards"])


if __name__ == "__main__":
    unittest.main()
