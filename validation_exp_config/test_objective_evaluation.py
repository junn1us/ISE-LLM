from pathlib import Path
import tempfile
import unittest
import json
import numpy as np

from ppo_finetune import ppo_finetune
from eval_policy import evaluate_checkpoint
from eval_ppo_ablation import compare_checkpoints
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.teacher.translator import build_teacher_context


class ObjectiveTests(unittest.TestCase):
    def test_paired_evaluator_rejects_unmatched_training_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"anchor_mode": "stride", "grid_size": 9, "max_steps": 2}
            scratch, teacher = f"{tmp}/scratch.pt", f"{tmp}/teacher.pt"
            ppo_finetune(num_iters=0, core_config=config, out_name=scratch)
            ppo_finetune(num_iters=0, core_config=config, out_name=teacher,
                init_mode="pretrain", teacher_samples=2, distill_steps=1)
            self.assertEqual(len(compare_checkpoints(scratch, teacher)["paired_deltas"]), 1)
            sidecar = Path(teacher + ".json")
            metadata = json.loads(sidecar.read_text())
            metadata["ppo_transitions"] = 100
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                compare_checkpoints(scratch, teacher)

    def core(self, **kwargs):
        return FixtureCore(anchor_mode="stride", grid_size=9, anchor_stride=4,
            max_steps=5, max_fixtures=3, reward_mode="max_deformation_reduction", **kwargs)

    def test_deformation_nonincreasing_and_rewards_telescope(self):
        core = self.core()
        core.reset()
        initial = core.sim_max_deformation()
        rewards = []
        previous = initial
        for node in (4, 0, 8):
            _, reward, _, _ = core.step(core.action_catalog.index("add", node))
            self.assertLessEqual(core.sim_max_deformation(), previous)
            previous = core.sim_max_deformation()
            rewards.append(reward)
        self.assertAlmostEqual(sum(rewards), (initial - previous) / initial)
        _, reward, _, _ = core.step(core.action_catalog.index("remove", 0))
        self.assertLessEqual(reward, 0)
        _, reward, _, _ = core.step(core.action_catalog.index("noop"))
        self.assertAlmostEqual(reward, 0)

    def test_gate_reports_infeasibility_not_zero_deformation_success(self):
        core = self.core(min_feedback_fixtures=3)
        core.reset()
        for node in (0, 4):
            _, reward, _, info = core.step(node)
            self.assertEqual(reward, 0)
            self.assertTrue(np.isnan(info.max_deformation))
        _, reward, _, info = core.step(8)
        self.assertTrue(info.feedback_available)
        self.assertAlmostEqual(reward, 1 - core.sim_max_deformation() / core.free_max_deformation)

    def test_context_states_objective(self):
        core = self.core()
        core.reset()
        context = build_teacher_context(core, core.candidate_actions())
        self.assertIn("maximum", context["objective"])
        self.assertEqual(context["max_fixtures"], 3)

    def test_canonical_evaluation_reproduces_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "policy.pt")
            ppo_finetune(num_iters=0, out_name=path, core_config={
                "anchor_mode": "twelve", "grid_size": 9, "max_steps": 3})
            a = evaluate_checkpoint(path)
            b = evaluate_checkpoint(path)
            self.assertEqual(a, b)
            with self.assertRaises(ValueError):
                evaluate_checkpoint(path, [{"reward_mode": "max_deformation_reduction"}])


if __name__ == "__main__":
    unittest.main()
