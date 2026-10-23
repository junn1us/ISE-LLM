import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ppo_finetune import ppo_finetune
from validation_runtime import collect_teacher, read_metadata, rollout
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv
from smartfixture_llm4teach.envs.teacher_env import SmartFixtureTeacherEnv


SMALL = dict(anchor_mode="stride", grid_size=9, anchor_stride=4,
             max_steps=3, max_fixtures=3, reward_mode="max_deformation_reduction")


class PipelineTests(unittest.TestCase):
    def test_explicit_load_and_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = ppo_finetune(num_iters=0, core_config=SMALL, init_mode="pretrain",
                teacher_samples=4, distill_steps=1, out_name=f"{tmp}/original.pt")
            student_path = f"{tmp}/original_artifacts/student_policy_init.pt"
            loaded = ppo_finetune(num_iters=0, core_config=SMALL, init_mode="load",
                student_checkpoint=student_path, out_name=f"{tmp}/loaded.pt")
            for key, value in original.pi_net.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.pi_net.state_dict()[key]))
            with self.assertRaises(ValueError):
                ppo_finetune(num_iters=0, core_config={**SMALL, "max_fixtures": 2}, init_mode="load",
                    student_checkpoint=student_path, out_name=f"{tmp}/mismatch.pt")

    def test_strict_mock_teacher_is_not_live_model_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ppo_finetune(num_iters=0, core_config=SMALL, init_mode="pretrain",
                teacher_source="llm", llm_fn=lambda _: '{"best_action_id": 0}',
                teacher_samples=4, distill_steps=1, out_name=f"{tmp}/mock.pt")
            metadata = read_metadata(f"{tmp}/mock.pt")
            self.assertEqual(metadata["teacher"]["callable_origin"], "injected_unverified")
            self.assertEqual(metadata["teacher"]["llm_samples"], 4)
            self.assertEqual(metadata["teacher"]["fallback_samples"], 0)

    def test_pretrain_transfers_weights_and_preserves_critic(self):
        with tempfile.TemporaryDirectory() as tmp:
            common = dict(num_iters=0, core_config=SMALL, seed=16)
            scratch = ppo_finetune(**common, init_mode="scratch", out_name=f"{tmp}/scratch.pt")
            rng_after_scratch = torch.random.get_rng_state()
            trained = ppo_finetune(**common, pretrain_with_teacher=True, force_scratch=True,
                teacher_samples=8, distill_steps=2, out_name=f"{tmp}/teacher.pt")
            self.assertTrue(torch.equal(rng_after_scratch, torch.random.get_rng_state()))
            student = torch.load(f"{tmp}/teacher_artifacts/student_policy_init.pt", weights_only=True)
            for key, value in trained.pi_net.state_dict().items():
                self.assertTrue(torch.equal(value, student["net." + key]))
            self.assertTrue(any(not torch.equal(v, scratch.pi_net.state_dict()[k])
                                for k, v in trained.pi_net.state_dict().items()))
            for key, value in trained.v_net.state_dict().items():
                self.assertTrue(torch.equal(value, scratch.v_net.state_dict()[key]))
            meta = read_metadata(f"{tmp}/teacher.pt")
            self.assertTrue(meta["teacher"]["weight_transfer_verified"])
            records = json.loads(Path(f"{tmp}/teacher_artifacts/teacher_records.json").read_text())
            self.assertEqual([r["timestep"] for r in records], [0, 1, 2, 0, 1, 2, 0, 1])

    def test_collector_resets_and_rejects_invalid_probabilities(self):
        core = FixtureCore(**SMALL)
        env = SmartFixtureStudentEnv(core)
        teacher = SmartFixtureTeacherEnv(core)
        x, p, records = collect_teacher(env, teacher, 20, 1)
        self.assertTrue(np.all(x[:, -3] > 0))
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        self.assertEqual(records[-1]["episode"], 6)
        teacher.get_teacher_distribution = lambda: np.full(env.action_space.n, np.nan)
        with self.assertRaises(ValueError):
            collect_teacher(env, teacher, 1, 1)

    def test_strict_teacher_and_explicit_fallback(self):
        core = FixtureCore(**SMALL)
        core.reset()
        bad = lambda _: "not json"
        with self.assertRaises(RuntimeError):
            SmartFixtureTeacherEnv(core, llm_fn=bad, strict=True).get_teacher_distribution()
        teacher = SmartFixtureTeacherEnv(core, llm_fn=bad, strict=True, allow_fallback=True)
        teacher.get_teacher_distribution()
        self.assertEqual(teacher.last_query["source"], "heuristic")
        self.assertTrue(teacher.last_query["fallback"])

    def test_terminal_and_negative_actions_rejected(self):
        core = FixtureCore(**SMALL)
        core.reset()
        with self.assertRaises(ValueError):
            core.step(-1)
        for _ in range(core.max_steps):
            core.step(core.action_catalog.index("noop"))
        with self.assertRaises(RuntimeError):
            core.step(0)

    def test_checkpoint_tamper_and_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scratch.pt"
            ppo_finetune(num_iters=0, core_config=SMALL, out_name=str(path))
            meta = read_metadata(path)
            meta["act_dim"] = 1
            Path(str(path) + ".json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                read_metadata(path)
            with self.assertRaises(ValueError):
                read_metadata(Path(tmp) / "legacy.pt")

    def test_one_iteration_and_raw_noop_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            ac = ppo_finetune(num_iters=1, episodes_per_iter=1, epochs=1,
                core_config=SMALL, out_name=f"{tmp}/smoke.pt")
            with torch.no_grad():
                for param in ac.pi_net.parameters():
                    param.zero_()
                ac.pi_net[-1].bias[-1] = 10
            result = rollout(ac, SMALL)
            self.assertEqual(result["fixtures"], 0)
            self.assertEqual(result["actions"], [18] * 3)


if __name__ == "__main__":
    unittest.main()
