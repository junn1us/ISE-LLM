import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from ppo_finetune import ppo_finetune
from run_validation_study import run_study
from report_validation_study import report
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from validation_instances import make_suite, validate_suite


class InstanceTests(unittest.TestCase):
    def test_split_reproducibility_and_leakage_rejection(self):
        suite = make_suite(train=2, validation=1, test=1, ood=1)
        self.assertEqual(suite, make_suite(train=2, validation=1, test=1, ood=1))
        duplicate = copy.deepcopy(suite["train"][0])
        duplicate["instance_id"] = "renamed-leak"
        suite["test"][0] = duplicate
        with self.assertRaises(ValueError):
            validate_suite(suite)

    def test_varied_shapes_and_fixed_observation_schema(self):
        suite = make_suite(train=3, validation=1, test=1, ood=1)
        cores = [FixtureCore(**{**suite["base_environment"], **item}) for item in suite["train"]]
        self.assertEqual(len({core.reset().shape for core in cores}), 1)
        self.assertFalse(np.allclose(cores[0]._free_sag / max(cores[0]._free_sag),
                                    cores[1]._free_sag / max(cores[1]._free_sag)))
        self.assertFalse(np.array_equal(cores[0].get_student_features(), cores[1].get_student_features()))

    def test_teacher_collection_uses_only_training_instances(self):
        suite = make_suite(train=2, validation=1, test=1, ood=1, anchor_mode="stride")
        suite["base_environment"]["max_steps"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            ppo_finetune(num_iters=0, init_mode="pretrain", teacher_samples=7, distill_steps=1,
                core_config=suite["base_environment"], training_instances=suite["train"], out_name=f"{tmp}/p.pt")
            records = json.loads(Path(f"{tmp}/p_artifacts/teacher_records.json").read_text())
            self.assertEqual({r["instance_id"] for r in records}, {i["instance_id"] for i in suite["train"]})

    def test_study_smoke_no_network(self):
        suite = make_suite(train=2, validation=1, test=1, ood=1, anchor_mode="stride")
        suite["base_environment"].update(max_steps=3, max_fixtures=3)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_study(Path(tmp) / "study", suite, seeds=(16,), iters=1,
                episodes_per_iter=2, teacher_samples=7, distill_steps=1, baseline_calls=15)
            self.assertEqual(result["manifest"]["scientific_status"], "smoke_only")
            self.assertEqual(result["manifest"]["llm_evidence"], "not_run")
            self.assertTrue(all(r["terminal_optimality_gap"] >= -1e-6 for r in result["records"]))
            self.assertEqual({r["phase"] for r in result["records"]}, {"initial", "final"})
            rows = report(Path(tmp) / "study")
            self.assertTrue(rows)
            self.assertTrue((Path(tmp) / "study" / "validation_learning_curve.png").exists())


if __name__ == "__main__":
    unittest.main()
