"""Canonical shared-core validation; optional artifacts must pass, not silently skip."""
import argparse
from pathlib import Path
import unittest
import numpy as np
import torch

from distill_student import StudentPolicyDiscrete
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv
from smartfixture_llm4teach.envs.teacher_env import SmartFixtureTeacherEnv
from validation_runtime import read_metadata


def validate_twelve(artifact_dir=None):
    core = FixtureCore(anchor_mode="twelve", grid_size=9)
    env = SmartFixtureStudentEnv(core)
    expected = np.asarray([(0, 0), (8, 0), (0, 8), (8, 8), (4, 0), (4, 8),
                           (0, 4), (8, 4), (2, 2), (6, 2), (2, 6), (6, 6)]) * 0.25
    np.testing.assert_allclose(core.anchor_xy, expected)
    assert env.reset().shape == (30,)
    assert env.action_space.n == 25
    teacher = SmartFixtureTeacherEnv(core)
    probabilities = teacher.get_teacher_distribution()
    assert probabilities.shape == (25,) and np.isclose(probabilities.sum(), 1)
    assert teacher.last_query["source"] == "heuristic"
    if artifact_dir:
        root = Path(artifact_dir)
        checkpoint = root / "student_policy_init.pt"
        meta = read_metadata(checkpoint)
        assert meta["environment"]["anchor_mode"] == "twelve"
        assert meta["model_type"] == "student" and meta["act_dim"] == 25
        x = np.load(root / "teacher_data_X.npy")
        p = np.load(root / "teacher_data_P.npy")
        assert x.ndim == 2 and x.shape[1] == meta["obs_dim"]
        assert p.shape == (len(x), 25) and len(x) > 0
        assert np.isfinite(x).all() and np.isfinite(p).all() and np.all(p >= 0)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
        student = StudentPolicyDiscrete(meta["obs_dim"], meta["act_dim"])
        student.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    return {"anchors": 12, "base_observation_dim": 30, "actions": 25,
            "artifact_checks": "passed" if artifact_dir else "not_requested"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir")
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    print(validate_twelve(args.artifact_dir))
    if args.tests:
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
