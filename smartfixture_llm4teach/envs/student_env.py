import numpy as np
import gymnasium as gym
from gymnasium import spaces

from smartfixture_llm4teach.core.fixture_core import FixtureCore


class SmartFixtureStudentEnv(gym.Env):
    """
    Student / PPO environment with numeric observations.
    Shares transition kernel via FixtureCore.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, core: FixtureCore):
        super().__init__()
        self.core = core
        obs0 = self.core.reset()
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones_like(obs0),
            high=np.inf * np.ones_like(obs0),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.core.action_catalog.action_dim)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        obs = self.core.reset()
        return obs

    def step(self, action: int):
        obs, reward, done, info = self.core.step(int(action))
        info_dict = {
            "rmse": info.rmse,
            "max_deformation": info.max_deformation,
            "max_deformation_before": info.max_deformation_before,
            "max_deformation_after": info.max_deformation_after,
            "max_deformation_delta": info.max_deformation_delta,
            "mask": info.mask.copy(),
            "step": info.step,
            "feedback_available": info.feedback_available,
            "useful_fixture_added": info.useful_fixture_added,
            "useful_fixture_gain": info.useful_fixture_gain,
            "useful_fixture_bonus": info.useful_fixture_bonus,
            "pre_threshold_progress_bonus": info.pre_threshold_progress_bonus,
            "base_reward": info.base_reward,
            "fixture_usage_penalty": info.fixture_usage_penalty,
            "action_idx": info.action_idx,
            "action_kind": info.action_kind,
            "node_id": info.node_id,
            "applied": info.applied,
            "active_fixtures_before": info.active_fixtures_before,
            "active_fixtures_after": info.active_fixtures_after,
            "gravity_considered": info.gravity_considered,
        }
        return obs, reward, done, info_dict

    def render(self, mode="human"):
        print(
            f"t={self.core.step_count} mask={self.core.mask.astype(int)} "
            f"max_deformation={self.core.prev_max_deformation:.4f}"
        )
