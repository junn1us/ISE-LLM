import numpy as np
import gymnasium as gym
from gymnasium import spaces

from smartfixture_llm4teach.core.fixture_core import FixtureCore


class SmartFixtureStudentEnv(gym.Env):
    """
    Student agent - PPO environment with numeric observations
    Shares transition kernel via FixtureCore
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
            "mask": info.mask.copy(),
            "step": info.step,
        }
        return obs, reward, done, info_dict

    def render(self, mode="human"):
        print(
            f"t={self.core.step_count} mask={self.core.mask.astype(int)} "
            f"rmse={self.core.prev_err:.4f}"
        )
