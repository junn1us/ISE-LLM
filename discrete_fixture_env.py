# env_fixture.py
# environment implementation (core reproducibility)

import numpy as np
import gym
from gym import spaces


class DiscreteFixtureEnv(gym.Env):
    """
    the almighty discrete fixture placement environment

    16 candidate fixture locations (e.g., boundary nodes on a sheet).
    Binary mask m in {0,1}^16 indicates whether each location has a fixture.
    at most max_fixtures (default 8) may be active at once.
    - Action space: Discrete(17)
        0      : noop
        1..16  : toggle fixture at index k-1 (add if 0 and below max; remove if 1).

    Reward:
        r_t = max( 1 - err_t / err_{t-1}, 0.0 )

      where err_t is the current RMSE (or any scalar error) after applying the
      new mask, and err_{t-1} is the previous error. This ensures non-negative
      rewards, i.e., we only pay for *improvements*.

    The environment also tracks:
        - reward_traj : list of per-step rewards
        - cum_reward  : scalar cumulative reward over the episode

    Must implement:
        - self.sim_apply(mask)  -> updates self.deform, self.err
        - self.sim_error()      -> returns current scalar error
        - self._extract_deform_features() -> np.ndarray of deformation features
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_anchors: int = 16,
        max_fixtures: int = 8,
        max_steps: int = 20,
        seed: int = 0,
    ):
        super().__init__()
        assert max_fixtures <= num_anchors, "max_fixtures must be <= num_anchors"

        self.num_anchors = int(num_anchors)
        self.max_fixtures = int(max_fixtures)
        self.max_steps = int(max_steps)

        self.rng = np.random.default_rng(seed)

        # Binary mask of fixtures: 1 = fixture present, 0 = free
        self.mask = np.zeros(self.num_anchors, dtype=np.int8)

        # Action: 0 = noop, 1..num_anchors = toggle index-1
        self.action_space = spaces.Discrete(1 + self.num_anchors)

        # Observation will be a fixed-size vector: deformation features + fixture count
        # Here we reserve 16 floats for deformation summary + 1 scalar (fixture count)
        self._feat_dim = 16
        obs_low = -np.inf * np.ones(self._feat_dim + 1, dtype=np.float32)
        obs_high = np.inf * np.ones(self._feat_dim + 1, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Error tracking
        self.err_prev = None
        self.err_curr = None
        self.best_err = None

        # Reward tracking
        self.reward_traj = []
        self.cum_reward = 0.0
        self.t = 0  # step counter

        # Placeholder for the deformation field, if you store it
        self.deform = None

    # ------------------------------------------------------------------
    # Core simulation hooks
    # ------------------------------------------------------------------
    def sim_apply(self, mask: np.ndarray):
        """
        Apply the fixture mask to the surrogate / FE model and update
        the internal deformation and error.

        NOTE: You MUST replace this stub with your own linear compliance + gravity
        calculation (or call out to your existing surrogate).

        Example pattern:
            u = self.compliance.solve(mask, gravity=True)
            self.deform = u            # shape (N_nodes, 2) or similar
            self.err_curr = self.sim_error()
        """
        # --- Stub: simple fake "error" that prefers ~7-8 fixtures -------------
        num = mask.sum()
        # pretend error is lowest when num_fixtures ~= 7.5
        self.deform = None
        self.err_curr = float((num - 7.5) ** 2 + 0.1)
        # ----------------------------------------------------------------------

    def sim_error(self) -> float:
        """
        In a real implementation:
            err = sqrt(mean((u - u_target)^2))
        """
        if self.err_curr is None:
            return 0.0
        return float(self.err_curr)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_deform_features(self) -> np.ndarray:
        """
        Summarize the current deformation into ~16 floats for the observation.

        In a real implementation you might:
          - partition the interior into quadrants and take mean/ max displacement,
          - include d_max, interface error, etc.

        For now we just encode:
          [err_curr, best_err, num_fixtures, step_norm] + zeros to length 16.
        """
        num = float(self.mask.sum())
        err = float(self.err_curr if self.err_curr is not None else 0.0)
        best = float(self.best_err if self.best_err is not None else err)
        step_norm = float(self.t) / max(self.max_steps - 1, 1)

        feats = np.zeros(self._feat_dim, dtype=np.float32)
        feats[0] = err
        feats[1] = best
        feats[2] = num
        feats[3] = step_norm
        return feats

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # start with no fixtures
        self.mask[:] = 0
        self.t = 0
        self.reward_traj = []
        self.cum_reward = 0.0

        # apply once to initialize error
        self.sim_apply(self.mask)
        self.err_curr = self.sim_error()
        self.err_prev = self.err_curr
        self.best_err = self.err_curr

        obs = self._obs()
        return obs

    def _obs(self) -> np.ndarray:
        feats = self._extract_deform_features()
        num = float(self.mask.sum())
        obs = np.concatenate([feats, np.array([num], dtype=np.float32)], axis=0)
        return obs.astype(np.float32)

    def step(self, action: int):
        assert self.action_space.contains(action)

        # --- update mask based on action ---
        if action > 0:
            idx = action - 1
            if self.mask[idx] == 1:
                # remove fixture
                self.mask[idx] = 0
            else:
                # add fixture only if below max
                if self.mask.sum() < self.max_fixtures:
                    self.mask[idx] = 1
                # else: treat as noop (cannot exceed max fixtures)

        # --- simulate with new mask ---
        self.err_prev = self.err_curr
        self.sim_apply(self.mask)
        self.err_curr = self.sim_error()
        self.best_err = min(self.best_err, self.err_curr)

        # --- reward: max(1 - err_t / err_{t-1}, 0) ---
        denom = max(self.err_prev, 1e-8)
        raw = 1.0 - self.err_curr / denom
        reward = max(raw, 0.0)

        self.cum_reward += reward
        self.reward_traj.append(reward)
        self.t += 1

        done = self.t >= self.max_steps
        info = {
            "err_prev": self.err_prev,
            "err_curr": self.err_curr,
            "best_err": self.best_err,
            "cum_reward": self.cum_reward,
            "num_fixtures": int(self.mask.sum()),
        }

        return self._obs(), float(reward), done, info

    # ------------------------------------------------------------------
    # Helpers for logging
    # ------------------------------------------------------------------
    def get_reward_trajectory(self) -> np.ndarray:
        """Return np.array of per-step rewards for the current episode."""
        return np.array(self.reward_traj, dtype=np.float32)

    def get_cumulative_reward(self) -> float:
        """Return scalar cumulative reward for the current episode."""
        return float(self.cum_reward)

    # ------------------------------------------------------------------
    # Simple text render
    # ------------------------------------------------------------------
    def render(self, mode="human"):
        print(f"t={self.t} mask={self.mask} err={self.err_curr:.4f} "
              f"cumR={self.cum_reward:.3f}")
