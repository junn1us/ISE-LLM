# env_fixture.py
import numpy as np
import matplotlib.pyplot as plt


class DiscreteFixtureEnv:
    """
    Environment for discrete fixture placement qwq

    Features:
    9x9 sheet (coords 0..8 in x and y)
    Candidate anchor locations are configurable:
        anchor_mode="boundary": all boundary nodes
        anchor_mode="stride": grid with step=anchor_stride (includes center if stride divides 4)
        anchor_mode="twelve": 12 strategic locations (8 boundary corners/edges + 4 interior)
        anchor_mode="all": all 81 nodes
      Default is stride=4 -> anchors at {0,4,8}x{0,4,8} (9 anchors, includes center).
    
    Action space: toggle an anchor ON/OFF (plus an optional noop)
    
    State: sag at each anchor + global stats (max/mean/std)
    
    Reward: improvement in RMSE toward target bowl
    """

    def __init__(self, seed=13, anchor_mode="stride", anchor_stride=4, include_noop=True):
        self.rng = np.random.default_rng(seed)

        # Sheet grid (9x9)
        self.NX = 9
        self.NY = 9
        xs = np.arange(self.NX)
        ys = np.arange(self.NY)
        GX, GY = np.meshgrid(xs, ys)
        self.coords = np.stack([GX.reshape(-1), GY.reshape(-1)], axis=1)  # (81,2)

        # ----------------------------
        # Anchor (fixture) candidate sites
        # ----------------------------
        all_nodes = [(x, y) for x in range(self.NX) for y in range(self.NY)]

        if anchor_mode == "boundary":
            anchors = [(x, y) for (x, y) in all_nodes
                       if (x == 0 or x == self.NX - 1 or y == 0 or y == self.NY - 1)]
        elif anchor_mode == "stride":
            s = int(anchor_stride)
            if s <= 0:
                raise ValueError("anchor_stride must be >= 1")
            anchors = [(x, y) for x in range(0, self.NX, s)
                             for y in range(0, self.NY, s)]
        elif anchor_mode == "twelve":
            # 12 strategic locations: 8 boundary + 4 interior
            # Boundary: 4 corners + 4 midpoints
            # Interior: 4 points around center (lattice interior)
            anchors = [
                # Corners (4)
                (0, 0), (8, 0), (0, 8), (8, 8),
                # Edge midpoints (4)
                (4, 0), (4, 8), (0, 4), (8, 4),
                # Interior lattice points (4)
                (2, 2), (6, 2), (2, 6), (6, 6),
            ]
        elif anchor_mode == "all":
            anchors = all_nodes
        else:
            raise ValueError(f"Unknown anchor_mode={anchor_mode}. Use 'boundary', 'stride', 'twelve', or 'all'.")

        self.anchor_xy = np.array(anchors, dtype=np.float32)
        self.num_anchors = int(self.anchor_xy.shape[0])

        # action ids:
        # if include_noop:
        #   0 = noop
        #   1..num_anchors = toggle anchor k-1
        # else:
        #   0..num_anchors-1 = toggle anchor k
        self.include_noop = bool(include_noop)
        self.action_dim = (1 + self.num_anchors) if self.include_noop else self.num_anchors

        # deflection state for all 81 nodes
        self.disp = np.zeros((self.coords.shape[0], 2), dtype=np.float32)

        # target sag profile: shallow bowl
        self.target_disp = self._make_target_bowl()

        # anchor mask: 1 if fixture ON, 0 if OFF
        self.mask = np.zeros(self.num_anchors, dtype=np.float32)

        # tracking error for reward shaping
        self.prev_err = None
        self.step_count = 0
        self.max_steps = 20  # cap per-episode

    def _make_target_bowl(self):
        """
        Desired sag shape: shallow symmetric bowl.
        """
        center = np.array([4.0, 4.0])
        diffs = self.coords - center[None, :]
        r2 = np.sum(diffs**2, axis=1)

        sag = 1.5 * np.exp(-r2 / (2 * (6.0)))
        bowl = np.zeros((self.coords.shape[0], 2), dtype=np.float32)
        bowl[:, 1] = sag.astype(np.float32)
        return bowl

    # ----------------------------
    # Core simulation (surrogate)
    # ----------------------------

    def sim_apply(self):
        """
        Update self.disp given current fixture mask.
        Stand-in for compliance solve:
          - Start from free sag under gravity
          - Each ON fixture locally pulls sag toward 0
        """
        center = np.array([4.0, 4.0])
        diffs = self.coords - center[None, :]
        r2 = np.sum(diffs**2, axis=1)
        free_sag = 2.0 * np.exp(-r2 / (2 * (6.0)))  # deeper than target

        disp = np.zeros_like(self.disp)
        disp[:, 1] = free_sag

        for k, (ax, ay) in enumerate(self.anchor_xy):
            if self.mask[k] > 0.5:
                diffa = self.coords - np.array([ax, ay])
                r2a = np.sum(diffa**2, axis=1)
                influence = np.exp(-r2a / (2 * (2.0**2)))  # radius ~2
                disp[:, 1] *= (1.0 - 0.8 * influence)

        self.disp = disp.astype(np.float32)

    def sim_error(self):
        """
        RMSE btn current sag and target sag.
        """
        cur = self.disp[:, 1]
        tgt = self.target_disp[:, 1]
        mse = np.mean((cur - tgt) ** 2)
        return float(np.sqrt(mse))

    def _extract_deform_features(self):
        """
        Features = sag at each anchor + global stats (max/mean/std)
        Total = num_anchors + 3
        
        This function is designed to work with ANY number of anchors (9, 12, etc.)
        To freeze the observation schema and avoid repeated incompatibility issues,
        we use a fixed-size feature vector based on num_anchors at environment creation
        """
        feats = []

        for (ax, ay) in self.anchor_xy:
            d2 = np.sum((self.coords - np.array([ax, ay])) ** 2, axis=1)
            idx = int(np.argmin(d2))
            feats.append(self.disp[idx, 1])

        sag_all = self.disp[:, 1]
        feats.append(float(np.max(sag_all)))
        feats.append(float(np.mean(sag_all)))
        feats.append(float(np.std(sag_all)))

        return np.array(feats, dtype=np.float32)

    # ----------------------------
    # Gym-like API
    # ----------------------------

    def reset(self):
        self.mask[:] = 0.0
        self.step_count = 0

        self.sim_apply()
        err = self.sim_error()
        self.prev_err = err

        obs = self._extract_deform_features()
        return obs

    def step(self, action_id):
        """
        If include_noop:
          action_id=0 => noop
          action_id in [1..num_anchors] => toggle (action_id-1)
        Else:
          action_id in [0..num_anchors-1] => toggle action_id
        """
        self.step_count += 1

        if self.include_noop:
            if action_id > 0:
                k = int(action_id) - 1
                if 0 <= k < self.num_anchors:
                    self.mask[k] = 1.0 - self.mask[k]
        else:
            k = int(action_id)
            if 0 <= k < self.num_anchors:
                self.mask[k] = 1.0 - self.mask[k]

        self.sim_apply()
        err_now = self.sim_error()

        denom = max(self.prev_err, 1e-6)
        reward = 1.0 - (err_now / denom)
        reward = float(np.clip(reward, -1.0, 1.0))

        self.prev_err = err_now
        obs = self._extract_deform_features()
        done = (self.step_count >= self.max_steps)

        info = {
            "rmse": err_now,
            "mask": self.mask.copy(),
        }
        return obs, reward, done, info

    def render(self, title="Discrete fixtures"):
        """
        Plot sheet grid. Filled marker = fixture ON, open = OFF.
        """
        plt.figure(figsize=(6, 6))

        # draw coarse grid
        for x in range(self.NX):
            plt.plot([x, x], [0, self.NY - 1], color="tab:green", alpha=0.3)
        for y in range(self.NY):
            plt.plot([0, self.NX - 1], [y, y], color="tab:orange", alpha=0.3)

        # plot anchors
        for k, (ax, ay) in enumerate(self.anchor_xy):
            if self.mask[k] > 0.5:
                plt.scatter(ax, ay, c="k", s=100, marker="o")
                plt.scatter(ax, ay, c="k", s=100, marker="^")
            else:
                plt.scatter(ax, ay, facecolors="white", edgecolors="black", s=100)

        plt.xlim(-1, self.NX)
        plt.ylim(-1, self.NY)
        plt.gca().set_aspect("equal", "box")
        plt.title(title)
        plt.grid(False)
        plt.show()
