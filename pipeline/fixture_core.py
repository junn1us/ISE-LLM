from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .action_space import ActionCatalog


@dataclass
class StepInfo:
    rmse: float
    mask: np.ndarray
    step: int
    reward: float
    done: bool


class FixtureCore:
    """
    Canonical SmartFixture core that owns the true state + transition kernel.
    Shared by both student and teacher environments.
    """

    def __init__(
        self,
        seed: int = 13,
        anchor_mode: str = "stride",
        anchor_stride: int = 4,
        max_fixtures: int = 8,
        max_steps: int = 20,
    ):
        self.rng = np.random.default_rng(seed)
        self.max_fixtures = int(max_fixtures)
        self.max_steps = int(max_steps)

        # Sheet grid (9x9)
        self.NX = 9
        self.NY = 9
        xs = np.arange(self.NX)
        ys = np.arange(self.NY)
        GX, GY = np.meshgrid(xs, ys)
        self.coords = np.stack([GX.reshape(-1), GY.reshape(-1)], axis=1)

        # Anchor candidates
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
            anchors = [
                (0, 0), (8, 0), (0, 8), (8, 8),
                (4, 0), (4, 8), (0, 4), (8, 4),
                (2, 2), (6, 2), (2, 6), (6, 6),
            ]
        elif anchor_mode == "all":
            anchors = all_nodes
        else:
            raise ValueError(f"Unknown anchor_mode={anchor_mode}.")

        self.anchor_xy = np.array(anchors, dtype=np.float32)
        self.num_anchors = int(self.anchor_xy.shape[0])
        self.action_catalog = ActionCatalog(self.num_anchors)

        # State
        self.mask = np.zeros(self.num_anchors, dtype=np.float32)
        self.disp = np.zeros((self.coords.shape[0], 2), dtype=np.float32)
        self.target_disp = self._make_target_bowl()
        self.prev_err = None
        self.step_count = 0

    def _make_target_bowl(self) -> np.ndarray:
        center = np.array([4.0, 4.0])
        diffs = self.coords - center[None, :]
        r2 = np.sum(diffs ** 2, axis=1)
        sag = 1.5 * np.exp(-r2 / (2 * (6.0)))
        bowl = np.zeros((self.coords.shape[0], 2), dtype=np.float32)
        bowl[:, 1] = sag.astype(np.float32)
        return bowl

    # ----------------------------
    # Core simulation (surrogate)
    # ----------------------------
    def sim_apply(self) -> None:
        center = np.array([4.0, 4.0])
        diffs = self.coords - center[None, :]
        r2 = np.sum(diffs ** 2, axis=1)
        free_sag = 2.0 * np.exp(-r2 / (2 * (6.0)))

        disp = np.zeros_like(self.disp)
        disp[:, 1] = free_sag

        for k, (ax, ay) in enumerate(self.anchor_xy):
            if self.mask[k] > 0.5:
                diffa = self.coords - np.array([ax, ay])
                r2a = np.sum(diffa ** 2, axis=1)
                influence = np.exp(-r2a / (2 * (2.0 ** 2)))
                disp[:, 1] *= (1.0 - 0.8 * influence)

        self.disp = disp.astype(np.float32)

    def sim_error(self) -> float:
        cur = self.disp[:, 1]
        tgt = self.target_disp[:, 1]
        mse = np.mean((cur - tgt) ** 2)
        return float(np.sqrt(mse))

    # ----------------------------
    # Feature extraction
    # ----------------------------
    def get_student_features(self) -> np.ndarray:
        feats: List[float] = []
        for (ax, ay) in self.anchor_xy:
            d2 = np.sum((self.coords - np.array([ax, ay])) ** 2, axis=1)
            idx = int(np.argmin(d2))
            feats.append(float(self.disp[idx, 1]))

        sag_all = self.disp[:, 1]
        feats.append(float(np.max(sag_all)))
        feats.append(float(np.mean(sag_all)))
        feats.append(float(np.std(sag_all)))
        feats.append(float(self.max_steps - self.step_count))
        feats.append(float(self.mask.sum()))
        feats.append(float(self.step_count) / max(self.max_steps, 1))
        return np.array(feats, dtype=np.float32)

    def get_teacher_features(self) -> Dict[str, object]:
        anchor_sags = []
        for (ax, ay) in self.anchor_xy:
            d2 = np.sum((self.coords - np.array([ax, ay])) ** 2, axis=1)
            idx = int(np.argmin(d2))
            anchor_sags.append(float(self.disp[idx, 1]))

        anchor_sags_np = np.array(anchor_sags, dtype=np.float32)
        d_max = float(np.max(anchor_sags_np)) if anchor_sags_np.size else 0.0
        d_mean = float(np.mean(anchor_sags_np)) if anchor_sags_np.size else 0.0
        return {
            "anchor_sags": anchor_sags,
            "d_max": d_max,
            "d_mean": d_mean,
            "mask": self.mask.astype(int).tolist(),
            "budget_left": int(self.max_steps - self.step_count),
        }

    # ----------------------------
    # RL API
    # ----------------------------
    def reset(self) -> np.ndarray:
        self.mask[:] = 0.0
        self.step_count = 0
        self.sim_apply()
        self.prev_err = self.sim_error()
        return self.get_student_features()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, StepInfo]:
        action = self.action_catalog.get(action_idx)
        if action.kind == "add":
            if self.mask[action.node_id] < 0.5 and self.mask.sum() < self.max_fixtures:
                self.mask[action.node_id] = 1.0
        elif action.kind == "remove":
            if self.mask[action.node_id] > 0.5:
                self.mask[action.node_id] = 0.0
        elif action.kind == "noop":
            pass
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

        self.step_count += 1
        self.sim_apply()
        err_now = self.sim_error()
        denom = max(self.prev_err if self.prev_err is not None else err_now, 1e-6)
        reward = 1.0 - (err_now / denom)
        reward = float(np.clip(reward, -1.0, 1.0))
        self.prev_err = err_now

        obs = self.get_student_features()
        done = self.step_count >= self.max_steps
        info = StepInfo(
            rmse=err_now,
            mask=self.mask.copy(),
            step=self.step_count,
            reward=reward,
            done=done,
        )
        return obs, reward, done, info

    # ----------------------------
    # Candidate action helpers
    # ----------------------------
    def candidate_actions(
        self,
        topk_add: int = 5,
        topk_remove: int = 3,
        include_noop: bool = True,
    ) -> List[int]:
        anchor_sags = self.get_teacher_features()["anchor_sags"]
        anchor_sags_np = np.array(anchor_sags, dtype=np.float32)

        add_ids = np.argsort(-anchor_sags_np)  # largest sag first
        remove_ids = np.argsort(anchor_sags_np)  # smallest sag first

        add_candidates = [int(i) for i in add_ids if self.mask[i] < 0.5][:topk_add]
        remove_candidates = [int(i) for i in remove_ids if self.mask[i] > 0.5][:topk_remove]

        action_ids: List[int] = []
        for i in add_candidates:
            action_ids.append(self.action_catalog.index("add", i))
        for i in remove_candidates:
            action_ids.append(self.action_catalog.index("remove", i))
        if include_noop:
            action_ids.append(self.action_catalog.index("noop"))

        return action_ids
