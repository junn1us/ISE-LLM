from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from .action_space import ActionCatalog


MATERIAL_PRESETS = {
    "structural_steel": {
        "display_name": "structural steel",
        "youngs_modulus_pa": 200.0e9,
        "density_kg_m3": 7850.0,
        "poisson_ratio": 0.30,
        "thickness_m": 0.002,
    },
    "composite": {
        "display_name": "composite",
        "youngs_modulus_pa": 70.0e9,
        "density_kg_m3": 1600.0,
        "poisson_ratio": 0.25,
        "thickness_m": 0.004,
    },
}


@dataclass
class StepInfo:
    rmse: float
    max_deformation: float
    mask: np.ndarray
    step: int
    reward: float
    done: bool
    feedback_available: bool = True
    useful_fixture_added: bool = False
    useful_fixture_gain: float = 0.0
    useful_fixture_bonus: float = 0.0
    action_idx: int = -1
    action_kind: str = "noop"
    node_id: int | None = None
    applied: bool = False
    active_fixtures_before: int = 0
    active_fixtures_after: int = 0
    gravity_considered: bool = True
    pre_threshold_progress_bonus: float = 0.0
    base_reward: float = 0.0
    fixture_usage_penalty: float = 0.0
    rmse_before: float = float("nan")
    rmse_after: float = float("nan")
    rmse_delta: float = float("nan")
    max_deformation_before: float = float("nan")
    max_deformation_after: float = float("nan")
    max_deformation_delta: float = float("nan")


class FixtureCore:
    """
    Canonical SmartFixture core that owns the true state + transition kernel.
    Shared by both student and teacher environments.
    """

    def __init__(
        self,
        seed: int = 13,
        anchor_mode: str = "grid100",
        anchor_stride: int = 4,
        max_fixtures: int = -1,
        max_steps: int = 20,
        grid_size: int = 10,
        sheet_size_m: float = 2.0,
        sheet_length_m: float | None = None,
        sheet_width_m: float | None = None,
        sheet_thickness_m: float | None = None,
        material: str = "structural_steel",
        reward_scale: float = 1.0,
        fixture_penalty: float = 0.0,
        invalid_action_penalty: float = 0.0,
        min_feedback_fixtures: int = 0,
        side_support: str = "none",
        reward_mode: str = "normalized_rmse",
        reward_clip_min: float | None = None,
        reward_clip_max: float | None = 1.0,
        useful_fixture_bonus: float = 0.0,
        useful_fixture_min_improvement: float = 1e-6,
        pre_threshold_progress_bonus: float = 0.0,
        instance_id: str = "fixed",
        load_center_x: float = 0.5,
        load_center_y: float = 0.5,
        load_width_x: float = 0.75,
        load_width_y: float = 0.75,
        load_scale: float = 1.0,
        load_pattern: str = "single",
        influence_width: float = 0.22,
        support_strength: float = 0.8,
        target_scale: float = 0.75,
        include_instance_features: bool = False,
    ):
        self.instance_id = str(instance_id)
        self.load_center_x, self.load_center_y = float(load_center_x), float(load_center_y)
        self.load_width_x, self.load_width_y = float(load_width_x), float(load_width_y)
        self.load_scale, self.load_pattern = float(load_scale), str(load_pattern)
        self.influence_width, self.support_strength = float(influence_width), float(support_strength)
        self.target_scale = float(target_scale)
        self.include_instance_features = bool(include_instance_features)
        values = [load_center_x, load_center_y, load_width_x, load_width_y,
                  load_scale, influence_width, support_strength, target_scale]
        if not np.isfinite(values).all():
            raise ValueError("Instance parameters must be finite")
        if not (0 <= load_center_x <= 1 and 0 <= load_center_y <= 1):
            raise ValueError("Load center must be inside normalized sheet coordinates")
        if min(load_width_x, load_width_y, load_scale, influence_width) <= 0:
            raise ValueError("Load widths/scale and influence width must be positive")
        if not 0 <= support_strength <= 1 or not 0 <= target_scale <= 1:
            raise ValueError("Support strength and target scale must lie in [0,1]")
        if load_pattern not in {"single", "double"}:
            raise ValueError("load_pattern must be single or double")
        self.rng = np.random.default_rng(seed)
        self.max_fixtures = int(max_fixtures)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        material_key = str(material).lower().replace("-", "_").replace(" ", "_")
        if material_key not in MATERIAL_PRESETS:
            raise ValueError(f"Unknown material={material}. Expected one of {sorted(MATERIAL_PRESETS)}")
        preset = MATERIAL_PRESETS[material_key]
        self.material = material_key
        self.material_name = str(preset["display_name"])
        self.youngs_modulus_pa = float(preset["youngs_modulus_pa"])
        self.density_kg_m3 = float(preset["density_kg_m3"])
        self.poisson_ratio = float(preset["poisson_ratio"])

        # Physical sheet dimensions (metres); sheet_size_m remains as a square legacy default.
        self.sheet_size_m = float(sheet_size_m)
        self.sheet_length_m = float(sheet_length_m if sheet_length_m is not None else sheet_size_m)
        self.sheet_width_m = float(sheet_width_m if sheet_width_m is not None else sheet_size_m)
        self.sheet_thickness_m = float(
            sheet_thickness_m if sheet_thickness_m is not None else preset["thickness_m"]
        )
        if self.sheet_length_m <= 0.0 or self.sheet_width_m <= 0.0 or self.sheet_thickness_m <= 0.0:
            raise ValueError("sheet dimensions and thickness must be positive")
        self.reward_scale = float(reward_scale)
        self.fixture_penalty = float(fixture_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.reward_mode = str(reward_mode).lower().replace("-", "_").replace(" ", "_")
        if self.reward_mode not in {
            "relative_improvement", "rmse_improvement", "deformation_reduction",
            "normalized_rmse", "negative_rmse", "max_deformation_reduction"
        }:
            raise ValueError(
                "reward_mode must be one of: relative_improvement, rmse_improvement, "
                "deformation_reduction, normalized_rmse, negative_rmse, "
                "max_deformation_reduction"
            )
        self.reward_clip_min = None if reward_clip_min is None else float(reward_clip_min)
        self.reward_clip_max = None if reward_clip_max is None else float(reward_clip_max)
        self.useful_fixture_bonus = float(useful_fixture_bonus)
        self.useful_fixture_min_improvement = float(useful_fixture_min_improvement)
        self.pre_threshold_progress_bonus = float(pre_threshold_progress_bonus)
        if (
            self.useful_fixture_bonus < 0.0
            or self.useful_fixture_min_improvement < 0.0
            or self.pre_threshold_progress_bonus < 0.0
        ):
            raise ValueError("fixture bonuses and minimum improvement must be >= 0")
        if (
            self.reward_clip_min is not None
            and self.reward_clip_max is not None
            and self.reward_clip_min > self.reward_clip_max
        ):
            raise ValueError("reward_clip_min cannot exceed reward_clip_max")
        self.min_feedback_fixtures = int(min_feedback_fixtures)
        if self.min_feedback_fixtures < 0:
            raise ValueError("min_feedback_fixtures must be >= 0")
        if self.max_fixtures > 0 and self.min_feedback_fixtures > self.max_fixtures:
            raise ValueError("min_feedback_fixtures cannot exceed max_fixtures")
        self.side_support = str(side_support).lower().replace("-", "_").replace(" ", "_")
        if self.side_support not in {"none", "free"}:
            raise ValueError("side_support must be 'none'/'free'; clamped side supports are not enabled")

        # Sheet grid — coordinates in physical metres [0, length] × [0, width]
        self.NX = int(grid_size)
        self.NY = int(grid_size)
        if self.NX < 3 or self.NY < 3:
            raise ValueError("grid_size must be >= 3")
        xs = np.linspace(0.0, self.sheet_length_m, self.NX)
        ys = np.linspace(0.0, self.sheet_width_m, self.NY)
        GX, GY = np.meshgrid(xs, ys)
        self.coords = np.stack([GX.reshape(-1), GY.reshape(-1)], axis=1)

        # anchor candidate locations
        all_nodes = [(x, y) for x in range(self.NX) for y in range(self.NY)]
        if anchor_mode == "boundary":
            anchors = [(x, y) for (x, y) in all_nodes
                       if (x == 0 or x == self.NX - 1 or y == 0 or y == self.NY - 1)]
        elif anchor_mode == "grid100":
            # Dense 10x10 design space expected by request.
            if self.NX != 10 or self.NY != 10:
                raise ValueError("anchor_mode='grid100' requires grid_size=10")
            anchors = all_nodes
        elif anchor_mode == "stride":
            s = int(anchor_stride)
            if s <= 0:
                raise ValueError("anchor_stride must be >= 1")
            anchors = [(x, y) for x in range(0, self.NX, s)
                       for y in range(0, self.NY, s)]
        elif anchor_mode == "twelve":
            if self.NX != 9 or self.NY != 9:
                raise ValueError("anchor_mode='twelve' currently requires grid_size=9")
            anchors = [
                (0, 0), (8, 0), (0, 8), (8, 8),
                (4, 0), (4, 8), (0, 4), (8, 4),
                (2, 2), (6, 2), (2, 6), (6, 6),
            ]
        elif anchor_mode == "all":
            anchors = all_nodes
        else:
            raise ValueError(f"Unknown anchor_mode={anchor_mode}.")

        # Convert anchor grid-index tuples to physical metre coordinates.
        _dx = self.sheet_length_m / max(self.NX - 1, 1)
        _dy = self.sheet_width_m / max(self.NY - 1, 1)
        _anchor_idx = np.array(anchors, dtype=np.float32)  # (N, 2) grid indices
        self.anchor_xy = _anchor_idx * np.array([_dx, _dy], dtype=np.float32)
        self.num_anchors = int(self.anchor_xy.shape[0])
        self.action_catalog = ActionCatalog(self.num_anchors)

        # precompute anchor helpers used in every step/feature extraction
        # avoid repeated O(num_nodes * num_anchors) nearest-node scans
        self._anchor_node_idx = np.empty(self.num_anchors, dtype=np.int32)
        for k, (ax, ay) in enumerate(self.anchor_xy):
            d2 = np.sum((self.coords - np.array([ax, ay], dtype=np.float32)) ** 2, axis=1)
            self._anchor_node_idx[k] = int(np.argmin(d2))

        # Precompute per-anchor influence fields used by sim_apply().
        # influence_sigma: ~22 % of characteristic sheet span in metres (spatial reach of one fixture).
        characteristic_span = float(np.sqrt(self.sheet_length_m * self.sheet_width_m))
        influence_sigma = self.influence_width * characteristic_span
        self._anchor_influence = np.empty((self.num_anchors, self.coords.shape[0]), dtype=np.float32)
        for k, (ax, ay) in enumerate(self.anchor_xy):
            diffa = self.coords - np.array([ax, ay], dtype=np.float32)
            r2a = np.sum(diffa ** 2, axis=1)
            self._anchor_influence[k] = np.exp(-r2a / (2 * (influence_sigma ** 2))).astype(np.float32)

        # Precompute gravity-induced free-sag surrogate. The amplitude is scaled by
        # self-weight divided by plate bending stiffness relative to the 2 mm steel baseline.
        center = np.array([self.sheet_length_m * self.load_center_x,
                   self.sheet_width_m * self.load_center_y], dtype=np.float32)
        diffs = self.coords - center[None, :]
        normalized_r2 = (diffs[:, 0] / (self.load_width_x * self.sheet_length_m)) ** 2
        normalized_r2 += (diffs[:, 1] / (self.load_width_y * self.sheet_width_m)) ** 2
        baseline_stiffness = 200.0e9 * (0.002 ** 3) / (12.0 * (1.0 - 0.30 ** 2))
        bending_stiffness = self.youngs_modulus_pa * (self.sheet_thickness_m ** 3)
        bending_stiffness /= 12.0 * max(1.0 - self.poisson_ratio ** 2, 1e-6)
        self_weight_ratio = (self.density_kg_m3 * self.sheet_thickness_m) / (7850.0 * 0.002)
        stiffness_ratio = baseline_stiffness / max(bending_stiffness, 1e-12)
        span_ratio = (characteristic_span / 2.0) ** 4
        sag_peak_mm = 2.0 * self_weight_ratio * stiffness_ratio * span_ratio
        shape = np.exp(-0.5 * normalized_r2)
        if self.load_pattern == "double":
            reflected = np.array([self.sheet_length_m, self.sheet_width_m]) - center
            delta = self.coords - reflected
            second_r2 = (delta[:, 0] / (self.load_width_y * self.sheet_length_m)) ** 2
            second_r2 += (delta[:, 1] / (self.load_width_x * self.sheet_width_m)) ** 2
            shape = 0.6 * shape + 0.4 * np.exp(-0.5 * second_r2)
        self._free_sag = (self.load_scale * sag_peak_mm * shape).astype(np.float32)

        # State
        self.mask = np.zeros(self.num_anchors, dtype=np.float32)
        self.disp = np.zeros((self.coords.shape[0], 2), dtype=np.float32)
        self.target_disp = self._make_target_bowl()
        self.free_err = None
        self.prev_err = None
        self.free_max_deformation = None
        self.prev_max_deformation = None
        self.step_count = 0

    def _make_target_bowl(self) -> np.ndarray:
        # Target bowl centred at the sheet mid-point in physical metres.
        center = np.array([self.sheet_length_m / 2.0, self.sheet_width_m / 2.0])
        diffs = self.coords - center[None, :]
        normalized_r2 = (diffs[:, 0] / max(0.75 * self.sheet_length_m, 1e-6)) ** 2
        normalized_r2 += (diffs[:, 1] / max(0.75 * self.sheet_width_m, 1e-6)) ** 2
        sag = self.target_scale * float(np.max(self._free_sag)) * np.exp(-0.5 * normalized_r2)
        bowl = np.zeros((self.coords.shape[0], 2), dtype=np.float32)
        bowl[:, 1] = sag.astype(np.float32)
        return bowl

    # ----------------------------
    # Core simulation (surrogate)
    # ----------------------------
    def sim_apply(self) -> None:
        disp = np.zeros_like(self.disp)
        # With fewer than the minimum fixture count, the sheet has no supports
        # and gravity is intentionally excluded.  This avoids implicitly
        # treating the four corners (or any boundary) as fixed during the
        # fixture-initiation stage.
        if self.gravity_considered:
            disp_y = self._free_sag.copy()
            active = np.flatnonzero(self.mask > 0.5)
            for k in active:
                disp_y *= (1.0 - self.support_strength * self._anchor_influence[k])
        else:
            disp_y = np.zeros_like(self._free_sag)

        disp[:, 1] = disp_y

        self.disp = disp.astype(np.float32)

    def sim_error(self) -> float:
        cur = self.disp[:, 1]
        tgt = self.target_disp[:, 1]
        mse = np.mean((cur - tgt) ** 2)
        return float(np.sqrt(mse))

    def sim_max_deformation(self) -> float:
        return float(np.max(np.abs(self.disp[:, 1])))

    def _feedback_available(self) -> bool:
        return self.gravity_considered

    @property
    def gravity_considered(self) -> bool:
        """Whether the fixture threshold permits gravity in the simulation."""
        return int(self.mask.sum()) >= self.min_feedback_fixtures

    def _reported_sag_y(self) -> np.ndarray:
        # Never leak the gravity-only field into the initiation heuristic.
        return self.disp[:, 1]

    # ----------------------------
    # Feature extraction
    # ----------------------------
    def get_student_features(self) -> np.ndarray:
        reported_sag = self._reported_sag_y()
        feats: List[float] = reported_sag[self._anchor_node_idx].astype(float).tolist()
        feats.extend(self.mask.astype(float).tolist())

        sag_all = reported_sag
        feats.append(float(np.max(np.abs(sag_all))))
        feats.append(float(np.mean(sag_all)))
        feats.append(float(np.std(sag_all)))
        feats.append(float(self.max_steps - self.step_count))
        feats.append(float(self.mask.sum()))
        feats.append(float(self.step_count) / max(self.max_steps, 1))
        if self.include_instance_features:
            # Explicit instance context, not just different random seeds. Fixed grid
            # and anchor counts keep this schema constant across held-out geometries.
            feats.extend(self.anchor_xy.reshape(-1).astype(float).tolist())
            feats.extend(self._free_sag.astype(float).tolist())
            feats.extend(self.target_disp[:, 1].astype(float).tolist())
            feats.extend([self.sheet_length_m, self.sheet_width_m,
                self.sheet_thickness_m * 1000, self.youngs_modulus_pa / 1e11,
                self.density_kg_m3 / 10000, self.poisson_ratio,
                self.influence_width, self.support_strength])
        return np.array(feats, dtype=np.float32)

    def get_teacher_features(self) -> Dict[str, object]:
        reported_sag = self._reported_sag_y()
        anchor_sags = reported_sag[self._anchor_node_idx].astype(float).tolist()

        anchor_sags_np = np.array(anchor_sags, dtype=np.float32)
        d_max = float(np.max(np.abs(anchor_sags_np))) if anchor_sags_np.size else 0.0
        d_mean = float(np.mean(anchor_sags_np)) if anchor_sags_np.size else 0.0
        return {
            "anchor_sags": anchor_sags,
            "d_max": d_max,
            "d_mean": d_mean,
            "mask": self.mask.astype(int).tolist(),
            "budget_left": int(self.max_steps - self.step_count),
            "feedback_available": self._feedback_available(),
            "gravity_considered": self.gravity_considered,
            "min_feedback_fixtures": int(self.min_feedback_fixtures),
        }

    # ----------------------------
    # RL API
    # ----------------------------
    def reset(self) -> np.ndarray:
        self.mask[:] = 0.0
        self.step_count = 0
        self.sim_apply()
        # Keep normalized reward referenced to the gravity-loaded, unfixtured
        # sheet even though gravity is hidden before the fixture gate opens.
        self.free_err = float(
            np.sqrt(np.mean((self._free_sag - self.target_disp[:, 1]) ** 2))
        )
        self.prev_err = self.free_err
        self.free_max_deformation = float(np.max(np.abs(self._free_sag)))
        self.prev_max_deformation = self.free_max_deformation
        return self.get_student_features()

    def _compute_reward(self, err_now: float) -> float:
        if self.reward_mode == "max_deformation_reduction":
            metric_before = self.prev_max_deformation
            denom = max(self.free_max_deformation, np.finfo(float).tiny)
            reward = self.reward_scale * ((metric_before - err_now) / denom)
        elif self.reward_mode == "relative_improvement":
            denom = max(self.prev_err if self.prev_err is not None else err_now, 1e-6)
            reward = self.reward_scale * (1.0 - (err_now / denom))
        elif self.reward_mode in {"rmse_improvement", "deformation_reduction"}:
            err_before = self.prev_err if self.prev_err is not None else err_now
            denom = max(self.free_err if self.free_err is not None else err_before, 1e-6)
            reward = self.reward_scale * ((err_before - err_now) / denom)
        elif self.reward_mode == "normalized_rmse":
            denom = max(self.free_err if self.free_err is not None else err_now, 1e-6)
            reward = self.reward_scale * (1.0 - (err_now / denom))
        elif self.reward_mode == "negative_rmse":
            reward = -self.reward_scale * err_now
        else:
            raise ValueError(f"Unknown reward_mode={self.reward_mode}")

        if self.reward_clip_min is not None or self.reward_clip_max is not None:
            lo = -np.inf if self.reward_clip_min is None else self.reward_clip_min
            hi = np.inf if self.reward_clip_max is None else self.reward_clip_max
            reward = float(np.clip(reward, lo, hi))
        return float(reward)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, StepInfo]:
        if self.step_count >= self.max_steps:
            raise RuntimeError("Episode terminated; reset before stepping again")
        if not 0 <= action_idx < self.action_catalog.action_dim:
            raise ValueError("Action index out of range")
        action = self.action_catalog.get(action_idx)
        unlimited_fixtures = self.max_fixtures <= 0
        applied = False
        active_before = int(self.mask.sum())
        rmse_before = float(self.sim_error())
        max_deformation_before = float(self.sim_max_deformation())
        collecting_initial_fixtures = active_before < self.min_feedback_fixtures
        if action.kind == "add":
            if self.mask[action.node_id] < 0.5 and (unlimited_fixtures or self.mask.sum() < self.max_fixtures):
                self.mask[action.node_id] = 1.0
                applied = True
        elif action.kind == "remove":
            if active_before > self.min_feedback_fixtures and self.mask[action.node_id] > 0.5:
                self.mask[action.node_id] = 0.0
                applied = True
        elif action.kind == "noop":
            applied = not collecting_initial_fixtures
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

        self.step_count += 1
        self.sim_apply()
        rmse_now = self.sim_error()
        max_deformation_now = self.sim_max_deformation()
        active_after = int(self.mask.sum())
        feedback_available = self._feedback_available()
        if feedback_available:
            reward_metric = (
                max_deformation_now
                if self.reward_mode == "max_deformation_reduction"
                else rmse_now
            )
            reward = self._compute_reward(reward_metric)
        else:
            reward = 0.0
        base_reward = float(reward)

        progress_bonus = 0.0
        if (
            self.reward_mode != "max_deformation_reduction"
            and
            action.kind == "add"
            and applied
            and active_after < self.min_feedback_fixtures
        ):
            progress_bonus = self.pre_threshold_progress_bonus
            reward += progress_bonus

        useful_fixture_gain = 0.0
        useful_fixture_added = False
        useful_fixture_bonus = 0.0
        if (
            feedback_available
            and active_before >= self.min_feedback_fixtures
            and action.kind == "add"
            and applied
        ):
            if self.reward_mode == "max_deformation_reduction":
                useful_fixture_gain = float(max_deformation_before - max_deformation_now)
                useful_fixture_added = useful_fixture_gain > 0.0
            else:
                useful_fixture_gain = float(rmse_before - rmse_now)
                useful_fixture_added = useful_fixture_gain >= self.useful_fixture_min_improvement
            if useful_fixture_added and self.reward_mode != "max_deformation_reduction":
                useful_fixture_bonus = self.useful_fixture_bonus
                reward += useful_fixture_bonus

        fixture_usage_penalty = 0.0
        if (
            self.reward_mode != "max_deformation_reduction"
            and
            feedback_available
            and applied
            and action.kind in {"add", "remove"}
            and self.fixture_penalty > 0.0
        ):
            fixture_capacity = self.max_fixtures if self.max_fixtures > 0 else self.num_anchors
            fixture_usage_penalty = self.fixture_penalty * (active_after / max(fixture_capacity, 1))
            reward -= fixture_usage_penalty

        # Penalize invalid/disallowed actions (e.g., add on full budget or remove inactive).
        if feedback_available and (not applied) and self.invalid_action_penalty > 0.0:
            reward -= self.invalid_action_penalty

        if self.reward_clip_min is not None or self.reward_clip_max is not None:
            lo = -np.inf if self.reward_clip_min is None else self.reward_clip_min
            hi = np.inf if self.reward_clip_max is None else self.reward_clip_max
            reward = float(np.clip(reward, lo, hi))

        reward = float(reward)
        if feedback_available:
            self.prev_err = rmse_now
            self.prev_max_deformation = max_deformation_now

        obs = self.get_student_features()
        done = self.step_count >= self.max_steps
        info = StepInfo(
            rmse=rmse_now if feedback_available else float("nan"),
            max_deformation=max_deformation_now if feedback_available else float("nan"),
            mask=self.mask.copy(),
            step=self.step_count,
            reward=reward,
            done=done,
            feedback_available=feedback_available,
            useful_fixture_added=useful_fixture_added,
            useful_fixture_gain=useful_fixture_gain,
            useful_fixture_bonus=useful_fixture_bonus,
            action_idx=int(action_idx),
            action_kind=action.kind,
            node_id=action.node_id,
            applied=applied,
            active_fixtures_before=active_before,
            active_fixtures_after=active_after,
            gravity_considered=self.gravity_considered,
            pre_threshold_progress_bonus=progress_bonus,
            base_reward=base_reward,
            fixture_usage_penalty=fixture_usage_penalty,
            rmse_before=rmse_before if active_before >= self.min_feedback_fixtures else float("nan"),
            rmse_after=rmse_now if feedback_available else float("nan"),
            rmse_delta=(
                rmse_before - rmse_now
                if active_before >= self.min_feedback_fixtures and feedback_available
                else float("nan")
            ),
            max_deformation_before=(
                self.free_max_deformation
                if active_before < self.min_feedback_fixtures and feedback_available
                else max_deformation_before
                if active_before >= self.min_feedback_fixtures
                else float("nan")
            ),
            max_deformation_after=(
                max_deformation_now if feedback_available else float("nan")
            ),
            max_deformation_delta=(
                (
                    self.free_max_deformation
                    if active_before < self.min_feedback_fixtures
                    else max_deformation_before
                ) - max_deformation_now
                if feedback_available
                else float("nan")
            ),
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
        anchor_sags_np = self._reported_sag_y()[self._anchor_node_idx]
        active_count = int(self.mask.sum())
        at_fixture_cap = self.max_fixtures > 0 and active_count >= self.max_fixtures

        if self.gravity_considered:
            add_ids = np.argsort(-anchor_sags_np)  # largest sag first
        else:
            # Gravity is unavailable during fixture initiation. Rank candidates
            # only by geometry: start near the sheet centre, then maximize
            # distance from fixtures already chosen for broad spatial coverage.
            inactive = np.flatnonzero(self.mask < 0.5)
            active = np.flatnonzero(self.mask > 0.5)
            if active.size:
                delta = self.anchor_xy[inactive, None, :] - self.anchor_xy[active, :][None, :, :]
                min_d2 = np.min(np.sum(delta ** 2, axis=2), axis=1)
                add_ids = inactive[np.argsort(-min_d2, kind="stable")]
            else:
                center = np.array(
                    [self.sheet_length_m / 2.0, self.sheet_width_m / 2.0],
                    dtype=np.float32,
                )
                center_d2 = np.sum((self.anchor_xy[inactive] - center) ** 2, axis=1)
                add_ids = inactive[np.argsort(center_d2, kind="stable")]
        remove_ids = np.argsort(anchor_sags_np)  # smallest sag first

        add_candidates = (
            []
            if at_fixture_cap
            else [int(i) for i in add_ids if self.mask[i] < 0.5][:topk_add]
        )
        if active_count < self.min_feedback_fixtures:
            return [self.action_catalog.index("add", i) for i in add_candidates]

        if active_count > self.min_feedback_fixtures:
            remove_candidates = [int(i) for i in remove_ids if self.mask[i] > 0.5][:topk_remove]
        else:
            remove_candidates = []

        action_ids: List[int] = []
        for i in add_candidates:
            action_ids.append(self.action_catalog.index("add", i))
        for i in remove_candidates:
            action_ids.append(self.action_catalog.index("remove", i))
        if include_noop:
            action_ids.append(self.action_catalog.index("noop"))

        return action_ids
