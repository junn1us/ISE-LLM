import json
from typing import Dict, List

from smartfixture_llm4teach.core.fixture_core import FixtureCore


def build_teacher_context(
    core: FixtureCore,
    candidate_action_ids: List[int],
) -> Dict[str, object]:
    features = core.get_teacher_features()
    anchor_sags = features["anchor_sags"]
    mask = features["mask"]

    action_options = []
    for action_id in candidate_action_ids:
        spec = core.action_catalog.get(action_id)
        action_options.append(
            {
                "action_id": int(action_id),
                "type": spec.kind,
                "node_id": spec.node_id,
                "node_sag": float(anchor_sags[spec.node_id]) if spec.node_id is not None else None,
                "node_active": int(mask[spec.node_id]) if spec.node_id is not None else None,
                "label": core.action_catalog.describe(action_id),
            }
        )

    ctx = {
        "reward_mode": core.reward_mode,
        "objective": "minimize maximum absolute deformation" if core.reward_mode == "max_deformation_reduction" else "minimize RMSE to target displacement",
        "target_displacement": core.target_disp[:, 1].astype(float).tolist(),
        "anchor_coordinates_m": core.anchor_xy.astype(float).tolist(),
        "sheet_dimensions_m": [core.sheet_length_m, core.sheet_width_m, core.sheet_thickness_m],
        "material": core.material,
        "instance_id": core.instance_id,
        "model_limitations": "Synthetic Gaussian attenuation surrogate; not calibrated structural mechanics",
        "free_sag_field": core._free_sag.astype(float).tolist(),
        "node_coordinates_m": core.coords.astype(float).tolist(),
        "support_strength": core.support_strength,
        "support_influence_width_fraction": core.influence_width,
        "max_fixtures": core.max_fixtures,
        "stage": int(core.step_count),
        "budget_left": int(features["budget_left"]),
        "active_fixture_count": int(sum(mask)),
        "gravity_fixture_threshold": int(features["min_feedback_fixtures"]),
        "gravity_considered": bool(features["gravity_considered"]),
        "d_max_mm": float(features["d_max"]),
        "d_mean_mm": float(features["d_mean"]),
        "anchor_sags": [float(x) for x in anchor_sags],
        "fixture_mask": list(mask),
        "candidate_action_ids": [int(x) for x in candidate_action_ids],
        "candidate_actions": action_options,
        "ops_enabled": ["add", "remove", "noop"],
    }
    return ctx


def build_teacher_prompt(ctx: Dict[str, object]) -> str:
    """
    Serialise the context dict to a compact JSON string.
    The native gpt-oss adapter (build_gptoss_llm_fn) and the HF adapter
    (build_hf_llm_fn) both call json.loads() on this string to reconstruct
    the context and build their own internal prompt.  This function must
    therefore return valid JSON.
    """
    return json.dumps(ctx, separators=(",", ":"))
