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
        "stage": int(core.step_count),
        "budget_left": int(features["budget_left"]),
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
    Compact JSON-only prompt for LLM teacher.
    Must return JSON with action_logits or action_probs.
    """
    return json.dumps(ctx, separators=(",", ":"))
