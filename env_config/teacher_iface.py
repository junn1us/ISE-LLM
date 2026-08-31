#teacher_iface.py
#small adapter functions (translator_context, parse_teacher_reply) used by multiple tools
import json
import numpy as np


def translator_context(env, obs, topk: int = 4):
    """
    A compact JSON-compatible context dict from (env, obs)
    This version handles ANY number of anchors (9, 12, etc.) and provides
    deformation-driven context for interior placement decisions =0w0=
    Assumptions:
    env.num_anchors exists
    obs ~ a 1D numpy array of deformation features
    obs[:num_anchors] are sag values at anchor locations
    obs[num_anchors:] are global stats (max, mean, std)
    """
    num_anchors = getattr(env, "num_anchors", getattr(env, "A", 0))
    if num_anchors <= 0:
        raise ValueError("Env must define num_anchors or A > 0.")

    obs = np.asarray(obs, dtype=float)
    
    # Extract anchor sag values
    anchor_sags = obs[:num_anchors]
    
    # Global stats (max, mean, std)
    if len(obs) > num_anchors:
        d_max = float(obs[num_anchors])  # max sag
    else:
        d_max = float(np.max(np.abs(anchor_sags)))

    # Pick top-k anchors by absolute deflection (sag magnitude)
    k = min(topk, num_anchors)
    top_ids = np.argsort(-np.abs(anchor_sags))[:k]

    top_nodes_by_deflection = [
        {"id": int(i), "mm": float(anchor_sags[i])} for i in top_ids
    ]

    # Get anchor positions for providing deformation context
    anchor_xy = getattr(env, "anchor_xy", None)
    if anchor_xy is not None:
        anchor_xy_np = np.asarray(anchor_xy, dtype=float)
        # Determine boundary vs interior using relative position on the sheet.
        x_max = float(anchor_xy_np[:, 0].max()) if anchor_xy_np.shape[0] > 0 else 1.0
        y_max = float(anchor_xy_np[:, 1].max()) if anchor_xy_np.shape[0] > 0 else 1.0
        interior_flags = []
        for xy in anchor_xy_np:
            is_interior = (
                0.01 * x_max < xy[0] < 0.99 * x_max
                and 0.01 * y_max < xy[1] < 0.99 * y_max
            )
            interior_flags.append(int(is_interior))
    else:
        interior_flags = [0] * num_anchors

    ctx = {
        "stage": 0,
        "budget_left": int(getattr(env, "max_steps", 20)),
        "d_max_mm": d_max,
        "e_iface_mm": 0.0,  # not used in this toy env
        "top_nodes_by_deflection": top_nodes_by_deflection,
        "candidate_nodes": list(range(num_anchors)),
        "candidate_types": [
            {
                "id": i,
                "type": "interior" if interior_flags[i] else "boundary",
                "current_sag": float(anchor_sags[i]),
            }
            for i in range(num_anchors)
        ],
        "disabled_neighbors": [],
        "spacing_mm": 1.0,
        "goals": {"d_max_mm": "<=0.2", "e_iface_mm": "<=0.0"},
        "ops_enabled": ["add", "remove", "noop"],
        "seams_enabled": [],
    }
    return ctx


def teacher_llm_fn(ctx_json_str: str) -> str:
    """
    Stand-in 'teacher' that *mimics* an LLM by returning a JSON blob
    with 'action_logits'.
    
    handles any number of candidates (9, 12, etc.) and uses
    deformation-driven heuristics including interior points.

    in theory, shd be able to be replaced by any LLM call that:
      1) takes ctx_json_str as input,
      2) returns JSON with 'action_logits': [float,...].
    """
    ctx = json.loads(ctx_json_str)
    num_anchors = len(ctx["candidate_nodes"])
    top = ctx["top_nodes_by_deflection"]
    candidate_types = ctx.get("candidate_types", [])

    # Initialize logits based on deformation magnitude
    logits = np.zeros(num_anchors, dtype=float)
    
    # Heuristic 1: High logits for anchors with large deflection
    for t in top:
        i = t["id"]
        if 0 <= i < num_anchors:
            logits[i] += t["mm"]

    # Heuristic 2: Slight boost for interior candidates that have high sag
    # (interior points are often harder to reach but very effective)
    for info in candidate_types:
        i = info["id"]
        if info["type"] == "interior" and info["current_sag"] > 0.5:
            logits[i] += 0.3  # modest boost for promising interior points

    # Heuristic 3: Small negative for boundary nodes already well-positioned
    for info in candidate_types:
        i = info["id"]
        if info["type"] == "boundary" and info["current_sag"] < 0.2:
            logits[i] -= 0.1  # slight penalty for boundary that's already good

    # Add small noise for stochasticity
    logits += 0.01 * np.random.randn(num_anchors)

    return json.dumps({"action_logits": logits.tolist()})


def parse_teacher_reply(raw_json_str: str, num_anchors: int) -> np.ndarray:
    """
    Parse teacher JSON & return a probability vector over anchors.
    expected schema:
      {"action_logits": [l_0, l_1, ..., l_{num_anchors-1}]}
    automatically pads or truncates if logits dimension doesn't match :,(
    """
    obj = json.loads(raw_json_str)
    logits = np.array(obj["action_logits"], dtype=float)

    if logits.shape[0] < num_anchors:
        # pad if shorter than expected
        logits = np.pad(logits, (0, num_anchors - logits.shape[0]), mode="constant")
    elif logits.shape[0] > num_anchors:
        logits = logits[:num_anchors]

    # softmax for probabilities
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-8
    return probs
