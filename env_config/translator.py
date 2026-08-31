import json

def translator_context(obs_vec, info, anchor_xy, max_steps, step_idx):
    """
    obs_vec: env observation (mask || feat)
    info:    env info dict from .reset() or .step()
    anchor_xy: (A,2) array of candidate fixture coords
    """

    A = anchor_xy.shape[0]
    mask = obs_vec[:A]             # 0/1 fixture occupancy
    feat = obs_vec[A:]             # deflection summary

    # pick "top nodes by deflection" from feat or from info if you store per-node error
    # for now we'll stub:
    top_nodes = [{"id": int(i), "mm": float(val)}
                 for i,val in enumerate(feat[:4])]  # pretend first 4 are largest

    ctx = {
        "stage":        int(step_idx),
        "budget_left":  int(max_steps - step_idx),
        "d_max_mm":     float(np.max(feat)),        # worst residual deflection-ish
        "top_nodes_by_deflection": top_nodes,
        "candidate_nodes": list(range(A)),          # anchors we *could* toggle
        "disabled_neighbors": [],                   # can fill spacing mask later
        "spacing_mm":  10.0,                        # dummy spacing rule for now
        "ops_enabled": ["add","remove","noop"],
        "fixture_mask": mask.astype(int).tolist(),  # which anchors are active
        "goals": {"d_max_mm": "<=0.5"},
    }

    # critically: ask model to OUTPUT STRICT JSON ONLY
    prompt = (
        "You are an expert manufacturing planner. "
        "Given the sheet fixturing state below, choose which anchor index to "
        "toggle (add/remove fixture) next to reduce deflection.\n"
        "Return STRICT JSON ONLY with the keys:\n"
        "{\n"
        '  "action_logits": [p0, p1, ..., pA, p_noop],\n'
        '  "rationale": "<brief reason>"\n'
        "}\n"
        "where action_logits is an unnormalized preference over each discrete action.\n\n"
        f"STATE:\n{json.dumps(ctx)}"
    )
    return prompt
