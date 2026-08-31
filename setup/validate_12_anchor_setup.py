#!/usr/bin/env python3
"""
validate_12_anchor_setup.py

Validation script to verify the 12-anchor setup is working correctly.
"""

import numpy as np
import torch
from pathlib import Path
from env_fixture import DiscreteFixtureEnv
from distill_student import StudentPolicyDiscrete
from teacher_iface import translator_context, teacher_llm_fn, parse_teacher_reply


def validate_env_12_anchors():
    """Verify environment creates 12 anchors correctly."""
    print("\n[TEST 1] Environment configuration (12 anchors)...")
    env = DiscreteFixtureEnv(anchor_mode="twelve")
    
    assert env.num_anchors == 12, f"Expected 12 anchors, got {env.num_anchors}"
    print(f"✓ num_anchors = {env.num_anchors}")
    
    # Check anchor positions
    expected_anchors = {
        (0, 0), (8, 0), (0, 8), (8, 8),  # corners
        (4, 0), (4, 8), (0, 4), (8, 4),  # edge midpoints
        (2, 2), (6, 2), (2, 6), (6, 6),  # interior
    }
    actual_anchors = set(tuple(map(int, xy)) for xy in env.anchor_xy)
    assert actual_anchors == expected_anchors, f"Anchor mismatch: {actual_anchors} vs {expected_anchors}"
    print(f"✓ Anchor positions correct: {sorted(actual_anchors)}")
    
    return env


def validate_observation_shape(env):
    """Verify observation vector has correct shape."""
    print("\n[TEST 2] Observation shape...")
    obs = env.reset()
    expected_dim = env.num_anchors + 3  # anchors + max/mean/std
    assert obs.shape[0] == expected_dim, f"Expected obs dim {expected_dim}, got {obs.shape[0]}"
    print(f"✓ Observation shape: {obs.shape} (num_anchors={env.num_anchors}, global_stats=3)")
    
    return obs


def validate_action_space(env):
    """Verify action space matches 12 anchors."""
    print("\n[TEST 3] Action space...")
    env.reset()
    
    # With include_noop=True, action_dim = num_anchors + 1
    expected_action_dim = env.num_anchors + 1  # includes noop action
    assert env.action_dim == expected_action_dim, f"Expected action_dim {expected_action_dim}, got {env.action_dim}"
    print(f"✓ Action space: {env.action_dim} actions (12 anchors + 1 noop)")
    
    # Test a few step operations
    for action_id in [0, 1, 6, 12]:
        obs, reward, done, info = env.step(action_id)
        assert obs.shape[0] == env.num_anchors + 3, f"Step produced wrong obs shape: {obs.shape}"
    print(f"✓ Step operations work correctly")


def validate_teacher_context(env, obs):
    """Verify teacher context generation."""
    print("\n[TEST 4] Teacher context (translator_context)...")
    ctx = translator_context(env, obs)
    
    assert len(ctx["candidate_nodes"]) == 12, f"Expected 12 candidates, got {len(ctx['candidate_nodes'])}"
    print(f"✓ candidate_nodes: {len(ctx['candidate_nodes'])}")
    
    assert len(ctx["candidate_types"]) == 12, f"Expected 12 candidate_types, got {len(ctx['candidate_types'])}"
    print(f"✓ candidate_types: {len(ctx['candidate_types'])}")
    
    # Verify interior/boundary distinction
    interior_count = sum(1 for ct in ctx["candidate_types"] if ct["type"] == "interior")
    boundary_count = sum(1 for ct in ctx["candidate_types"] if ct["type"] == "boundary")
    assert interior_count == 4, f"Expected 4 interior, got {interior_count}"
    assert boundary_count == 8, f"Expected 8 boundary, got {boundary_count}"
    print(f"✓ Interior/boundary split: {interior_count} interior, {boundary_count} boundary")


def validate_teacher_llm(env, obs):
    """Verify teacher LLM handles 12 actions."""
    print("\n[TEST 5] Teacher LLM output...")
    import json
    ctx = translator_context(env, obs)
    ctx_json = json.dumps(ctx)
    
    reply = teacher_llm_fn(ctx_json)
    reply_obj = json.loads(reply)
    
    assert "action_logits" in reply_obj, "Missing 'action_logits' in reply"
    logits = np.array(reply_obj["action_logits"], dtype=float)
    assert len(logits) == 12, f"Expected 12 logits, got {len(logits)}"
    print(f"✓ action_logits shape: ({len(logits)},)")


def validate_teacher_probs(env, obs):
    """Verify teacher probability parsing."""
    print("\n[TEST 6] Teacher probability parsing...")
    import json
    ctx = translator_context(env, obs)
    ctx_json = json.dumps(ctx)
    
    reply = teacher_llm_fn(ctx_json)
    probs = parse_teacher_reply(reply, env.num_anchors)
    
    assert probs.shape[0] == 12, f"Expected 12 probs, got {probs.shape[0]}"
    assert abs(probs.sum() - 1.0) < 1e-6, f"Probs don't sum to 1: {probs.sum()}"
    print(f"✓ Probabilities shape: {probs.shape}, sum: {probs.sum():.6f}")


def validate_dataset_compatibility():
    """Verify saved dataset is compatible."""
    print("\n[TEST 7] Saved dataset compatibility...")
    
    X_path = Path("teacher_data_12_X.npy")
    P_path = Path("teacher_data_12_P.npy")
    
    if not X_path.exists() or not P_path.exists():
        print(f"⚠ Dataset files not found (OK if not regenerated yet)")
        return
    
    X = np.load(X_path)
    P = np.load(P_path)
    
    # Expected: X has 15 features (12 anchors + 3 global), P has 12 actions
    expected_feat_dim = 12 + 3
    assert X.shape[1] == expected_feat_dim, f"Expected X dim {expected_feat_dim}, got {X.shape[1]}"
    assert P.shape[1] == 12, f"Expected P dim 12, got {P.shape[1]}"
    print(f"✓ Dataset shapes: X {X.shape}, P {P.shape}")
    
    # Verify probabilities sum to 1
    p_sums = P.sum(axis=1)
    assert np.allclose(p_sums, 1.0, atol=1e-5), f"Some probability rows don't sum to 1"
    print(f"✓ Probabilities normalized (verified across {len(P)} samples)")


def validate_student_checkpoint():
    """Verify distilled student checkpoint."""
    print("\n[TEST 8] Student checkpoint compatibility...")
    
    ckpt_path = Path("student_policy_init.pt")
    if not ckpt_path.exists():
        print(f"⚠ Checkpoint not found (OK if distillation not run yet)")
        return
    
    # Load checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    
    # Infer action dimension from output layer
    output_bias = None
    for k, v in state.items():
        if "net.4.bias" in k:
            output_bias = v
            break
    
    assert output_bias is not None, "Could not find output layer in checkpoint"
    act_dim = output_bias.shape[0]
    assert act_dim == 12, f"Expected 12 actions, got {act_dim}"
    print(f"✓ Checkpoint output dimension: {act_dim}")
    
    # Try loading into StudentPolicyDiscrete
    student = StudentPolicyDiscrete(obs_dim=15, n_actions=12)
    student.load_state_dict(state)
    print(f"✓ Successfully loaded checkpoint into StudentPolicyDiscrete")


def main():
    print("=" * 70)
    print("VALIDATION: 12-Anchor Setup")
    print("=" * 70)
    
    try:
        env = validate_env_12_anchors()
        obs = validate_observation_shape(env)
        validate_action_space(env)
        validate_teacher_context(env, obs)
        validate_teacher_llm(env, obs)
        validate_teacher_probs(env, obs)
        validate_dataset_compatibility()
        validate_student_checkpoint()
        
        print("\n" + "=" * 70)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("=" * 70)
        print("\nThe 12-anchor setup is ready for use!")
        print("You can now:")
        print("  - Train PPO with: python ppo_finetune.py")
        print("  - Run ablation study with: python eval_ppo_ablation.py")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ VALIDATION FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
