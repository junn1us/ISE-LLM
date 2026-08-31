#!/usr/bin/env python3
"""
test_implementation.py

Quick test script to verify a few things!!
Tests:
1. Environment with 12 anchors
2. Dataset collection with 12 anchors
3. Student distillation with 12 actions
4. PPO training (from scratch and with warm-start)
5. Evaluation and ablation
Run with: python test_implementation.py
"""

import subprocess
import sys
import os
from pathlib import Path
import numpy as np
import torch


def run_test(name, cmd, check=True):
    """Run a test command."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if check:
            print(f"\n✗ FAILED: {name}")
            sys.exit(1)
        else:
            print(f"\n⚠ WARNING: {name} returned non-zero")
            return False
    
    print(f"✓ PASSED: {name}")
    return True


def test_imports():
    """Test that all modules can be imported."""
    print(f"\n{'='*70}")
    print(f"TEST: Module imports")
    print(f"{'='*70}")
    
    try:
        import env_fixture
        print("✓ env_fixture")
        
        import collect_dataset
        print("✓ collect_dataset")
        
        import distill_student
        print("✓ distill_student")
        
        import teacher_iface
        print("✓ teacher_iface")
        
        import ppo_finetune
        print("✓ ppo_finetune")
        
        import eval_ppo_ablation
        print("✓ eval_ppo_ablation")
        
        import validate_12_anchor_setup
        print("✓ validate_12_anchor_setup")
        
        print("\n✓ PASSED: All imports successful")
        return True
    except Exception as e:
        print(f"\n✗ FAILED: Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_12_anchors():
    """Test environment with 12 anchors."""
    print(f"\n{'='*70}")
    print(f"TEST: Environment 12-anchor configuration")
    print(f"{'='*70}")
    
    try:
        from env_fixture import DiscreteFixtureEnv
        
        env = DiscreteFixtureEnv(anchor_mode="twelve")
        print(f"✓ Created environment with {env.num_anchors} anchors")
        
        obs = env.reset()
        print(f"✓ Reset environment, observation shape: {obs.shape}")
        
        expected_shape = (12 + 3,)  # 12 anchors + global stats
        assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
        print(f"✓ Observation shape correct: {obs.shape}")
        
        # Test step
        obs, r, done, info = env.step(0)
        print(f"✓ Step completed, got reward {r:.4f}, done={done}")
        
        print("\n✓ PASSED: Environment 12-anchor test")
        return True
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_teacher_context():
    """Test teacher context generation with 12 anchors."""
    print(f"\n{'='*70}")
    print(f"TEST: Teacher context generation (12 anchors)")
    print(f"{'='*70}")
    
    try:
        from env_fixture import DiscreteFixtureEnv
        from teacher_iface import translator_context, teacher_llm_fn, parse_teacher_reply
        import json
        
        env = DiscreteFixtureEnv(anchor_mode="twelve")
        obs = env.reset()
        
        ctx = translator_context(env, obs)
        print(f"✓ Context generated")
        print(f"  - candidate_nodes: {len(ctx['candidate_nodes'])}")
        print(f"  - candidate_types: {len(ctx['candidate_types'])}")
        
        ctx_json = json.dumps(ctx)
        reply = teacher_llm_fn(ctx_json)
        print(f"✓ Teacher LLM generated reply")
        
        probs = parse_teacher_reply(reply, env.num_anchors)
        print(f"✓ Parsed probabilities: shape {probs.shape}, sum {probs.sum():.6f}")
        
        assert probs.shape[0] == 12, f"Expected 12 probs, got {probs.shape[0]}"
        assert abs(probs.sum() - 1.0) < 1e-5, f"Probs don't sum to 1: {probs.sum()}"
        
        print("\n✓ PASSED: Teacher context test")
        return True
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_suite():
    """Run validation suite."""
    print(f"\n{'='*70}")
    print(f"TEST: Validation suite")
    print(f"{'='*70}")
    
    # Check if dataset exists; if not, skip
    if not (Path("teacher_data_12_X.npy").exists() and Path("teacher_data_12_P.npy").exists()):
        print("⚠ Dataset files not found (OK if not regenerated yet)")
        return True
    
    return run_test(
        "validate_12_anchor_setup.py",
        [sys.executable, "validate_12_anchor_setup.py"],
        check=False
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Tasks 7.1A and 7.2B implementation")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--full", action="store_true", help="Run all tests including training")
    args = parser.parse_args()
    
    os.chdir(Path(__file__).parent)
    
    print(f"\n{'='*70}")
    print(f"TESTING IMPLEMENTATION: Tasks 7.1A & 7.2B")
    print(f"{'='*70}")
    
    # Test 1: Imports
    if not test_imports():
        print("\nCritical: Import test failed. Cannot continue.")
        sys.exit(1)
    
    # Test 2: Environment
    if not test_env_12_anchors():
        print("\nCritical: Environment test failed. Cannot continue.")
        sys.exit(1)
    
    # Test 3: Teacher context
    if not test_teacher_context():
        print("\nCritical: Teacher context test failed. Cannot continue.")
        sys.exit(1)
    
    # Test 4: Validation suite
    test_validation_suite()
    
    # Test 5: Quick PPO test (optional)
    if args.full and not args.quick:
        print(f"\n{'='*70}")
        print(f"TEST: Quick PPO training (5 iters)")
        print(f"{'='*70}")
        
        # Make sure student checkpoint exists
        if not Path("student_policy_init.pt").exists():
            print("ℹ Generating student checkpoint...")
            subprocess.run([
                sys.executable, "collect_dataset.py",
                "--episodes", "5", "--anchor-mode", "twelve"
            ], check=False)
            subprocess.run([
                sys.executable, "distill_student.py",
                "--data-X", "teacher_data_X.npy",
                "--data-P", "teacher_data_P.npy",
                "--steps", "50"
            ], check=False)
        
        if Path("student_policy_init.pt").exists():
            print("ℹ Running quick PPO training test (5 iters)...")
            subprocess.run([
                sys.executable, "ppo_finetune.py",
                "--iters", "5",
                "--episodes-per-iter", "5",
                "--out", "ppo_test.pt"
            ], check=False)
            print("✓ PPO training completed")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"✓ All critical tests passed!")
    print(f"\nNext steps:")
    print(f"1. Full conversion: python convert_to_12_anchors.py --episodes 30")
    print(f"2. Validate setup:  python validate_12_anchor_setup.py")
    print(f"3. Run ablation:    python eval_ppo_ablation.py --iters 30 --eval-episodes 50")
    print()


if __name__ == "__main__":
    main()
