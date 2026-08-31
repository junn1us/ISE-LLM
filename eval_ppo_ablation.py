#!/usr/bin/env python3
"""
eval_ppo_ablation.py
Compares:
1. PPO from random initialization (no student checkpoint)
2. PPO warm-started from distilled student checkpoint

Evaluates both over the same seeds/episodes and reports:
- mean±std RMSE for each
- mean±std return for each
- Paired deltas (PPO-teacher-init - PPO-random-init)
"""

import argparse
import json
import numpy as np
import torch
import copy
from pathlib import Path
from typing import Dict, Tuple, List

from env_fixture import DiscreteFixtureEnv
from ppo_finetune import ppo_finetune, ActorCriticDiscrete
from distill_student import StudentPolicyDiscrete


def set_episode_seed(env, seed: int):
    """Set environment seed for reproducibility."""
    if hasattr(env, "rng"):
        try:
            env.rng = np.random.default_rng(seed)
            return
        except Exception:
            pass
    np.random.seed(seed)


def run_episode_with_policy(
    env: DiscreteFixtureEnv,
    ac: ActorCriticDiscrete,
    deterministic: bool = False,
) -> Tuple[float, float]:
    """
    Run one episode with policy.
    Returns: (rmse_final, return_sum)
    """
    obs = env.reset()
    done = False
    ep_rews = []
    info = {}
    
    while not done:
        with torch.no_grad():
            if deterministic:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                logits = ac.policy_logits(obs_t)
                a_id = int(torch.argmax(logits).item())
            else:
                a_id, _, _, _ = ac.act(obs)
        
        env_action = a_id % env.num_anchors
        obs, r, done, info = env.step(env_action)
        ep_rews.append(r)
    
    rmse = float(info.get("rmse", np.nan))
    ret = float(np.sum(ep_rews))
    return rmse, ret


def train_ppo_from_scratch(
    num_iters: int = 50,
    episodes_per_iter: int = 20,
    seed: int = 13,
    out_name: str = "ppo_scratch.pt",
) -> ActorCriticDiscrete:
    """
    Train PPO from random initialization (no warm-start).
    """
    print(f"\n{'='*70}")
    print(f"Training PPO from SCRATCH (random init)...")
    print(f"{'='*70}")
    
    ac = ppo_finetune(
        num_iters=num_iters,
        episodes_per_iter=episodes_per_iter,
        out_name=out_name,
        seed=seed,
        force_scratch=True,  # Force random initialization
    )
    return ac


def train_ppo_with_distilled_init(
    num_iters: int = 50,
    episodes_per_iter: int = 20,
    seed: int = 13,
    out_name: str = "ppo_teacher_init.pt",
) -> ActorCriticDiscrete:
    """
    Train PPO with teacher-distilled warm-start.
    This is the default behavior of ppo_finetune if student_policy_init.pt exists.
    """
    print(f"\n{'='*70}")
    print(f"Training PPO with TEACHER-DISTILLED INITIALIZATION...")
    print(f"{'='*70}")
    
    ac = ppo_finetune(
        num_iters=num_iters,
        episodes_per_iter=episodes_per_iter,
        out_name=out_name,
        seed=seed,
    )
    return ac


def evaluate_policy(
    ac: ActorCriticDiscrete,
    num_episodes: int = 50,
    seed0: int = 123,
) -> Dict[str, List[float]]:
    """
    Evaluate policy over num_episodes with fixed seeds.
    Returns dict with 'rmse' and 'ret' lists.
    """
    results = {"rmse": [], "ret": []}
    
    for ep in range(num_episodes):
        env = DiscreteFixtureEnv()
        set_episode_seed(env, seed0 + ep)
        
        rmse, ret = run_episode_with_policy(env, ac, deterministic=False)
        results["rmse"].append(rmse)
        results["ret"].append(ret)
    
    return results


def print_summary(name: str, results: Dict[str, List[float]]):
    """Print mean±std summary for policy."""
    rmse_arr = np.array(results["rmse"], dtype=float)
    ret_arr = np.array(results["ret"], dtype=float)
    
    rmse_mean = float(np.nanmean(rmse_arr))
    rmse_std = float(np.nanstd(rmse_arr))
    ret_mean = float(np.mean(ret_arr))
    ret_std = float(np.std(ret_arr))
    
    print(f"\n{name}:")
    print(f"  RMSE:   {rmse_mean:.6f} ± {rmse_std:.6f}")
    print(f"  Return: {ret_mean:.6f} ± {ret_std:.6f}")
    
    return rmse_mean, rmse_std, ret_mean, ret_std


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: PPO-from-scratch vs PPO-teacher-initialized"
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="PPO iterations (default 50)"
    )
    parser.add_argument(
        "--episodes-per-iter",
        type=int,
        default=20,
        help="Episodes collected per PPO iteration (default 20)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Episodes for final evaluation (default 50)",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=13,
        help="Seed for PPO training (default 13)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=123,
        help="Base seed for evaluation episodes (default 123)",
    )
    parser.add_argument(
        "--scratch-only",
        action="store_true",
        help="Only train PPO from scratch, skip warm-start variant",
    )
    parser.add_argument(
        "--no-scratch",
        action="store_true",
        help="Skip training from scratch, only train warm-start variant",
    )
    args = parser.parse_args()

    # =========================================================================
    # TRAIN: PPO FROM SCRATCH
    # =========================================================================
    results_scratch = None
    ac_scratch = None
    if not args.no_scratch:
        ac_scratch = train_ppo_from_scratch(
            num_iters=args.iters,
            episodes_per_iter=args.episodes_per_iter,
            seed=args.train_seed,
            out_name="ppo_scratch.pt",
        )

        # Evaluate PPO from scratch
        print(f"\n{'='*70}")
        print(f"Evaluating PPO-from-scratch over {args.eval_episodes} episodes...")
        print(f"{'='*70}")
        results_scratch = evaluate_policy(
            ac_scratch, num_episodes=args.eval_episodes, seed0=args.eval_seed
        )

    # =========================================================================
    # TRAIN: PPO WITH TEACHER-DISTILLED INITIALIZATION
    # =========================================================================
    results_teacher_init = None
    ac_teacher_init = None
    if not args.scratch_only:
        ac_teacher_init = train_ppo_with_distilled_init(
            num_iters=args.iters,
            episodes_per_iter=args.episodes_per_iter,
            seed=args.train_seed,
            out_name="ppo_teacher_init.pt",
        )

        # Evaluate PPO with teacher init
        print(f"\n{'='*70}")
        print(f"Evaluating PPO-teacher-init over {args.eval_episodes} episodes...")
        print(f"{'='*70}")
        results_teacher_init = evaluate_policy(
            ac_teacher_init, num_episodes=args.eval_episodes, seed0=args.eval_seed
        )

    # =========================================================================
    # REPORT RESULTS
    # =========================================================================
    print(f"\n\n{'='*70}")
    print(f"ABLATION STUDY SUMMARY")
    print(f"{'='*70}")

    scratch_rmse_m, scratch_rmse_s = np.nan, np.nan
    scratch_ret_m, scratch_ret_s = np.nan, np.nan
    if results_scratch is not None:
        scratch_rmse_m, scratch_rmse_s, scratch_ret_m, scratch_ret_s = print_summary(
            "PPO-from-scratch", results_scratch
        )

    teacher_rmse_m, teacher_rmse_s = np.nan, np.nan
    teacher_ret_m, teacher_ret_s = np.nan, np.nan
    if results_teacher_init is not None:
        teacher_rmse_m, teacher_rmse_s, teacher_ret_m, teacher_ret_s = print_summary(
            "PPO-teacher-init", results_teacher_init
        )

    # Paired deltas
    if results_scratch is not None and results_teacher_init is not None:
        print(f"\n{'='*70}")
        print(f"PAIRED DELTAS (PPO-teacher-init - PPO-scratch)")
        print(f"{'='*70}")
        
        rmse_scratch = np.array(results_scratch["rmse"], dtype=float)
        rmse_teacher = np.array(results_teacher_init["rmse"], dtype=float)
        ret_scratch = np.array(results_scratch["ret"], dtype=float)
        ret_teacher = np.array(results_teacher_init["ret"], dtype=float)
        
        delta_rmse = rmse_teacher - rmse_scratch
        delta_ret = ret_teacher - ret_scratch
        
        print(f"RMSE improvement (teacher-init - scratch):")
        print(f"  Mean Δ: {float(np.mean(delta_rmse)):+.6f}")
        print(f"  Std Δ:  {float(np.std(delta_rmse)):.6f}")
        print(f"  (negative = teacher-init is better)")
        
        print(f"\nReturn improvement (teacher-init - scratch):")
        print(f"  Mean Δ: {float(np.mean(delta_ret)):+.6f}")
        print(f"  Std Δ:  {float(np.std(delta_ret)):.6f}")
        print(f"  (positive = teacher-init is better)")
        
        # Save results to JSON for downstream analysis
        ablation_results = {
            "ppo_scratch": {
                "rmse_mean": float(scratch_rmse_m),
                "rmse_std": float(scratch_rmse_s),
                "return_mean": float(scratch_ret_m),
                "return_std": float(scratch_ret_s),
            },
            "ppo_teacher_init": {
                "rmse_mean": float(teacher_rmse_m),
                "rmse_std": float(teacher_rmse_s),
                "return_mean": float(teacher_ret_m),
                "return_std": float(teacher_ret_s),
            },
            "paired_deltas": {
                "rmse_mean_delta": float(np.mean(delta_rmse)),
                "rmse_std_delta": float(np.std(delta_rmse)),
                "return_mean_delta": float(np.mean(delta_ret)),
                "return_std_delta": float(np.std(delta_ret)),
            }
        }
        
        with open("ablation_results.json", "w") as f:
            json.dump(ablation_results, f, indent=2)
        print(f"\nResults saved to ablation_results.json")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
