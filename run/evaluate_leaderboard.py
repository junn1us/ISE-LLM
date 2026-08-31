import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ppo_finetune import ActorCriticDiscrete
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv


def evaluate_checkpoint(
    ckpt: str,
    episodes: int,
    eval_seed: int,
    anchor_mode: str,
    anchor_stride: int,
    grid_size: int,
    max_fixtures: int,
):
    core = FixtureCore(
        seed=eval_seed,
        anchor_mode=anchor_mode,
        anchor_stride=anchor_stride,
        grid_size=grid_size,
        max_fixtures=max_fixtures,
    )
    env = SmartFixtureStudentEnv(core)
    obs0 = env.reset()

    model = ActorCriticDiscrete(obs_dim=obs0.shape[0], act_dim=env.action_space.n, hidden=64)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    returns, rmses = [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        last_rmse = np.nan

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits = model.policy_logits(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())
            obs, r, done, info = env.step(action)
            ep_ret += float(r)
            last_rmse = float(info.get("rmse", np.nan))

        returns.append(ep_ret)
        rmses.append(last_rmse)

    returns = np.asarray(returns, dtype=float)
    rmses = np.asarray(rmses, dtype=float)

    z = 1.96
    n = float(episodes)

    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=1))
    se_ret = std_ret / math.sqrt(n)

    mean_rmse = float(rmses.mean())
    std_rmse = float(rmses.std(ddof=1))
    se_rmse = std_rmse / math.sqrt(n)

    return {
        "checkpoint": ckpt,
        "episodes": episodes,
        "mean_return": mean_ret,
        "return_std": std_ret,
        "return_ci95_low": mean_ret - z * se_ret,
        "return_ci95_high": mean_ret + z * se_ret,
        "mean_final_rmse": mean_rmse,
        "rmse_std": std_rmse,
        "rmse_ci95_low": mean_rmse - z * se_rmse,
        "rmse_ci95_high": mean_rmse + z * se_rmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints and produce leaderboard with 95% CIs.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths")
    parser.add_argument("--episodes", type=int, default=300, help="Evaluation episodes per checkpoint")
    parser.add_argument("--eval-seed", type=int, default=16, help="Environment seed for evaluation")
    parser.add_argument("--anchor-mode", type=str, default="grid100", choices=["grid100", "stride", "twelve", "boundary", "all"], help="Candidate fixture design space")
    parser.add_argument("--anchor-stride", type=int, default=4, help="Stride when anchor-mode='stride'")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size for the sheet")
    parser.add_argument("--max-fixtures", type=int, default=-1, help="Max active fixtures; <=0 means unconstrained")
    parser.add_argument("--out-csv", type=str, default="leaderboard_100iter_seeds.csv")
    args = parser.parse_args()

    rows = []
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f"[skip] missing checkpoint: {ckpt}")
            continue
        rows.append(
            evaluate_checkpoint(
                ckpt,
                args.episodes,
                args.eval_seed,
                args.anchor_mode,
                args.anchor_stride,
                args.grid_size,
                args.max_fixtures,
            )
        )

    if not rows:
        raise RuntimeError("No valid checkpoints were evaluated.")

    df = pd.DataFrame(rows).sort_values(
        by=["mean_return", "mean_final_rmse"],
        ascending=[False, True],
    ).reset_index(drop=True)

    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved leaderboard: {args.out_csv}")


if __name__ == "__main__":
    main()
