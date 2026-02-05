# eval_policy.py
# evaluation harness and teacher vs PPO comparison 

import argparse
import numpy as np
import torch
from pathlib import Path

from env_fixture import DiscreteFixtureEnv
from ppo_finetune import ActorCriticDiscrete
from teacher_iface import translator_context, teacher_llm_fn, parse_teacher_reply


def _set_episode_seed(env, seed: int):
    """
    Best-effort seeding
    Works if env exposes rng, np_rng, or seed().
    """
    if hasattr(env, "seed") and callable(getattr(env, "seed")):
        env.seed(seed)
        return
    if hasattr(env, "rng"):
        try:
            env.rng = np.random.default_rng(seed)
            return
        except Exception:
            pass
    # fallback: set global numpy RNG for anything using np.random
    np.random.seed(seed)


def _run_episode_with_policy(env, ac, deterministic: bool = False, render: bool = False, title: str = ""):
    obs = env.reset()
    done = False
    ep_rews = []
    info = {}
    while not done:
        with torch.no_grad():
            if deterministic:
                # greedy action: argmax logits
                obs_t = torch.tensor(obs, dtype=torch.float32)
                logits = ac.policy_logits(obs_t)
                a_id = int(torch.argmax(logits).item())
            else:
                a_id, _, _, _ = ac.act(obs)

        env_action = a_id % env.num_anchors
        obs, r, done, info = env.step(env_action)
        ep_rews.append(r)

    if render:
        env.render(title=title)

    rmse = float(info.get("rmse", np.nan))
    ret = float(np.sum(ep_rews))
    return rmse, ret


def _run_episode_with_teacher(env, mode: str = "greedy", render: bool = False, title: str = ""):
    """
    mode:
      - "greedy": pick argmax over teacher probs
      - "sample": sample from teacher probs
    """
    obs = env.reset()
    done = False
    ep_rews = []
    info = {}
    while not done:
        ctx = translator_context(env, obs)
        raw = teacher_llm_fn(__import__("json").dumps(ctx))
        probs = parse_teacher_reply(raw, env.num_anchors)

        if mode == "greedy":
            env_action = int(np.argmax(probs))
        elif mode == "sample":
            env_action = int(np.random.choice(np.arange(env.num_anchors), p=probs))
        else:
            raise ValueError("mode must be 'greedy' or 'sample'")

        obs, r, done, info = env.step(env_action)
        ep_rews.append(r)

    if render:
        env.render(title=title)

    rmse = float(info.get("rmse", np.nan))
    ret = float(np.sum(ep_rews))
    return rmse, ret


def load_ppo(policy_path: str):
    here = Path(__file__).resolve().parent
    ckpt_path = Path(policy_path)
    if not ckpt_path.is_absolute():
        ckpt_path = here / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found at: {ckpt_path}")

    env = DiscreteFixtureEnv()
    obs_dim = env.reset().shape[0]

    state = torch.load(ckpt_path, map_location="cpu")

    act_dim = None
    for k, v in state.items():
        if "pi_net.4.bias" in k:
            act_dim = v.shape[0]
            break
    if act_dim is None:
        raise RuntimeError("Could not infer act_dim from PPO state dict.")

    ac = ActorCriticDiscrete(obs_dim, act_dim, hidden=64)
    ac.load_state_dict(state)
    ac.eval()
    return ckpt_path, ac


def eval_all(policy_path: str, num_episodes: int = 50, render: bool = False, seed0: int = 123):
    ckpt_path, ac = load_ppo(policy_path)

    # Pre-generate per-episode seeds so PPO/teacher see the same randomness
    episode_seeds = [seed0 + i for i in range(num_episodes)]

    results = {
        "ppo": {"rmse": [], "ret": []},
        "teacher_greedy": {"rmse": [], "ret": []},
        "teacher_sample": {"rmse": [], "ret": []},
    }

    for ep, s in enumerate(episode_seeds, start=1):
        # PPO
        env = DiscreteFixtureEnv()
        _set_episode_seed(env, s)
        rmse, ret = _run_episode_with_policy(env, ac, deterministic=False, render=render, title=f"PPO ep {ep}")
        results["ppo"]["rmse"].append(rmse)
        results["ppo"]["ret"].append(ret)

        # Teacher greedy
        env = DiscreteFixtureEnv()
        _set_episode_seed(env, s)
        rmse, ret = _run_episode_with_teacher(env, mode="greedy", render=render, title=f"Teacher(greedy) ep {ep}")
        results["teacher_greedy"]["rmse"].append(rmse)
        results["teacher_greedy"]["ret"].append(ret)

        # Teacher sample
        env = DiscreteFixtureEnv()
        _set_episode_seed(env, s)
        rmse, ret = _run_episode_with_teacher(env, mode="sample", render=render, title=f"Teacher(sample) ep {ep}")
        results["teacher_sample"]["rmse"].append(rmse)
        results["teacher_sample"]["ret"].append(ret)

    def summar(name):
        rmse = np.array(results[name]["rmse"], dtype=float)
        ret = np.array(results[name]["ret"], dtype=float)
        return (
            float(np.nanmean(rmse)), float(np.nanstd(rmse)),
            float(np.mean(ret)), float(np.std(ret))
        )

    ppo_m, ppo_s, ppo_ret_m, ppo_ret_s = summar("ppo")
    tg_m, tg_s, tg_ret_m, tg_ret_s = summar("teacher_greedy")
    ts_m, ts_s, ts_ret_m, ts_ret_s = summar("teacher_sample")

    print(f"Evaluated PPO at: {ckpt_path}")
    print(f"PPO:            mean rmse = {ppo_m:.6f} | std rmse = {ppo_s:.6f} | mean return = {ppo_ret_m:.6f} | std return = {ppo_ret_s:.6f}")
    print(f"Teacher(greedy): mean rmse = {tg_m:.6f} | std rmse = {tg_s:.6f} | mean return = {tg_ret_m:.6f} | std return = {tg_ret_s:.6f}")
    print(f"Teacher(sample): mean rmse = {ts_m:.6f} | std rmse = {ts_s:.6f} | mean return = {ts_ret_m:.6f} | std return = {ts_ret_s:.6f}")

    # Paired deltas: PPO - Teacher
    d_rmse_g = np.array(results["ppo"]["rmse"]) - np.array(results["teacher_greedy"]["rmse"])
    d_ret_g  = np.array(results["ppo"]["ret"])  - np.array(results["teacher_greedy"]["ret"])
    print(f"\nPaired delta (PPO - Teacher greedy): mean Δrmse = {float(np.nanmean(d_rmse_g)):.6f}, mean Δreturn = {float(np.mean(d_ret_g)):.6f}")

    d_rmse_s = np.array(results["ppo"]["rmse"]) - np.array(results["teacher_sample"]["rmse"])
    d_ret_s  = np.array(results["ppo"]["ret"])  - np.array(results["teacher_sample"]["ret"])
    print(f"Paired delta (PPO - Teacher sample): mean Δrmse = {float(np.nanmean(d_rmse_s)):.6f}, mean Δreturn = {float(np.mean(d_ret_s)):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, default="ppo_fixture_policy.pt")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed0", type=int, default=123)
    args = parser.parse_args()
    eval_all(args.policy_path, args.num_episodes, args.render, args.seed0)
