#!/usr/bin/env python
"""
Full teacher-student pipeline with gpt-oss-20b as the LLM teacher.
 - Collects teacher data using gpt-oss-20b (locally, on MPS)
 - Distills into student policy
 - PPO fine-tunes for 1000 iterations
 - Generates: deformation visualization, sample LLM prompts/outputs, metrics CSV
 - Produces a LaTeX-ready summary report

Usage:
    python scripts/run_gptoss20b_1000iter.py [--iters 1000] [--seed 42]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- project imports ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv
from smartfixture_llm4teach.envs.teacher_env import SmartFixtureTeacherEnv
from smartfixture_llm4teach.teacher.llm_adapter import build_gptoss_llm_fn
from distill_student import StudentPolicyDiscrete, distill_train
from ppo_finetune import ActorCriticDiscrete, ppo_update, gae_advantage


def parse_args():
    p = argparse.ArgumentParser(description="Full gpt-oss-20b teacher-student pipeline")
    p.add_argument("--checkpoint-dir", type=str,
                    default=str(ROOT / "gpt-oss-20b" / "original"),
                    help="Path to gpt-oss-20b original/ directory")
    p.add_argument("--iters", type=int, default=1000, help="PPO iterations")
    p.add_argument("--episodes-per-iter", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-fixtures", type=int, default=5)
    p.add_argument("--teacher-samples", type=int, default=32,
                    help="Number of state-action samples to collect from LLM teacher")
    p.add_argument("--teacher-temperature", type=float, default=0.7)
    p.add_argument("--llm-temperature", type=float, default=0.0,
                    help="gpt-oss-20b sampling temperature (0=greedy)")
    p.add_argument("--llm-max-tokens", type=int, default=30)
    p.add_argument("--distill-steps", type=int, default=500)
    p.add_argument("--distill-kl-weight", type=float, default=1.0)
    # PPO hypers
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    # Output
    p.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "gptoss20b_run"))
    return p.parse_args()


def collect_teacher_data(student_env, teacher_env, n_samples, log_dir):
    """Collect (obs, teacher_prob) pairs. Returns X, P numpy arrays."""
    X, P = [], []
    obs = student_env.reset().astype(np.float32)
    for i in range(n_samples):
        probs = teacher_env.get_teacher_distribution()
        X.append(obs.flatten())
        P.append(probs.copy())
        action_id = int(np.random.choice(len(probs), p=probs))
        obs_next, _, done, _ = student_env.step(action_id)
        if done:
            obs = student_env.reset().astype(np.float32)
        else:
            obs = obs_next.astype(np.float32)
        if (i + 1) % 50 == 0:
            print(f"  [teacher data] {i+1}/{n_samples} samples collected")
    X = np.stack(X, 0).astype(np.float32)
    P = np.stack(P, 0).astype(np.float32)
    return X, P


def distill_student_policy(X_np, P_np, obs_dim, act_dim, steps=500, kl_w=1.0):
    """Distill teacher probs into student network."""
    X_t = torch.tensor(X_np, dtype=torch.float32)
    P_t = torch.tensor(P_np, dtype=torch.float32)
    model = StudentPolicyDiscrete(obs_dim, act_dim, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for it in range(steps):
        probs, logits = model(X_t)
        kl = torch.sum(P_t * (torch.log(P_t + 1e-8) - torch.log(probs + 1e-8)), dim=1).mean()
        reg = sum((p ** 2).sum() for p in model.parameters())
        loss = kl_w * kl + 1e-4 * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 100 == 0:
            print(f"  [distill] step {it+1}/{steps}  kl={kl.item():.4f}  loss={loss.item():.4f}")
    return model


def run_ppo(ac, env, num_iters, episodes_per_iter, out_dir, **ppo_kwargs):
    """PPO training loop. Returns metrics history."""
    lr = ppo_kwargs.get("lr", 3e-4)
    clip = ppo_kwargs.get("clip", 0.2)
    ent_coef = ppo_kwargs.get("ent_coef", 0.01)
    vf_coef = ppo_kwargs.get("vf_coef", 0.5)
    epochs = ppo_kwargs.get("epochs", 4)
    minibatch = ppo_kwargs.get("minibatch", 64)
    gamma = ppo_kwargs.get("gamma", 0.99)
    lam = ppo_kwargs.get("lam", 0.95)

    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    history = {"iter": [], "mean_return": [], "mean_rmse": [], "std_rmse": []}

    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for it in range(1, num_iters + 1):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        ep_returns, ep_rmses = [], []

        for ep in range(episodes_per_iter):
            obs = env.reset()
            done = False
            ep_rews = []
            last_info = {}
            while not done:
                a_id, logp, v, _ = ac.act(obs)
                next_obs, r, done, info = env.step(a_id)
                last_info = info
                obs_buf.append(obs.copy())
                act_buf.append(a_id)
                logp_buf.append(logp)
                rew_buf.append(float(r))
                val_buf.append(float(v))
                done_buf.append(bool(done))
                ep_rews.append(float(r))
                obs = next_obs
            ep_returns.append(float(np.sum(ep_rews)))
            ep_rmses.append(float(last_info.get("rmse", np.nan)))

        batch = {
            "obs_buf": np.array(obs_buf, dtype=np.float32),
            "act_buf": np.array(act_buf, dtype=np.int64),
            "logp_buf": np.array(logp_buf, dtype=np.float32),
            "rew_buf": np.array(rew_buf, dtype=np.float32),
            "val_buf": np.array(val_buf, dtype=np.float32),
            "done_buf": done_buf,
        }
        if len(obs_buf) > 0:
            ppo_update(ac, opt, batch, clip=clip, vf_coef=vf_coef, ent_coef=ent_coef,
                       epochs=epochs, minibatch=minibatch, max_grad_norm=0.5,
                       gamma=gamma, lam=lam)

        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_rmse = float(np.nanmean(ep_rmses)) if ep_rmses else float("nan")
        std_rmse = float(np.nanstd(ep_rmses)) if ep_rmses else float("nan")

        history["iter"].append(it)
        history["mean_return"].append(mean_ret)
        history["mean_rmse"].append(mean_rmse)
        history["std_rmse"].append(std_rmse)

        if it % 10 == 0 or it == 1 or it == num_iters:
            print(f"[PPO] Iter {it:04d}/{num_iters} | ret {mean_ret:+.4f} | rmse {mean_rmse:.4f}±{std_rmse:.4f}")

        if it % 50 == 0 or it == num_iters:
            snap = ckpt_dir / f"policy_iter{it:04d}.pt"
            torch.save(ac.state_dict(), snap)

    return history


def generate_deformation_figure(env, ac, out_path, title="Final Deformation"):
    """Run one episode and plot the deformation field + fixture locations."""
    obs = env.reset()
    done = False
    while not done:
        a_id, _, _, _ = ac.act(obs)
        obs, r, done, info = env.step(a_id)

    core = env.core
    d = core.displacement.copy()
    n = core.grid_size
    grid = d.reshape(n, n) if d.size == n * n else d.reshape(int(np.sqrt(d.size)), -1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(grid, cmap="RdBu_r", origin="lower")
    plt.colorbar(im, ax=ax, label="Displacement (mm)")

    # overlay fixture locations
    if hasattr(core, "mask") and hasattr(core, "anchor_xy"):
        active = np.where(core.mask > 0)[0]
        for idx in active:
            xy = core.anchor_xy[idx]
            ax.plot(xy[0] * (n - 1), xy[1] * (n - 1), "g^", markersize=10, markeredgecolor="k")

    rmse_val = info.get("rmse", np.nan)
    ax.set_title(f"{title}\nRMSE = {rmse_val:.4f}, Fixtures = {int(core.mask.sum())}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved deformation figure: {out_path}")
    return rmse_val


def generate_training_curves(history, out_path):
    """Plot return and RMSE over PPO iterations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    iters = history["iter"]
    ax1.plot(iters, history["mean_return"], color="steelblue", lw=0.8)
    ax1.set_ylabel("Mean Return")
    ax1.set_title("PPO Training Curves (gpt-oss-20b Teacher)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, history["mean_rmse"], color="firebrick", lw=0.8)
    ax2.fill_between(iters,
                     np.array(history["mean_rmse"]) - np.array(history["std_rmse"]),
                     np.array(history["mean_rmse"]) + np.array(history["std_rmse"]),
                     color="firebrick", alpha=0.15)
    ax2.set_ylabel("RMSE")
    ax2.set_xlabel("PPO Iteration")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved training curves: {out_path}")


def generate_llm_prompt_output_report(log_path, out_dir, n_samples=5):
    """Extract sample prompts and outputs from the JSONL log and write a readable report."""
    if not Path(log_path).exists():
        print(f"  WARNING: LLM log not found at {log_path}")
        return

    entries = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        print("  WARNING: LLM log is empty")
        return

    # Select evenly spaced samples
    indices = np.linspace(0, len(entries) - 1, min(n_samples, len(entries)), dtype=int)
    samples = [entries[i] for i in indices]

    # Write text report
    txt_path = Path(out_dir) / "llm_prompt_output_samples.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  GPT-OSS-20B TEACHER: SAMPLE PROMPTS & OUTPUTS\n")
        f.write("=" * 80 + "\n\n")
        for s in samples:
            f.write(f"--- Call #{s['call_id']} ---\n")
            f.write(f"Prompt tokens: {s['n_prompt_tokens']}  |  Output tokens: {s['n_output_tokens']}  |  Time: {s['elapsed_s']}s\n\n")
            f.write("PROMPT (first 2000 chars):\n")
            f.write(s["prompt"][:2000] + "\n\n")
            f.write("RAW LLM OUTPUT:\n")
            f.write(s["raw_output"][:2000] + "\n")
            f.write("-" * 80 + "\n\n")

    # Write LaTeX-friendly version
    tex_path = Path(out_dir) / "llm_prompt_output_samples.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated LLM prompt/output samples\n")
        f.write("\\subsection{Sample LLM Teacher Prompts and Outputs}\n\n")
        for i, s in enumerate(samples):
            f.write(f"\\subsubsection{{Sample {i+1} (Call \\#{s['call_id']})}}\n")
            f.write(f"Prompt tokens: {s['n_prompt_tokens']}, "
                    f"Output tokens: {s['n_output_tokens']}, "
                    f"Latency: {s['elapsed_s']}s\n\n")
            f.write("\\textbf{Prompt (truncated):}\n")
            f.write("\\begin{lstlisting}[basicstyle=\\tiny\\ttfamily,breaklines=true]\n")
            prompt_clean = s["prompt"][:1500].replace("\\", "\\\\")
            f.write(prompt_clean + "\n")
            f.write("\\end{lstlisting}\n\n")
            f.write("\\textbf{LLM Output:}\n")
            f.write("\\begin{lstlisting}[basicstyle=\\tiny\\ttfamily,breaklines=true]\n")
            output_clean = s["raw_output"][:1500].replace("\\", "\\\\")
            f.write(output_clean + "\n")
            f.write("\\end{lstlisting}\n\n")

    # Summary statistics
    total_calls = len(entries)
    total_time = sum(e["elapsed_s"] for e in entries)
    avg_time = total_time / total_calls if total_calls > 0 else 0
    avg_out_tok = np.mean([e["n_output_tokens"] for e in entries])
    n_fallback = sum(1 for e in entries if '"action_logits": [0.0' in e.get("raw_output", ""))

    stats_path = Path(out_dir) / "llm_stats.json"
    stats = {
        "total_calls": total_calls,
        "total_time_s": round(total_time, 1),
        "avg_latency_s": round(avg_time, 2),
        "avg_output_tokens": round(float(avg_out_tok), 1),
        "fallback_count": n_fallback,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved LLM samples: {txt_path}")
    print(f"  Saved LLM LaTeX:   {tex_path}")
    print(f"  Saved LLM stats:   {stats_path}")
    print(f"  LLM summary: {total_calls} calls, avg {avg_time:.1f}s/call, {n_fallback} fallbacks")
    return stats


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = str(out_dir / "logs")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("  GPT-OSS-20B Teacher-Student Pipeline")
    print(f"  iters={args.iters}  seed={args.seed}  max_fixtures={args.max_fixtures}")
    print(f"  teacher_samples={args.teacher_samples}  distill_steps={args.distill_steps}")
    print(f"  checkpoint_dir={args.checkpoint_dir}")
    print(f"  output_dir={out_dir}")
    print("=" * 70)

    # ---- Step 1: Build LLM teacher function ----
    print("\n[Step 1] Loading gpt-oss-20b model...")
    t0 = time.time()
    llm_fn = build_gptoss_llm_fn(
        checkpoint_dir=args.checkpoint_dir,
        max_new_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        log_dir=log_dir,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ---- Step 2: Build environments ----
    print("\n[Step 2] Setting up environments...")
    core = FixtureCore(
        seed=args.seed,
        anchor_mode="grid100",
        grid_size=10,
        max_fixtures=args.max_fixtures,
    )
    student_env = SmartFixtureStudentEnv(core)
    teacher_env = SmartFixtureTeacherEnv(
        core, llm_fn=llm_fn, temperature=args.teacher_temperature
    )
    obs0 = student_env.reset()
    obs_dim = obs0.shape[0]
    act_dim = student_env.action_space.n
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    # ---- Step 3: Collect teacher data ----
    print(f"\n[Step 3] Collecting {args.teacher_samples} teacher samples...")
    t0 = time.time()
    X_np, P_np = collect_teacher_data(student_env, teacher_env, args.teacher_samples, log_dir)
    np.save(str(out_dir / "teacher_data_X.npy"), X_np)
    np.save(str(out_dir / "teacher_data_P.npy"), P_np)
    print(f"  Collected in {time.time() - t0:.1f}s  shapes X={X_np.shape} P={P_np.shape}")

    # ---- Step 4: Distill student ----
    print(f"\n[Step 4] Distilling student policy ({args.distill_steps} steps)...")
    student_model = distill_student_policy(
        X_np, P_np, obs_dim, act_dim,
        steps=args.distill_steps, kl_w=args.distill_kl_weight
    )
    student_ckpt = out_dir / "student_policy_init.pt"
    torch.save(student_model.state_dict(), student_ckpt)
    print(f"  Saved distilled student: {student_ckpt}")

    # ---- Step 5: Initialize PPO actor-critic from distilled student ----
    print("\n[Step 5] Initializing PPO actor-critic from distilled student...")
    ac = ActorCriticDiscrete(obs_dim, act_dim, hidden=64)
    try:
        ac.pi_net.load_state_dict(student_model.net.state_dict())
        print("  Loaded distilled policy into PPO actor network.")
    except Exception as e:
        print(f"  WARNING: Could not load distilled weights: {e}. Starting from random.")

    # ---- Step 6: PPO fine-tuning ----
    print(f"\n[Step 6] PPO fine-tuning for {args.iters} iterations...")
    t0 = time.time()
    history = run_ppo(
        ac, student_env, args.iters, args.episodes_per_iter, str(out_dir),
        lr=args.lr, clip=args.clip, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        epochs=args.epochs, minibatch=args.minibatch, gamma=args.gamma, lam=args.lam,
    )
    ppo_time = time.time() - t0
    print(f"  PPO done in {ppo_time:.1f}s ({ppo_time/60:.1f} min)")

    # Save final policy
    final_ckpt = out_dir / "ppo_final_policy.pt"
    torch.save(ac.state_dict(), final_ckpt)
    print(f"  Saved final policy: {final_ckpt}")

    # Save metrics history
    import csv
    csv_path = out_dir / "training_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "mean_return", "mean_rmse", "std_rmse"])
        w.writeheader()
        for i in range(len(history["iter"])):
            w.writerow({k: history[k][i] for k in w.fieldnames})
    print(f"  Saved metrics CSV: {csv_path}")

    # ---- Step 7: Generate figures ----
    print("\n[Step 7] Generating report figures...")

    # Deformation visualization
    deform_path = out_dir / "deformation_final.png"
    final_rmse = generate_deformation_figure(ac=ac, env=student_env, out_path=deform_path,
                                              title=f"gpt-oss-20b Teacher | seed={args.seed}")

    # Training curves
    curves_path = out_dir / "training_curves.png"
    generate_training_curves(history, curves_path)

    # ---- Step 8: LLM prompt/output report ----
    print("\n[Step 8] Generating LLM prompt/output report...")
    llm_log = Path(log_dir) / "llm_log.jsonl"
    llm_stats = generate_llm_prompt_output_report(str(llm_log), str(out_dir), n_samples=5)

    # ---- Final summary ----
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    final_mean_ret = history["mean_return"][-1] if history["mean_return"] else 0
    final_mean_rmse = history["mean_rmse"][-1] if history["mean_rmse"] else float("nan")
    print(f"  Final (iter {args.iters}): return={final_mean_ret:+.4f}  rmse={final_mean_rmse:.4f}")
    print(f"  Deformation figure:  {deform_path}")
    print(f"  Training curves:     {curves_path}")
    print(f"  Metrics CSV:         {csv_path}")
    print(f"  LLM log:             {llm_log}")
    print(f"  LLM samples report:  {out_dir / 'llm_prompt_output_samples.txt'}")
    print(f"  Output directory:    {out_dir}")

    # Save run config
    config = vars(args)
    config["final_return"] = final_mean_ret
    config["final_rmse"] = final_mean_rmse
    config["ppo_time_s"] = round(ppo_time, 1)
    if llm_stats:
        config["llm_stats"] = llm_stats
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
