# ppo_finetune.py
"""
PPO fine-tuning for DiscreteFixtureEnv 
main pipeline (teacher collection, distillation, warm-start PPO, sweeps)
Categorical actions =UwU=

Key upgrades from prev version:
- Default to 50 iters (more stable magnitude).
- Collects a *larger batch per PPO iter* by rolling out multiple short episodes
  (episodes_per_iter) instead of a single episode.
- Uses a *persistent Adam optimizer* (does NOT reset optimizer state each iter).
- Logs: mean episode return + mean/std RMSE (and any extra metrics if env provides them).
- Saves checkpoint every iter + final save to ppo_fixture_policy.pt
- Optional CLI args for quick hyperparameter sweeps.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# dynamic import utilities (for loading the LLM4Teach script that has a space in filename)
import copy
import json
import csv
import matplotlib.pyplot as plt
import time

from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv
from smartfixture_llm4teach.envs.teacher_env import SmartFixtureTeacherEnv
from smartfixture_llm4teach.envs.teacher_env import build_strict_hf_llm_fn
from distill_student import StudentPolicyDiscrete
from validation_runtime import (environment_config, action_schema, source_hashes, package_versions, archive_sources,
    save_checkpoint, read_metadata, write_json, collect_teacher, transfer_student,
    rollout)



# PPO pieces


class ActorCriticDiscrete(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.pi_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def policy_logits(self, obs_t):
        return self.pi_net(obs_t)

    def value(self, obs_t):
        return self.v_net(obs_t).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_np):
        """
        obs_np: numpy vector (F,)
        returns: action_id, logp, val, dist
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        logits = self.policy_logits(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        v = self.value(obs_t)
        return int(a.item()), float(logp.item()), float(v.item()), dist

    def evaluate_actions(self, obs_batch_t, act_batch_t):
        """
        obs_batch_t: (B,F) tensor
        act_batch_t: (B,) int tensor
        returns: new_logp, entropy, value_pred
        """
        logits = self.policy_logits(obs_batch_t)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_batch_t)
        ent = dist.entropy()
        v_pred = self.value(obs_batch_t)
        return logp, ent, v_pred


def gae_advantage(rews, vals, dones, gamma=0.99, lam=0.95):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - float(dones[t])
        nextv = vals[t + 1] if (t + 1) < T else 0.0
        delta = rews[t] + gamma * nextv * nonterm - vals[t]
        lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        adv[t] = lastgaelam
    ret = adv + np.array(vals[:T], dtype=np.float32)
    return adv, ret


def ppo_update(
    ac: ActorCriticDiscrete,
    opt: torch.optim.Optimizer,
    batch: dict,
    clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    epochs=4,
    minibatch=64,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
):
    """
    One PPO update over batch.
    batch keys:
      obs_buf (T,F), act_buf (T,), logp_buf (T,), rew_buf (T,), val_buf (T,), done_buf (T,)
    """
    obs_all      = torch.tensor(batch["obs_buf"],  dtype=torch.float32)
    act_all      = torch.tensor(batch["act_buf"],  dtype=torch.int64)
    oldlogp_all  = torch.tensor(batch["logp_buf"], dtype=torch.float32)
    val_all      = torch.tensor(batch["val_buf"],  dtype=torch.float32)
    rew_all      = torch.tensor(batch["rew_buf"],  dtype=torch.float32)
    done_all     = batch["done_buf"]  # list[bool]

    # advantages/returns
    vals_np = list(val_all.numpy())
    adv_np, ret_np = gae_advantage(
        list(rew_all.numpy()),
        vals_np,
        done_all,
        gamma=gamma,
        lam=lam,
    )
    adv_all = torch.tensor(adv_np, dtype=torch.float32)
    ret_all = torch.tensor(ret_np, dtype=torch.float32)

    # normalize advantages
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std(unbiased=False) + 1e-8)

    T = obs_all.shape[0]
    idxs = np.arange(T)

    for _ in range(epochs):
        np.random.shuffle(idxs)
        for start in range(0, T, minibatch):
            mb = idxs[start:start + minibatch]
            if len(mb) == 0:
                continue

            mb_obs   = obs_all[mb]
            mb_act   = act_all[mb]
            mb_oldlp = oldlogp_all[mb]
            mb_adv   = adv_all[mb]
            mb_ret   = ret_all[mb]

            new_logp, entropy, v_pred = ac.evaluate_actions(mb_obs, mb_act)
            ratio = torch.exp(new_logp - mb_oldlp)

            # policy loss
            clip_ratio = torch.clamp(ratio, 1 - clip, 1 + clip)
            pg_loss = -(torch.min(ratio * mb_adv, clip_ratio * mb_adv)).mean()

            # value loss
            v_loss = F.mse_loss(v_pred, mb_ret)

            # entropy bonus
            ent_loss = -ent_coef * entropy.mean()

            loss = pg_loss + vf_coef * v_loss + ent_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            opt.step()


def _infer_student_act_dim(ckpt_path: Path):
    state = torch.load(ckpt_path, map_location="cpu")
    bias_keys = [k for k in state.keys() if k.endswith("net.4.bias")]
    if not bias_keys:
        raise RuntimeError("Could not find net.4.bias in checkpoint.")
    return int(state[bias_keys[0]].shape[0]), state


def ppo_finetune(
    num_iters=50,
    episodes_per_iter=20,
    render_every=0,
    save_every=1,
    out_name="ppo_fixture_policy.pt",
    # PPO hypers
    lr=3e-4,
    clip=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    epochs=4,
    minibatch=64,
    max_grad_norm=0.5,
    gamma=0.99,
    lam=0.95,
    seed=13,
    force_scratch=False,
    pretrain_with_teacher=False,
    teacher_samples=256,
    teacher_temperature=0.7,
    distill_steps=300,
    distill_temp=1.0,
    distill_kl_weight=1.0,
    distill_mode='kl',  # 'kl' or 'hybrid'
    distill_ce_weight=0.0,
    hf_model_id: str = "",
    hf_max_new_tokens: int = 256,
    hf_temperature: float = 0.2,
    hf_top_p: float = 0.9,
    hf_cache_dir: str = "",
    hf_local_files_only: bool = False,
    hf_device_map: str = "auto",
    save_iter_snapshots: bool = False,
    anchor_mode: str = "grid100",
    anchor_stride: int = 4,
    grid_size: int = 10,
    max_fixtures: int = -1,
    init_mode=None,
    student_checkpoint=None,
    teacher_source=None,
    allow_teacher_fallback=False,
    core_config=None,
    training_instances=None,
    llm_fn=None,
):
    """
    PPO fine-tuning.
    
    Args:
        force_scratch: legacy flag; with pretrain_with_teacher, ignore old checkpoints
            but still initialize from the freshly distilled student.
    """
    started = time.perf_counter()
    if num_iters < 0 or (num_iters > 0 and episodes_per_iter <= 0):
        raise ValueError("Invalid iteration/episode count")
    mode = init_mode or ("pretrain" if pretrain_with_teacher else "scratch" if force_scratch else "load" if student_checkpoint else "scratch")
    if mode not in {"scratch", "load", "pretrain"}:
        raise ValueError("init_mode must be scratch, load or pretrain")
    if init_mode and ((pretrain_with_teacher and mode != "pretrain") or (force_scratch and mode == "load")):
        raise ValueError("Conflicting legacy and explicit initialization flags")
    source = teacher_source or ("llm" if hf_model_id or llm_fn else "heuristic")
    if source not in {"heuristic", "llm"}:
        raise ValueError("teacher_source must be heuristic or llm")
    if mode == "pretrain" and source == "llm" and not (hf_model_id or llm_fn):
        raise ValueError("LLM pretraining requires a model ID or explicit callable")
    if mode == "pretrain" and (distill_steps <= 0 or distill_temp <= 0 or teacher_temperature <= 0):
        raise ValueError("Pretraining requires positive steps and temperatures")
    out_path = Path(out_name).resolve()
    artifact_dir = out_path.with_name(out_path.stem + "_artifacts")
    if out_path.exists() or artifact_dir.exists():
        raise FileExistsError(f"Use a new output path; refusing to overwrite {out_path} or its artifacts")
    artifact_dir.mkdir(parents=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = environment_config(dict(anchor_mode=anchor_mode, anchor_stride=anchor_stride,
        grid_size=grid_size, max_fixtures=max_fixtures))
    config.update(core_config or {})
    config = environment_config(config)
    config["seed"] = seed
    configs = [environment_config({**config, **item, "seed": seed}) for item in (training_instances or [{}])]
    core = FixtureCore(**configs[0])
    env = SmartFixtureStudentEnv(core)
    obs0 = env.reset()
    obs_dim = obs0.shape[0]
    act_dim = env.action_space.n
    for cfg in configs:
        candidate = FixtureCore(**cfg)
        if candidate.reset().size != obs_dim or action_schema(candidate) != action_schema(core):
            raise ValueError("Training instances must share observation and action schemas")
    ac = ActorCriticDiscrete(obs_dim, act_dim, hidden=64)
    metadata = {"schema_version": 1, "environment": configs[0], "training_instances": configs,
        "obs_dim": obs_dim, "act_dim": act_dim, "action_catalog": action_schema(core),
        "initialization": mode, "seed": seed, "source_hashes": source_hashes(), "package_versions": package_versions(),
        "teacher": {"source": source if mode == "pretrain" else None, "model_id": hf_model_id,
            "callable_origin": getattr(llm_fn, "provenance", {}).get("adapter", "injected_unverified") if llm_fn is not None else "strict_hf" if source == "llm" else "heuristic",
            "allow_fallback": allow_teacher_fallback, "temperature": teacher_temperature,
            "generation_temperature": hf_temperature, "top_p": hf_top_p, "max_new_tokens": hf_max_new_tokens},
        "training": {"iterations": num_iters, "episodes_per_iter": episodes_per_iter,
            "lr": lr, "clip": clip, "entropy": ent_coef, "value_coefficient": vf_coef,
            "epochs": epochs, "minibatch": minibatch, "gamma": gamma, "lambda": lam,
            "max_grad_norm": max_grad_norm, "distill_steps": distill_steps,
            "distill_temperature": distill_temp, "distill_mode": distill_mode,
            "distill_kl_weight": distill_kl_weight, "distill_ce_weight": distill_ce_weight}}
    ckpt_student = artifact_dir / "student_policy_init.pt"
    archive_sources(artifact_dir / "source_snapshot", metadata["source_hashes"])
    # Preserve the critic and subsequent PPO RNG streams regardless of teacher cost.
    numpy_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    if mode == "load":
        if not student_checkpoint:
            raise ValueError("init_mode=load requires student_checkpoint")
        loaded_meta = read_metadata(student_checkpoint)
        if ({k: v for k, v in loaded_meta["environment"].items() if k not in {"seed", "instance_id"}}
            != {k: v for k, v in configs[0].items() if k not in {"seed", "instance_id"}}
            or loaded_meta["action_catalog"] != metadata["action_catalog"]):
            raise ValueError("Student checkpoint configuration differs from this run")
        student = StudentPolicyDiscrete(obs_dim, act_dim)
        student.load_state_dict(torch.load(student_checkpoint, map_location="cpu", weights_only=True), strict=True)
        transfer_student(ac, student, obs0[None, :])
        metadata["loaded_student"] = loaded_meta
    if mode == "pretrain":
        np.random.seed(seed + 10000)
        torch.manual_seed(seed + 10000)
        print("Running teacher collection + distillation (this may take a while)...")
        if source == "llm" and llm_fn is None:
            llm_fn = build_strict_hf_llm_fn(
                model_id=hf_model_id,
                max_new_tokens=hf_max_new_tokens,
                temperature=hf_temperature,
                top_p=hf_top_p,
                device_map=hf_device_map,
                cache_dir=(hf_cache_dir or None),
                local_files_only=hf_local_files_only,
            )
        def teacher_episode(episode):
            episode_core = FixtureCore(**configs[episode % len(configs)])
            episode_env = SmartFixtureStudentEnv(episode_core)
            teacher = SmartFixtureTeacherEnv(episode_core,
                llm_fn=llm_fn if source == "llm" else None, temperature=teacher_temperature,
                strict=source == "llm", allow_fallback=allow_teacher_fallback if source == "llm" else True)
            return episode_env, teacher

        teacher_student_env, teacher_env = teacher_episode(0)
        X_np, P_np, records = collect_teacher(teacher_student_env, teacher_env,
            teacher_samples, seed + 10000, episode_factory=teacher_episode)
        write_json(artifact_dir / "teacher_records.json", records)
        metadata["teacher"].update(samples=len(records),
            llm_samples=sum(r.get("source") == "llm" for r in records),
            fallback_samples=sum(source == "llm" and r.get("fallback", False) for r in records),
            query_seconds=sum(r.get("query_seconds", 0) for r in records),
            input_tokens=sum(r.get("input_tokens") or 0 for r in records),
            output_tokens=sum(r.get("output_tokens") or 0 for r in records))
        # apply teacher temperature scaling (sharpening/softening)
        def apply_temperature(P, T):
            if abs(T - 1.0) < 1e-8:
                return P
            eps = 1e-12
            logp = np.log(P + eps)
            logp = logp / float(T)
            pnew = np.exp(logp)
            pnew = pnew / (pnew.sum(axis=1, keepdims=True) + eps)
            return pnew.astype(np.float32)

        P_np = apply_temperature(P_np, distill_temp)
        np.save(artifact_dir / "teacher_data_X.npy", X_np)
        np.save(artifact_dir / "teacher_data_P.npy", P_np)
        print(f"Saved teacher_data_X.npy ({X_np.shape}) and teacher_data_P.npy ({P_np.shape})")

        X_t = torch.tensor(X_np, dtype=torch.float32)
        P_t = torch.tensor(P_np, dtype=torch.float32)

        # local distillation routine with KL weighting
        def run_local_distill(obs_X, probs_P, steps=300, lr=5e-3, beta_reg=1e-4, kl_w=1.0, mode='kl', ce_w=0.0):
            model = StudentPolicyDiscrete(obs_X.shape[1], probs_P.shape[1], hidden=64)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            ce_loss = torch.nn.CrossEntropyLoss()
            hard_labels = torch.tensor(np.argmax(probs_P.numpy(), axis=1), dtype=torch.long)
            for it in range(steps):
                probs, logits = model(obs_X)
                # KL term (teacher || student)
                kl = torch.sum(probs_P * (torch.log(probs_P + 1e-8) - torch.log(probs + 1e-8)), dim=1).mean()
                reg = sum((p ** 2).sum() for p in model.parameters())
                loss = kl_w * kl
                if mode == 'hybrid' and ce_w > 0.0:
                    # Cross-entropy expects logits and long labels
                    ce = ce_loss(logits, hard_labels)
                    loss = kl_w * kl + ce_w * ce
                loss = loss + beta_reg * reg
                opt.zero_grad(); loss.backward(); opt.step()
                if (it + 1) % 50 == 0:
                    kl_val = kl.item() if isinstance(kl, torch.Tensor) else float(kl)
                    print(f"[distill] step {it+1:4d} loss={loss.item():.4f} kl={kl_val:.4f}")
            return model

        student_model = run_local_distill(X_t, P_t, steps=distill_steps, lr=5e-3, beta_reg=1e-4, kl_w=distill_kl_weight, mode=distill_mode, ce_w=distill_ce_weight)
        save_checkpoint(student_model, ckpt_student, {**metadata, "model_type": "student"})
        print(f"Saved distilled student checkpoint: {ckpt_student}")

        transfer_student(ac, student_model, X_np)
        metadata["teacher"]["weight_transfer_verified"] = True

    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    metadata["pretraining_seconds"] = time.perf_counter() - started
    metadata["model_type"] = "actor_critic"
    ac.validation_metadata = metadata
    save_checkpoint(ac, artifact_dir / "initial_actor.pt", metadata)
    history = [{"iteration": 0, "ppo_transitions": 0,
                "evaluation": [rollout(ac, cfg) for cfg in configs]}]
    transitions = 0

    # persistent optimizer (important!)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)

    # training
    out_path = Path(out_name).resolve()
    print(f"Training PPO; saving to: {out_path}")

    for it in range(1, num_iters + 1):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        ep_returns = []
        ep_rmses = []
        ep_extra = {"mean_abs": [], "max_abs": [], "min_abs": []}

        # collect multiple episodes per iter (bigger batch)
        for ep in range(episodes_per_iter):
            cfg = configs[((it - 1) * episodes_per_iter + ep) % len(configs)]
            env = SmartFixtureStudentEnv(FixtureCore(**cfg))
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

            # optional extra metrics (only if env provides)
            if "mean_abs" in last_info: ep_extra["mean_abs"].append(float(last_info["mean_abs"]))
            if "max_abs"  in last_info: ep_extra["max_abs"].append(float(last_info["max_abs"]))
            if "min_abs"  in last_info: ep_extra["min_abs"].append(float(last_info["min_abs"]))

        batch = {
            "obs_buf":  np.array(obs_buf,  dtype=np.float32),
            "act_buf":  np.array(act_buf,  dtype=np.int64),
            "logp_buf": np.array(logp_buf, dtype=np.float32),
            "rew_buf":  np.array(rew_buf,  dtype=np.float32),
            "val_buf":  np.array(val_buf,  dtype=np.float32),
            "done_buf": done_buf,
        }

        # PPO update
        if len(obs_buf) > 0:
            ppo_update(
                ac, opt, batch,
                clip=clip, vf_coef=vf_coef, ent_coef=ent_coef,
                epochs=epochs, minibatch=minibatch,
                max_grad_norm=max_grad_norm,
                gamma=gamma, lam=lam,
            )

        # logs (stable, episode-level)
        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_rmse = float(np.nanmean(ep_rmses)) if ep_rmses else float("nan")
        std_rmse  = float(np.nanstd(ep_rmses)) if ep_rmses else float("nan")

        msg = f"[PPO] Iter {it:03d}/{num_iters} | mean return {mean_ret:+.4f} | rmse {mean_rmse:.4f}±{std_rmse:.4f}"

        # append extra metrics if present
        if ep_extra["mean_abs"]:
            msg += f" | mean_abs {float(np.mean(ep_extra['mean_abs'])):.4f}"
        if ep_extra["max_abs"]:
            msg += f" | max_abs {float(np.mean(ep_extra['max_abs'])):.4f}"

        print(msg)
        transitions += len(obs_buf)
        history.append({"iteration": it, "ppo_transitions": transitions,
            "training_mean_return": mean_ret,
            "evaluation": [rollout(ac, cfg) for cfg in configs]})
        write_json(artifact_dir / "learning_curve.json", history)

        # render control (avoid matplotlib figure explosion)
        if render_every and (it % render_every == 0):
            env.render()

        # save periodically
        if save_every and (it % save_every == 0):
            save_checkpoint(ac, out_path, {**metadata, "iteration": it, "ppo_transitions": transitions})
            # optional iter-stamped snapshots (can be expensive for long runs)
            if save_iter_snapshots:
                snap = out_path.with_name(out_path.stem + f"_iter{it:03d}" + out_path.suffix)
                save_checkpoint(ac, snap, {**metadata, "iteration": it, "ppo_transitions": transitions})

    # final save
    metadata.update(iteration=num_iters, ppo_transitions=transitions,
                    teacher_collection_transitions=teacher_samples if mode == "pretrain" else 0,
                    training_diagnostic_transitions=(num_iters + 1) * sum(c["max_steps"] for c in configs),
                    total_seconds=time.perf_counter() - started)
    save_checkpoint(ac, out_path, metadata)
    write_json(artifact_dir / "learning_curve.json", history)
    write_json(artifact_dir / "manifest.json", metadata)
    print(f"Saved PPO fixture policy to {out_path}")
    return ac


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--episodes-per-iter", type=int, default=20)
    p.add_argument("--render-every", type=int, default=0, help="0 disables rendering during training")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--out", type=str, default="ppo_fixture_policy.pt")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--force-scratch", action="store_true", help="Ignore old checkpoints; with --pretrain-teacher still load the new student")
    p.add_argument("--init-mode", choices=["scratch", "load", "pretrain"])
    p.add_argument("--student-checkpoint")
    p.add_argument("--teacher-source", choices=["heuristic", "llm"])
    p.add_argument("--allow-teacher-fallback", action="store_true")
    p.add_argument("--env-config", help="JSON object of shared-core constructor settings")

    # PPO hypers
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    # Teacher pretraining options
    p.add_argument("--pretrain-teacher", action="store_true", help="Collect teacher data and distill student before PPO")
    p.add_argument("--teacher-samples", type=int, default=256, help="Number of teacher samples to collect for distillation")
    p.add_argument("--teacher-temperature", type=float, default=0.7, help="Teacher LLM temperature")
    p.add_argument("--distill-steps", type=int, default=300, help="Number of distillation training steps")
    p.add_argument("--distill-temp", type=float, default=1.0, help="Temperature to apply to teacher probabilities before distillation")
    p.add_argument("--distill-kl-weight", type=float, default=1.0, help="Weight for KL term during distillation")
    p.add_argument("--distill-mode", choices=['kl','hybrid'], default='kl', help="Distillation mode: 'kl' or 'hybrid' (CE+KL)")
    p.add_argument("--distill-ce-weight", type=float, default=0.0, help="Weight for cross-entropy term when using hybrid distillation")
    p.add_argument("--hf-model-id", type=str, default="", help="HF model id for teacher (e.g., openai/gpt-oss-120b)")
    p.add_argument("--hf-max-new-tokens", type=int, default=256, help="HF max new tokens for teacher")
    p.add_argument("--hf-temperature", type=float, default=0.2, help="HF sampling temperature for teacher")
    p.add_argument("--hf-top-p", type=float, default=0.9, help="HF top-p for teacher")
    p.add_argument("--hf-cache-dir", type=str, default="", help="Optional local HF cache directory")
    p.add_argument("--hf-local-files-only", action="store_true", help="Load HF model from local cache only (no network)")
    p.add_argument("--hf-device-map", type=str, default="auto", help="HF device map (e.g., 'auto', 'cpu')")
    p.add_argument("--save-iter-snapshots", action="store_true", help="Also save per-iteration snapshot checkpoints (off by default to reduce I/O)")
    p.add_argument("--anchor-mode", type=str, default="grid100", choices=["grid100", "stride", "twelve", "boundary", "all"], help="Candidate fixture design space")
    p.add_argument("--anchor-stride", type=int, default=4, help="Stride when anchor-mode='stride'")
    p.add_argument("--grid-size", type=int, default=10, help="Grid size for the sheet (10 gives 100 locations)")
    p.add_argument("--max-fixtures", type=int, default=-1, help="Max active fixtures; <=0 means unconstrained")
    p.add_argument("--eval-student", action="store_true", help="Load student checkpoint and run detailed evaluation vs heuristic")
    p.add_argument("--eval-episodes", type=int, default=64, help="Episodes to use for student evaluation")
    p.add_argument("--sweep-teacher-samples", type=str, default=None, help="Comma-separated teacher sample counts to sweep, e.g. '8,16,32'")
    return p


def evaluate_student_checkpoint(ckpt_path: Path, env: SmartFixtureStudentEnv, n_episodes: int = 64):
    """Compatibility entry point; use only the checkpoint's verified environment."""
    from eval_policy import evaluate_checkpoint
    meta = read_metadata(ckpt_path)
    if action_schema(env.core) != meta["action_catalog"] or env.reset().size != meta["obs_dim"]:
        raise ValueError("Provided environment schema does not match student checkpoint")
    result = evaluate_checkpoint(ckpt_path, num_episodes=n_episodes)
    print("Raw student evaluation:", result)
    return result


def sweep_teacher_samples(samples_list, *, distill_steps=2000, distill_temp=1.0, distill_kl_weight=0.05, eval_episodes=512, distill_mode='kl', distill_ce_weight=0.0):
    """Legacy shared-output sweeps are unsupported; use the matched study runner."""
    raise RuntimeError("Legacy sweep retired. Use run_validation_study.py with distinct output directories and fixed suites.")


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.sweep_teacher_samples:
        raise SystemExit("Legacy sweep retired; use run_validation_study.py with distinct output directories")
    ppo_finetune(
        num_iters=args.iters,
        episodes_per_iter=args.episodes_per_iter,
        render_every=args.render_every,
        save_every=args.save_every,
        out_name=args.out,
        lr=args.lr,
        clip=args.clip,
        ent_coef=args.ent,
        vf_coef=args.vf,
        epochs=args.epochs,
        minibatch=args.minibatch,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        lam=args.lam,
        seed=args.seed,
        force_scratch=args.force_scratch,
        pretrain_with_teacher=args.pretrain_teacher if hasattr(args, 'pretrain_teacher') else args.pretrain_teacher,
        teacher_samples=args.teacher_samples,
        teacher_temperature=args.teacher_temperature,
        distill_steps=args.distill_steps,
        distill_temp=args.distill_temp,
        distill_kl_weight=args.distill_kl_weight,
        distill_mode=args.distill_mode,
        distill_ce_weight=args.distill_ce_weight,
        hf_model_id=args.hf_model_id,
        hf_max_new_tokens=args.hf_max_new_tokens,
        hf_temperature=args.hf_temperature,
        hf_top_p=args.hf_top_p,
        hf_cache_dir=args.hf_cache_dir,
        hf_local_files_only=args.hf_local_files_only,
        hf_device_map=args.hf_device_map,
        save_iter_snapshots=args.save_iter_snapshots,
        anchor_mode=args.anchor_mode,
        anchor_stride=args.anchor_stride,
        grid_size=args.grid_size,
        max_fixtures=args.max_fixtures,
        init_mode=args.init_mode,
        student_checkpoint=args.student_checkpoint,
        teacher_source=args.teacher_source,
        allow_teacher_fallback=args.allow_teacher_fallback,
        core_config=json.loads(Path(args.env_config).read_text()) if args.env_config else None,
    )
    # optional evaluation run
    if args.eval_student:
        from eval_policy import evaluate_checkpoint
        student_path = Path(args.out).resolve().with_name(Path(args.out).stem + "_artifacts") / "student_policy_init.pt"
        print(evaluate_checkpoint(student_path, num_episodes=args.eval_episodes))
    # optional sweep qwq
    if args.sweep_teacher_samples:
        parts = [int(x) for x in args.sweep_teacher_samples.split(",") if x.strip()]
        sweep_teacher_samples(
            parts,
            distill_steps=args.distill_steps,
            distill_temp=args.distill_temp,
            distill_kl_weight=args.distill_kl_weight,
            eval_episodes=args.eval_episodes,
            distill_mode=args.distill_mode,
            distill_ce_weight=args.distill_ce_weight,
        )
