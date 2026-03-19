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
import importlib.util
import types
import copy
import json
import csv
import matplotlib.pyplot as plt

from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv
from smartfixture_llm4teach.envs.teacher_env import SmartFixtureTeacherEnv
from smartfixture_llm4teach.teacher.llm_adapter import build_hf_llm_fn
from distill_student import StudentPolicyDiscrete, distill_train



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
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

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
):
    """
    PPO fine-tuning.
    
    Args:
        force_scratch: if True, always use random init and skip distilled checkpoint!!
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    core = FixtureCore()
    env = SmartFixtureStudentEnv(core)
    obs0 = env.reset()
    obs_dim = obs0.shape[0]

    # try load distilled student checkpoint (unless force_scratch=True)
    has_state = False
    state = None
    student_act_dim = env.action_space.n
    ckpt_student = Path("student_policy_init.pt")

    # helper: load the LLM4Teach module from the workspace file (it has spaces)
    def _load_llm4teach():
        p = Path("LLM4Teach inner loop.py").resolve()
        if not p.exists():
            return None
        spec = importlib.util.spec_from_file_location("llm4teach", str(p))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # type: ignore
        return m

    llm_mod = _load_llm4teach()
    if pretrain_with_teacher and llm_mod is None:
        print("Warning: requested teacher pretraining but 'LLM4Teach inner loop.py' not found.")

    if not force_scratch and ckpt_student.exists():
        try:
            student_act_dim, state = _infer_student_act_dim(ckpt_student)
            has_state = True
            print(f"Inferred student_act_dim = {student_act_dim}")
        except Exception as e:
            print("Warning: could not infer student_act_dim from student_policy_init.pt:", e)

    act_dim = student_act_dim
    ac = ActorCriticDiscrete(obs_dim, act_dim, hidden=64)

    # warm start from distilled policy if available
    if has_state:
        try:
            student = StudentPolicyDiscrete(obs_dim, act_dim)  # hidden=64 expected
            student.load_state_dict(state)
            ac.pi_net.load_state_dict(student.net.state_dict())
            print("Loaded distilled policy into PPO actor.")
        except Exception as e:
            print("Warning: could not load distilled weights into PPO actor, using random init.", e)
    else:
        if force_scratch:
            print("force_scratch=True; using random policy initialization (no checkpoint).")
        else:
            print("No distilled checkpoint loaded; PPO starts from random policy.")
    # Optionally run teacher collection + distillation to create `student_policy_init.pt`
    if pretrain_with_teacher and (force_scratch or not ckpt_student.exists()):
        print("Running teacher collection + distillation (this may take a while)...")
        llm_fn = None
        if hf_model_id:
            llm_fn = build_hf_llm_fn(
                model_id=hf_model_id,
                max_new_tokens=hf_max_new_tokens,
                temperature=hf_temperature,
                top_p=hf_top_p,
            )
        teacher_env = SmartFixtureTeacherEnv(core, llm_fn=llm_fn, temperature=teacher_temperature)

        def collect_teacher_for_student(student_env, teacher_env, n_samples=256):
            X, P = [], []
            obs = student_env.reset().astype(np.float32)
            for _ in range(n_samples):
                probs = teacher_env.get_teacher_distribution()
                X.append(obs.flatten())
                P.append(probs.copy())

                action_id = int(np.random.choice(len(probs), p=probs))
                obs, *_ = student_env.step(action_id)
            X = np.stack(X, 0).astype(np.float32)
            P = np.stack(P, 0).astype(np.float32)
            return X, P

        X_np, P_np = collect_teacher_for_student(env, teacher_env, n_samples=teacher_samples)
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
        np.save("teacher_data_X.npy", X_np)
        np.save("teacher_data_P.npy", P_np)
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
        torch.save(student_model.state_dict(), ckpt_student)
        print(f"Saved distilled student checkpoint: {ckpt_student}")

        def evaluate_student(student_model, env, n_episodes=32):
            rets = []
            match_counts = []
            idx_diffs = []
            for _ in range(n_episodes):
                obs = env.reset().astype(np.float32)
                done = False
                ep_ret = 0.0
                while not done:
                    obs_t = torch.tensor(obs.flatten(), dtype=torch.float32)
                    with torch.no_grad():
                        probs, _ = student_model(obs_t.unsqueeze(0))
                    act = int(torch.argmax(probs, dim=-1).item())

                    anchor_sags = obs[:env.core.num_anchors]
                    if env.core.mask.sum() < env.core.max_fixtures:
                        node_id = int(np.argmax(anchor_sags))
                        heur = env.core.action_catalog.index("add", node_id)
                    else:
                        node_id = int(np.argmin(anchor_sags))
                        heur = env.core.action_catalog.index("remove", node_id)

                    match_counts.append(1 if act == heur else 0)
                    idx_diffs.append(abs(act - heur))
                    next_obs, r, done, info = env.step(act)
                    ep_ret += float(r)
                    obs = next_obs
                rets.append(ep_ret)
            return {
                "mean_return": float(np.mean(rets)),
                "std_return": float(np.std(rets)),
                "heur_match_rate": float(np.mean(match_counts)),
                "mean_idx_diff": float(np.mean(idx_diffs)),
            }

        eval_res = evaluate_student(student_model, env, n_episodes=32)
        print("Pre-PPO distilled student eval:", eval_res)
        try:
            student_act_dim = env.action_space.n
            state = student_model.state_dict()
            has_state = True
        except Exception:
            pass

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

        # render control (avoid matplotlib figure explosion)
        if render_every and (it % render_every == 0):
            env.render(title=f"PPO iter {it}")

        # save periodically
        if save_every and (it % save_every == 0):
            torch.save(ac.state_dict(), out_path)
            # also keep an iter-stamped snapshot
            snap = out_path.with_name(out_path.stem + f"_iter{it:03d}" + out_path.suffix)
            try:
                torch.save(ac.state_dict(), snap)
            except Exception:
                pass

    # final save
    torch.save(ac.state_dict(), out_path)
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
    p.add_argument("--force-scratch", action="store_true", help="Force random init even if student checkpoint exists")

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
    p.add_argument("--eval-student", action="store_true", help="Load student checkpoint and run detailed evaluation vs heuristic")
    p.add_argument("--eval-episodes", type=int, default=64, help="Episodes to use for student evaluation")
    p.add_argument("--sweep-teacher-samples", type=str, default=None, help="Comma-separated teacher sample counts to sweep, e.g. '8,16,32'")
    return p


def evaluate_student_checkpoint(ckpt_path: Path, env: SmartFixtureStudentEnv, n_episodes: int = 64):
    """Load student checkpoint and evaluate detailed similarity metrics vs heuristic."""
    if not ckpt_path.exists():
        print(f"Student checkpoint {ckpt_path} not found.")
        return None

    # infer dims
    state = torch.load(ckpt_path, map_location="cpu")
    # try to find final bias to get action dim
    bias_keys = [k for k in state.keys() if k.endswith("net.4.bias")]
    if not bias_keys:
        print("Could not infer action dim from checkpoint.")
        return None
    act_dim = int(state[bias_keys[0]].shape[0])

    obs0 = env.reset()
    obs_dim = obs0.shape[0]

    student = StudentPolicyDiscrete(obs_dim, act_dim)
    try:
        student.load_state_dict(state)
    except Exception as e:
        print("Warning: could not fully load student state:", e)

    # metrics accumulators
    match = []
    idx_diffs = []
    kls = []
    imp = []
    returns_student = []
    returns_heur = []

    for ep in range(n_episodes):
        env0 = copy.deepcopy(env)
        obs = env0.reset()
        done = False
        ep_ret_s = 0.0
        ep_ret_h = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs, _ = student(obs_t)
            probs = probs.squeeze(0).numpy()
            student_idx = int(np.argmax(probs))

            # heuristic: add max sag if below max fixtures, else remove min sag
            anchor_sags = obs[:env0.core.num_anchors]
            if env0.core.mask.sum() < env0.core.max_fixtures:
                node_id = int(np.argmax(anchor_sags))
                heur_idx = env0.core.action_catalog.index("add", node_id)
            else:
                node_id = int(np.argmin(anchor_sags))
                heur_idx = env0.core.action_catalog.index("remove", node_id)

            # KL between student probs and heuristic one-hot
            one_hot = np.zeros_like(probs, dtype=np.float32)
            one_hot[heur_idx] = 1.0
            # avoid log(0)
            kl = float(np.sum(one_hot * (np.log(one_hot + 1e-12) - np.log(probs + 1e-12))))

            # simulate both actions from a copied env state
            env_s = copy.deepcopy(env0)
            env_h = copy.deepcopy(env0)
            _, r_s, done_s, _ = env_s.step(student_idx)
            _, r_h, done_h, _ = env_h.step(heur_idx)

            match.append(1 if student_idx == heur_idx else 0)
            idx_diffs.append(abs(student_idx - heur_idx))
            # coordinate distance between chosen anchors
            try:
                action_s = env0.core.action_catalog.get(student_idx)
                action_h = env0.core.action_catalog.get(heur_idx)
                if action_s.node_id is None or action_h.node_id is None:
                    coord_dist = float(abs(student_idx - heur_idx))
                else:
                    coord_s = env0.core.anchor_xy[action_s.node_id]
                    coord_h = env0.core.anchor_xy[action_h.node_id]
                    coord_dist = float(np.linalg.norm(coord_s - coord_h))
            except Exception:
                coord_dist = float(abs(student_idx - heur_idx))
            # store as KL/list append via kls/idx_diffs? add separate list
            # we'll repurpose kls list for numeric KL and create separate coord_dists
            if 'coord_dists' not in locals():
                coord_dists = []
            coord_dists.append(coord_dist)
            kls.append(kl)
            imp.append(1 if r_s >= r_h else 0)

            # advance original env with student action to continue episode for returns
            obs, r, done, _ = env0.step(student_idx)
            ep_ret_s += float(r)

            # also advance a separate env with heuristic for return comparison
            obs_h, r_h_step, done_h2, _ = env_h.step(heur_idx)
            ep_ret_h += float(r_h_step)

        returns_student.append(ep_ret_s)
        returns_heur.append(ep_ret_h)

    res = {
        "mean_return_student": float(np.mean(returns_student)),
        "mean_return_heuristic": float(np.mean(returns_heur)),
        "action_match_rate": float(np.mean(match)),
        "mean_idx_diff": float(np.mean(idx_diffs)),
        "mean_kl": float(np.mean(kls)),
        "mean_coord_dist": float(np.mean(coord_dists)) if 'coord_dists' in locals() and coord_dists else None,
        "improvement_rate_over_heuristic": float(np.mean(imp)),
    }
    print("Student evaluation result:", res)
    return res


def sweep_teacher_samples(samples_list, *, distill_steps=2000, distill_temp=1.0, distill_kl_weight=0.05, eval_episodes=512, distill_mode='kl', distill_ce_weight=0.0):
    """Run sequential distillation+eval for different teacher sample counts and save results.
    Returns list of result dicts.
    """
    results = []
    out_json = Path("sweep_teacher_samples_results.json")
    out_csv = Path("sweep_teacher_samples_results.csv")
    out_png = Path("sweep_teacher_samples_kl.png")

    for s in samples_list:
        print(f"\n=== Sweep run: teacher_samples={s} ===")
        # run distill-only + eval via ppo_finetune plumbing
        ppo_finetune(
            num_iters=0,
            episodes_per_iter=0,
            render_every=0,
            save_every=0,
            out_name="ppo_fixture_policy.pt",
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
            force_scratch=True,
            pretrain_with_teacher=True,
            teacher_samples=int(s),
            teacher_temperature=0.7,
            distill_steps=distill_steps,
            distill_temp=distill_temp,
            distill_kl_weight=distill_kl_weight,
            distill_mode=distill_mode,
            distill_ce_weight=distill_ce_weight,
        )

        # evaluate distilled student
        res = evaluate_student_checkpoint(Path("student_policy_init.pt"), SmartFixtureStudentEnv(FixtureCore()), n_episodes=eval_episodes)
        if res is None:
            res = {"mean_kl": None}
        res_entry = {"teacher_samples": int(s), **res}
        results.append(res_entry)

        # save partial results after each run
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        # write csv
        keys = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    # plot KL vs teacher_samples
    xs = [r["teacher_samples"] for r in results]
    ys = [r.get("mean_kl", float("nan")) for r in results]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("teacher_samples")
    plt.ylabel("mean_kl")
    plt.title("Mean KL vs teacher_samples")
    plt.grid(True)
    plt.savefig(out_png)
    print(f"Saved sweep JSON: {out_json}, CSV: {out_csv}, plot: {out_png}")
    return results


if __name__ == "__main__":
    args = _build_argparser().parse_args()
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
    )
    # optional evaluation run
    if args.eval_student:
        evaluate_student_checkpoint(Path("student_policy_init.pt"), SmartFixtureStudentEnv(FixtureCore()), n_episodes=args.eval_episodes)
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
