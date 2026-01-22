# ppo_finetune.py
"""
PPO fine-tuning for DiscreteFixtureEnv 
Categorical actions!

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

from env_fixture import DiscreteFixtureEnv


# -------------------------
# PPO pieces
# -------------------------

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
):
    """
    PPO fine-tuning.
    
    Args:
        force_scratch: if True, always use random init and skip distilled checkpoint!!
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DiscreteFixtureEnv()
    obs0 = env.reset()
    obs_dim = obs0.shape[0]

    # try load distilled student checkpoint (unless force_scratch=True)
    has_state = False
    state = None
    student_act_dim = env.num_anchors
    ckpt_student = Path("student_policy_init.pt")

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
            from distill_student import StudentPolicyDiscrete
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

                # env action mapping (safe if act_dim != env.num_anchors)
                env_action = a_id % env.num_anchors

                next_obs, r, done, info = env.step(env_action)
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
    return p


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
    )
