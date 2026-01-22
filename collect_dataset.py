# collect_dataset.py
import argparse
import numpy as np
import json
from env_fixture import DiscreteFixtureEnv
from teacher_iface import translator_context, teacher_llm_fn, parse_teacher_reply

def collect_teacher_dataset(
    episodes=20,
    max_steps=20,
    out_prefix="teacher_data",
    anchor_mode="stride",
    anchor_stride=4,
):
    """
    Collect dataset from teacher policy!
    
    Args:
        episodes: number of episodes to collect
        max_steps: maximum steps per episode
        out_prefix: prefix for output files (X.npy and P.npy)
        anchor_mode: fixture anchor configuration ("stride", "twelve", "boundary", "all")
        anchor_stride: stride for "stride" mode
    """
    env = DiscreteFixtureEnv(anchor_mode=anchor_mode, anchor_stride=anchor_stride)
    print(f"Environment created with anchor_mode='{anchor_mode}', num_anchors={env.num_anchors}")
    
    X_list = []
    P_list = []

    for ep in range(episodes):
        obs = env.reset()
        done = False

        for t in range(max_steps):
            # 1. build teacher prompt
            ctx = translator_context(env, obs)
            prompt_str = json.dumps(ctx)

            # 2. query LLM (or mock teacher)
            raw_reply = teacher_llm_fn(prompt_str)

            # 3. parse teacher into probs over actions
            teacher_probs = parse_teacher_reply(raw_reply, env.num_anchors)

            # 4. log (obs, teacher_probs)
            X_list.append(obs.copy())
            P_list.append(teacher_probs.copy())

            # 5. sample action from teacher policy
            action_id = np.random.choice(len(teacher_probs), p=teacher_probs)

            # 6. step env
            obs, reward, done, info = env.step(action_id)

            if done:
                break

    X = np.stack(X_list, axis=0).astype(np.float32)
    P = np.stack(P_list, axis=0).astype(np.float32)

    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_P.npy", P)

    print(f"Collected {episodes} episodes with {max_steps} max steps per episode")
    print(f"Saved dataset shapes: X {X.shape}, P {P.shape}")
    return X, P

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect teacher dataset for distillation")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--out-prefix", type=str, default="teacher_data", help="Output file prefix")
    parser.add_argument("--anchor-mode", type=str, default="stride", help="Anchor mode: stride, twelve, boundary, all")
    parser.add_argument("--anchor-stride", type=int, default=4, help="Stride for 'stride' mode")
    args = parser.parse_args()
    
    collect_teacher_dataset(
        episodes=args.episodes,
        max_steps=args.max_steps,
        out_prefix=args.out_prefix,
        anchor_mode=args.anchor_mode,
        anchor_stride=args.anchor_stride,
    )
