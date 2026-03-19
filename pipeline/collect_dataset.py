# collect_dataset.py
# data conversion and collection utilities used for dataset creation (:

import argparse
import numpy as np
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv
from smartfixture_llm4teach.envs.teacher_env import SmartFixtureTeacherEnv
from smartfixture_llm4teach.teacher.llm_adapter import build_hf_llm_fn

def collect_teacher_dataset(
    episodes=20,
    max_steps=20,
    out_prefix="teacher_data",
    anchor_mode="stride",
    anchor_stride=4,
    hf_model_id: str = "",
    hf_max_new_tokens: int = 256,
    hf_temperature: float = 0.2,
    hf_top_p: float = 0.9,
):
    """
    Collect dataset from teacher policy^^
    
    Args:
        episodes: number of episodes to collect
        max_steps: maximum steps per episode
        out_prefix: prefix for output files (X.npy and P.npy)
        anchor_mode: fixture anchor configuration ("stride", "twelve", "boundary", "all")
        anchor_stride: stride for "stride" mode
    """
    core = FixtureCore(anchor_mode=anchor_mode, anchor_stride=anchor_stride)
    env = SmartFixtureStudentEnv(core)
    llm_fn = None
    if hf_model_id:
        llm_fn = build_hf_llm_fn(
            model_id=hf_model_id,
            max_new_tokens=hf_max_new_tokens,
            temperature=hf_temperature,
            top_p=hf_top_p,
        )
    teacher_env = SmartFixtureTeacherEnv(core, llm_fn=llm_fn)
    print(f"Environment created with anchor_mode='{anchor_mode}', num_anchors={core.num_anchors}")
    
    X_list = []
    P_list = []

    for ep in range(episodes):
        obs = env.reset()
        done = False

        for t in range(max_steps):
            # 1. query teacher to get distribution over full action set
            teacher_probs = teacher_env.get_teacher_distribution()

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
    parser.add_argument("--hf-model-id", type=str, default="", help="HF model id for teacher (e.g., openai/gpt-oss-120b)")
    parser.add_argument("--hf-max-new-tokens", type=int, default=256, help="HF max new tokens for teacher")
    parser.add_argument("--hf-temperature", type=float, default=0.2, help="HF sampling temperature for teacher")
    parser.add_argument("--hf-top-p", type=float, default=0.9, help="HF top-p for teacher")
    args = parser.parse_args()
    
    collect_teacher_dataset(
        episodes=args.episodes,
        max_steps=args.max_steps,
        out_prefix=args.out_prefix,
        anchor_mode=args.anchor_mode,
        anchor_stride=args.anchor_stride,
        hf_model_id=args.hf_model_id,
        hf_max_new_tokens=args.hf_max_new_tokens,
        hf_temperature=args.hf_temperature,
        hf_top_p=args.hf_top_p,
    )
