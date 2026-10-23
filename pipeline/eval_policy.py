"""Evaluate raw policies using verified checkpoint configuration, never guessed defaults."""
import argparse
import json
from pathlib import Path

import torch

from ppo_finetune import ActorCriticDiscrete
from validation_runtime import read_metadata, rollout, write_json, environment_config, action_schema
from smartfixture_llm4teach.core.fixture_core import FixtureCore


def load_ppo(policy_path):
    path = Path(policy_path).resolve()
    meta = read_metadata(path)
    actor = ActorCriticDiscrete(meta["obs_dim"], meta["act_dim"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if meta.get("model_type") == "student":
        actor.pi_net.load_state_dict({k.removeprefix("net."): v for k, v in state.items()}, strict=True)
    else:
        actor.load_state_dict(state, strict=True)
    actor.eval()
    actor.validation_metadata = meta
    return path, actor


def evaluate_checkpoint(policy_path, instances=None, num_episodes=1, seed0=123, deterministic=True):
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    path, actor = load_ppo(policy_path)
    meta = actor.validation_metadata
    configs = [environment_config({**meta["environment"], **item}) for item in (instances or [{}])]
    results = []
    for index, config in enumerate(configs):
        core = FixtureCore(**config)
        if core.reset().size != meta["obs_dim"] or action_schema(core) != meta["action_catalog"]:
            raise ValueError("Held-out instance has an incompatible observation/action schema")
        # Geometry/load may change explicitly; evaluation must not change the task/reward.
        protocol_keys = ("reward_mode", "max_steps", "max_fixtures", "min_feedback_fixtures",
                         "fixture_penalty", "invalid_action_penalty", "reward_scale",
                         "reward_clip_min", "reward_clip_max", "useful_fixture_bonus",
                         "useful_fixture_min_improvement", "pre_threshold_progress_bonus",
                         "anchor_mode", "anchor_stride", "grid_size", "include_instance_features")
        if any(config.get(k) != meta["environment"].get(k) for k in protocol_keys):
            raise ValueError("Evaluation protocol differs from training")
        for episode in range(num_episodes):
            results.append({"instance_id": config.get("instance_id", str(index)),
                "episode": episode, "evaluation_seed": seed0 + episode,
                "environment": config,
                **rollout(actor, config, seed0 + episode, deterministic)})
    return {"schema_version": 1, "checkpoint": str(path),
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "training_seed": meta["seed"], "initialization": meta["initialization"],
            "note": "Repeated deterministic episodes are not independent instances or training seeds.",
            "records": results}


def eval_all(policy_path, num_episodes=1, render=False, seed0=123):
    if render:
        raise ValueError("Rendering is not part of the canonical quantitative evaluator")
    result = evaluate_checkpoint(policy_path, num_episodes=num_episodes, seed0=seed0)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=123)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--instances", help="JSON list of explicit held-out environment overrides")
    parser.add_argument("--out", help="New evaluation JSON; no postprocessed layouts")
    args = parser.parse_args()
    if args.out and Path(args.out).exists():
        raise FileExistsError(args.out)
    result = evaluate_checkpoint(args.policy_path,
        instances=json.loads(Path(args.instances).read_text()) if args.instances else None,
        num_episodes=args.num_episodes, seed0=args.seed0, deterministic=not args.sample)
    if args.out:
        write_json(args.out, result)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
