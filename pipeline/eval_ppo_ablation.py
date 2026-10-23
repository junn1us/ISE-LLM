"""Paired comparison of verified checkpoints, not an implicit retraining workflow."""
import argparse
import json
from pathlib import Path
import numpy as np

from eval_policy import evaluate_checkpoint
from validation_runtime import write_json, read_metadata


def compare_checkpoints(scratch, teacher, instances=None):
    left_meta, right_meta = read_metadata(scratch), read_metadata(teacher)
    protocol_keys = ("iterations", "episodes_per_iter", "lr", "clip", "entropy",
                     "value_coefficient", "epochs", "minibatch", "gamma", "lambda", "max_grad_norm")
    if (any(left_meta["training"].get(k) != right_meta["training"].get(k) for k in protocol_keys)
            or left_meta.get("ppo_transitions", 0) != right_meta.get("ppo_transitions", 0)
            or left_meta["training_instances"] != right_meta["training_instances"]):
        raise ValueError("Paired comparison requires matching PPO budgets, hyperparameters and training instances")
    left = evaluate_checkpoint(scratch, instances=instances)
    right = evaluate_checkpoint(teacher, instances=instances)
    if left["initialization"] != "scratch" or right["initialization"] not in {"pretrain", "load"}:
        raise ValueError("Expected verified scratch and teacher-initialized checkpoints")
    if left["training_seed"] != right["training_seed"]:
        raise ValueError("A paired comparison requires matching training seeds")
    deltas = []
    for a, b in zip(left["records"], right["records"], strict=True):
        if a["environment"] != b["environment"]:
            raise ValueError("Cannot compare mismatched environments")
        deltas.append({"instance_id": a["instance_id"],
            **{metric: b[metric] - a[metric] if a[metric] is not None and b[metric] is not None else None
               for metric in ("return", "rmse", "max_deformation")}})
    return {"scratch": left, "teacher_initialized": right, "paired_deltas": deltas,
        "note": "Teacher minus scratch. One training-seed pair cannot establish statistical significance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--instances")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    result = compare_checkpoints(args.scratch, args.teacher,
        json.loads(Path(args.instances).read_text()) if args.instances else None)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
