"""Frozen synthetic instance splits. Variation is not physical validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from validation_runtime import environment_config, write_json


def instance_fingerprint(config):
    config = environment_config(config)
    return hashlib.sha256(json.dumps({k: v for k, v in config.items()
        if k not in {"seed", "instance_id"}}, sort_keys=True).encode()).hexdigest()


def validate_suite(suite):
    seen_ids, seen_configs = set(), set()
    base = suite["base_environment"]
    if not base.get("include_instance_features"):
        raise ValueError("Generalization suite requires explicit instance observations")
    for split in ("train", "validation", "test", "ood"):
        if not suite[split]:
            raise ValueError(f"Empty split: {split}")
        for item in suite[split]:
            identity = item["instance_id"]
            fingerprint = instance_fingerprint({**base, **item})
            if identity in seen_ids or fingerprint in seen_configs:
                raise ValueError("Duplicate instance or train/test leakage")
            seen_ids.add(identity)
            seen_configs.add(fingerprint)
    return suite


def make_suite(seed=20261013, train=20, validation=5, test=5, ood=5, anchor_mode="twelve"):
    if min(train, validation, test, ood) <= 0:
        raise ValueError("Every split must be nonempty")
    rng = np.random.default_rng(seed)
    base = environment_config(dict(anchor_mode=anchor_mode,
        grid_size=10 if anchor_mode == "grid100" else 9, max_steps=20, max_fixtures=5,
        reward_mode="max_deformation_reduction", reward_clip_min=None,
        reward_clip_max=None, min_feedback_fixtures=0, include_instance_features=True))
    suite = {"schema_version": 1, "suite_seed": seed, "base_environment": base,
        "status": "synthetic_uncalibrated", "selection_rule": "final_checkpoint_fixed_budget",
        "note": "OOD varies spatial load concentration, not an unseen material extrapolation claim."}
    for split, count in (("train", train), ("validation", validation), ("test", test), ("ood", ood)):
        suite[split] = []
        for i in range(count):
            low, high = (0.08, 0.16) if split == "ood" else (0.22, 0.55)
            suite[split].append({"instance_id": f"{split}-{i:03d}",
                "sheet_length_m": float(rng.uniform(1.5, 2.5)),
                "sheet_width_m": float(rng.uniform(1.3, 2.3)),
                "sheet_thickness_m": float(rng.uniform(0.002, 0.004)),
                "material": str(rng.choice(["structural_steel", "composite"])),
                "load_center_x": float(rng.uniform(0.2, 0.8)),
                "load_center_y": float(rng.uniform(0.2, 0.8)),
                "load_width_x": float(rng.uniform(low, high)),
                "load_width_y": float(rng.uniform(low, high)),
                "load_pattern": str(rng.choice(["single", "double"])),
                "load_scale": float(rng.uniform(0.7, 1.4)),
                "influence_width": float(rng.uniform(0.16, 0.28)),
                "support_strength": float(rng.uniform(0.65, 0.9))})
    return validate_suite(suite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--anchor-mode", choices=["stride", "twelve", "grid100"], default="twelve")
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    write_json(args.out, make_suite(seed=args.seed, anchor_mode=args.anchor_mode))


if __name__ == "__main__":
    main()
