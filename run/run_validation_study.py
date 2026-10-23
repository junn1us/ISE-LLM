"""Matched fixed-budget synthetic ablations. Real LLM runs are explicit and fail closed."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from eval_policy import evaluate_checkpoint
from fixture_baselines import search_layouts
from ppo_finetune import ppo_finetune
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv
from smartfixture_llm4teach.envs.teacher_env import SmartFixtureTeacherEnv, build_strict_hf_llm_fn
from validation_instances import make_suite, validate_suite
from validation_runtime import isolated_rng, source_hashes, write_json, sha256, archive_sources


def teacher_episode(config, seed, llm_fn=None):
    core = FixtureCore(**config)
    env = SmartFixtureStudentEnv(core)
    teacher = SmartFixtureTeacherEnv(core, llm_fn=llm_fn, strict=llm_fn is not None,
                                      allow_fallback=llm_fn is None)
    env.reset()
    total, queries, actions = 0.0, [], []
    started = time.perf_counter()
    with isolated_rng(seed):
        for _ in range(core.max_steps):
            action = int(np.argmax(teacher.get_teacher_distribution()))
            queries.append(teacher.last_query.copy())
            actions.append(action)
            _, reward, done, info = env.step(action)
            total += reward
            if done:
                break
    return {"return": total, "max_deformation": info["max_deformation"], "rmse": info["rmse"],
        "fixtures": int(core.mask.sum()), "mask": core.mask.astype(int).tolist(),
        "actions": actions, "simulator_calls": len(actions), "teacher_queries": queries,
        "seconds": time.perf_counter() - started, "controller": "teacher_only", "feasible": info["feedback_available"]}


def paired_summary(records, seeds):
    """Average over the same held-out instances within each independent training seed."""
    summary = []
    rng = np.random.default_rng(910)
    for split in ("validation", "test", "ood"):
        conditions = sorted({r["condition"] for r in records if r["split"] == split and r["phase"] == "final"})
        for condition in conditions:
            if condition == "scratch":
                continue
            differences = []
            for seed in seeds:
                subsets = {}
                for label in ("scratch", condition):
                    subsets[label] = {r["instance_id"]: r for r in records
                        if r["condition"] == label and r["split"] == split
                        and r["training_seed"] == seed and r["phase"] == "final"}
                a, b = subsets["scratch"], subsets[condition]
                if a.keys() != b.keys() or not a:
                    continue
                differences.append(float(np.mean([b[i]["normalized_max_deformation"] - a[i]["normalized_max_deformation"] for i in a])))
            ci = None
            if len(differences) >= 2:
                bootstrap = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
                ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
            if differences:
                summary.append({"split": split, "condition": condition,
                    "paired_training_seed_differences": differences,
                    "mean_delta_normalized_max_deformation": float(np.mean(differences)),
                    "bootstrap_95_interval_over_training_seeds": ci,
                    "note": "Negative favors condition. Small-seed intervals are exploratory; instances are fixed, not population uncertainty."})
    return summary


def run_study(out, suite, seeds=(16, 17, 18), iters=20, episodes_per_iter=5,
              teacher_samples=256, distill_steps=300, model_id=None, baseline_calls=300,
              include_teacher_only=True, llm_fn=None):
    validate_suite(suite)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Training seeds must be nonempty and unique")
    if iters < 0 or episodes_per_iter <= 0 or teacher_samples <= 0 or distill_steps <= 0:
        raise ValueError("Invalid training budget")
    if llm_fn is not None and not model_id:
        raise ValueError("Injected LLM callable requires explicit model identity")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "suite.json", suite)
    base = suite["base_environment"]
    if base["reward_mode"] != "max_deformation_reduction" or base["min_feedback_fixtures"] != 0:
        raise ValueError("This study protocol requires ungated maximum-deformation reduction")
    manifest = {"schema_version": 1, "status": "running", "source_hashes": source_hashes(),
        "suite_sha256": sha256(root / "suite.json"), "seeds": list(seeds), "iterations": iters,
        "episodes_per_iter": episodes_per_iter, "teacher_samples": teacher_samples,
        "distill_steps": distill_steps, "model_id": model_id,
        "llm_evidence": "requested" if model_id else "not_run",
        "physical_validation": "not_performed", "checkpoint_selection": "final_fixed_budget",
        "teacher_filtering": "candidate shortlist only; no simulator-based safe filtering"}
    write_json(root / "study_manifest.json", manifest)
    archive_sources(root / "source_snapshot", manifest["source_hashes"])
    conditions = ["scratch", "heuristic"] + (["llm"] if model_id else [])
    records, baselines, curves = [], [], []
    started = time.perf_counter()
    injected_callable = llm_fn is not None
    if model_id and llm_fn is None:
        llm_fn = build_strict_hf_llm_fn(model_id)
    manifest["llm_model_load_seconds"] = time.perf_counter() - started

    def add_record(result, condition, phase, split, item, seed):
        cfg = {**base, **item}
        free = FixtureCore(**cfg)._free_sag
        if result["max_deformation"] is None or not result.get("feasible", True):
            raise ValueError("Infeasible result in ungated protocol")
        record = {**result, "condition": condition, "phase": phase, "split": split,
            "instance_id": item["instance_id"], "training_seed": seed,
            "normalized_max_deformation": result["max_deformation"] / float(np.max(free))}
        records.append(record)

    for seed in seeds:
        for condition in conditions:
            path = root / f"{condition}_s{seed}.pt"
            ppo_finetune(num_iters=iters, episodes_per_iter=episodes_per_iter, seed=seed,
                out_name=str(path), init_mode="scratch" if condition == "scratch" else "pretrain",
                teacher_source="llm" if condition == "llm" else "heuristic",
                teacher_samples=teacher_samples, distill_steps=distill_steps,
                hf_model_id=model_id if condition == "llm" else "",
                llm_fn=llm_fn if condition == "llm" else None,
                core_config=base, training_instances=suite["train"], save_iter_snapshots=True)
            initial = path.with_name(path.stem + "_artifacts") / "initial_actor.pt"
            for phase, checkpoint in (("initial", initial), ("final", path)):
                for split in ("validation", "test", "ood"):
                    evaluation = evaluate_checkpoint(checkpoint, instances=suite[split])
                    for item, result in zip(suite[split], evaluation["records"], strict=True):
                        add_record({**result, "checkpoint_sha256": evaluation["checkpoint_sha256"]},
                                   condition, phase, split, item, seed)
            # Validation curves only: never use test performance for checkpoint choice.
            for iteration in range(iters + 1):
                checkpoint = initial if iteration == 0 else path.with_name(f"{path.stem}_iter{iteration:03d}.pt")
                evaluation = evaluate_checkpoint(checkpoint, instances=suite["validation"])
                ratios = [r["max_deformation"] / float(np.max(FixtureCore(**r["environment"])._free_sag))
                          for r in evaluation["records"]]
                curves.append({"condition": condition, "training_seed": seed, "iteration": iteration,
                    "ppo_transitions": iteration * episodes_per_iter * base["max_steps"],
                    "mean_normalized_max_deformation": float(np.mean(ratios))})
            write_json(root / "evaluation_records.json", records)

    for split in ("validation", "test", "ood"):
        for item in suite[split]:
            cfg = {**base, **item}
            methods = ["random", "greedy", "local", "beam", "bayes"]
            if FixtureCore(**cfg).num_anchors <= 12:
                methods.insert(0, "exact")
            for method in methods:
                result = search_layouts(cfg, method=method, calls=baseline_calls, seed=seeds[0])
                baselines.append({"split": split, "instance_id": item["instance_id"], **result})
            if include_teacher_only:
                for condition in conditions[1:]:
                    for seed in seeds:
                        result = teacher_episode(cfg, seed, llm_fn if condition == "llm" else None)
                        add_record(result, condition + "_teacher_only", "final", split, item, seed)

    exact = {(r["split"], r["instance_id"]): r["best"]["objective"] for r in baselines if r["proven_optimum"]}
    for record in records:
        optimum = exact.get((record["split"], record["instance_id"]))
        record["terminal_optimality_gap"] = None if optimum is None else record["max_deformation"] - optimum
    write_json(root / "evaluation_records.json", records)
    write_json(root / "baselines.json", baselines)
    write_json(root / "validation_curves.json", curves)
    write_json(root / "paired_summary.json", paired_summary(records, seeds))
    columns = ["condition", "phase", "split", "instance_id", "training_seed", "return", "rmse",
        "max_deformation", "normalized_max_deformation", "fixtures", "terminal_optimality_gap"]
    with (root / "evaluation_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    manifest.update(status="completed", seconds=time.perf_counter() - started,
        llm_evidence="injected_callable_unverified" if injected_callable else "strict_hf_calls" if model_id else "not_run",
        scientific_status="smoke_only" if iters < 20 or len(seeds) < 5 else "requires_scientific_review",
        evaluation_records_sha256=sha256(root / "evaluation_records.json"),
        validation_curves_sha256=sha256(root / "validation_curves.json"),
        cost_accounting={"heldout_rollout_transitions": sum(r["simulator_calls"] for r in records),
            "validation_curve_transitions": len(curves) * len(suite["validation"]) * base["max_steps"],
            "baseline_layout_evaluations": sum(r["simulator_calls"] for r in baselines),
            "teacher_only_queries": sum(len(r.get("teacher_queries", [])) for r in records),
            "note": "Training/collection/diagnostic counts and times are in each checkpoint manifest; resets and neural-network compute are not layout evaluations."})
    write_json(root / "study_manifest.json", manifest)
    return {"records": records, "baselines": baselines, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suite", help="Frozen suite JSON; defaults to a generated 20/5/5/5 split")
    parser.add_argument("--seeds", default="16,17,18,19,20")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--episodes-per-iter", type=int, default=5)
    parser.add_argument("--teacher-samples", type=int, default=256)
    parser.add_argument("--distill-steps", type=int, default=300)
    parser.add_argument("--hf-model-id", help="Explicitly opt into real model loading and strict LLM calls")
    parser.add_argument("--baseline-calls", type=int, default=300)
    parser.add_argument("--skip-teacher-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny synthetic plumbing check, not manuscript evidence")
    args = parser.parse_args()
    if args.smoke:
        suite = make_suite(train=2, validation=1, test=1, ood=1, anchor_mode="stride")
        suite["base_environment"]["max_steps"] = 4
        suite["base_environment"]["max_fixtures"] = 3
        seeds, iters, episodes, samples, steps = (16,), 1, 2, 8, 2
    else:
        suite = json.loads(Path(args.suite).read_text()) if args.suite else make_suite()
        seeds = tuple(int(s) for s in args.seeds.split(","))
        iters, episodes, samples, steps = args.iters, args.episodes_per_iter, args.teacher_samples, args.distill_steps
    run_study(args.out, suite, seeds, iters, episodes, samples, steps,
        args.hf_model_id, args.baseline_calls, not args.skip_teacher_only)


if __name__ == "__main__":
    main()
