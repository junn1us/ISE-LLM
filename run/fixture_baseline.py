"""Terminal-layout baselines. Exact means exhaustive feasible masks, NOT return-optimal trajectories."""
import argparse
from itertools import combinations
import json
from math import comb
from pathlib import Path
import time

import numpy as np

from smartfixture_llm4teach.core.fixture_core import FixtureCore
from validation_runtime import environment_config, source_hashes, write_json
from fixture_search_advanced import bayesian_search, return_oracle, BAYES_SETTINGS


def search_layouts(config, method="exact", metric="max_deformation", calls=1000,
                   seed=0, max_exact_layouts=1000000, beam_width=8):
    if metric not in {"rmse", "max_deformation"}:
        raise ValueError("metric must be rmse or max_deformation")
    if method not in {"exact", "random", "greedy", "local", "beam", "bayes"}:
        raise ValueError("Unknown search method")
    if calls <= 0 or beam_width <= 0:
        raise ValueError("Search budget and beam width must be positive")
    config = environment_config(config)
    core = FixtureCore(**config)
    core.reset()
    n = core.num_anchors
    minimum = core.min_feedback_fixtures
    budget = min(n, core.max_steps, core.max_fixtures if core.max_fixtures > 0 else n)
    count = sum(comb(n, k) for k in range(minimum, budget + 1))
    if not count:
        raise ValueError("No terminal-feasible layouts within horizon/fixture budget")
    if method == "exact" and count > max_exact_layouts:
        raise ValueError(f"Exact search needs {count} evaluations; limit is {max_exact_layouts}")
    if method in {"greedy", "local", "beam"} and minimum:
        raise ValueError("Incremental search requires min_feedback_fixtures=0; use exact/random for gated protocols")
    rng = np.random.default_rng(seed)
    cache, trace, best, worst, per_count = {}, [], None, None, {}
    start = time.perf_counter()
    limit = count if method == "exact" else calls

    def evaluate(nodes):
        nonlocal best, worst
        nodes = tuple(sorted(nodes))
        if nodes in cache:
            return cache[nodes]
        if len(cache) >= limit:
            return None
        if not minimum <= len(nodes) <= budget:
            raise ValueError("Infeasible candidate")
        core.mask[:] = 0
        core.mask[list(nodes)] = 1
        core.sim_apply()
        record = {"nodes": list(nodes), "fixtures": len(nodes),
                  "rmse": core.sim_error(), "max_deformation": core.sim_max_deformation()}
        record["objective"] = record[metric]
        cache[nodes] = record
        if best is None or record["objective"] < best["objective"]:
            best = record
        if worst is None or record["objective"] > worst["objective"]:
            worst = record
        key = str(len(nodes))
        if key not in per_count or record["objective"] < per_count[key]["objective"]:
            per_count[key] = record
        trace.append({"simulator_calls": len(cache), "best_objective": best["objective"]})
        return record

    if method == "exact":
        for k in range(minimum, budget + 1):
            for nodes in combinations(range(n), k):
                evaluate(nodes)
    elif method == "random":
        # Uniform over fixture counts, then uniform subsets conditional on count.
        # A call is charged only on an uncached layout. Bound duplicate proposals.
        for _ in range(calls * 20):
            k = int(rng.integers(minimum, budget + 1))
            evaluate(rng.choice(n, size=k, replace=False).tolist())
            if len(cache) >= min(calls, count):
                break
    elif method == "bayes":
        bayesian_search(evaluate, cache, n, minimum, budget, calls, count, rng)
    else:
        evaluate(())
        frontier = [()]
        for _ in range(budget):
            children = {}
            for nodes in frontier:
                for node in range(n):
                    if node not in nodes:
                        child = tuple(sorted((*nodes, node)))
                        value = evaluate(child)
                        if value is not None:
                            children[child] = value["objective"]
            if not children or len(cache) >= limit:
                break
            frontier = sorted(children, key=children.get)[:beam_width if method == "beam" else 1]
        if method == "local":
            while len(cache) < limit:
                previous = best["objective"]
                nodes = set(best["nodes"])
                neighbors = set()
                for node in range(n):
                    if node in nodes:
                        neighbors.add(tuple(sorted(nodes - {node})))
                    elif len(nodes) < budget:
                        neighbors.add(tuple(sorted(nodes | {node})))
                for old in sorted(nodes):
                    for new in range(n):
                        if new not in nodes:
                            neighbors.add(tuple(sorted(nodes - {old} | {new})))
                for candidate in sorted(neighbors):
                    evaluate(candidate)
                if best["objective"] >= previous:
                    break

    return {"schema_version": 1, "environment": config, "anchor_coordinates_m": core.anchor_xy.tolist(),
        "source_hashes": source_hashes(), "method": method, "metric": metric,
        "scope": "terminal_feasible_layouts_not_cumulative_return", "seed": seed,
        "proven_optimum": method == "exact", "feasible_layout_count": count,
        "simulator_calls": len(cache), "requested_call_budget": None if method == "exact" else calls,
        "seconds": time.perf_counter() - start, "best": best,
        "worst": worst, "best_by_fixture_count": per_count, "trace": trace,
        "range_is_exact": method == "exact",
        "bayes_settings": BAYES_SETTINGS if method == "bayes" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--method", choices=["exact", "random", "greedy", "local", "beam", "bayes", "return-oracle"], default="exact")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-transition-evaluations", type=int, default=250000)
    parser.add_argument("--metric", choices=["rmse", "max_deformation"], default="max_deformation")
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    config = json.loads(Path(args.env_config).read_text())
    result = return_oracle(config, gamma=args.gamma, max_transition_evaluations=args.max_transition_evaluations) if args.method == "return-oracle" else search_layouts(
        config, args.method, args.metric, args.calls, args.seed)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
