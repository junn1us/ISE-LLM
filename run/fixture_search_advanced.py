"""Fixed-hyperparameter GP search and small deterministic trajectory oracle."""
from math import erf
import time
import numpy as np
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from validation_runtime import environment_config, source_hashes


BAYES_SETTINGS = {"kernel": "exp(-Hamming/fixture_budget)", "noise": 1e-6,
    "initial_points": 8, "recent_observation_window": 64,
    "random_acquisition_pool": 128, "acquisition": "expected_improvement"}


def bayesian_search(evaluate, cache, n, minimum, budget, calls, count, rng):
    def propose():
        k = int(rng.integers(minimum, budget + 1))
        return tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))

    while len(cache) < min(calls, count):
        pool = set()
        for _ in range(2560):
            candidate = propose()
            if candidate not in cache:
                pool.add(candidate)
            if len(pool) >= 128:
                break
        if not pool:
            break
        candidates = sorted(pool)
        if len(cache) < 8:
            evaluate(candidates[int(rng.integers(len(candidates)))])
            continue
        observations = list(cache.items())[-64:]
        x = np.zeros((len(observations), n))
        z = np.zeros((len(candidates), n))
        for i, (nodes, _) in enumerate(observations):
            x[i, list(nodes)] = 1
        for i, nodes in enumerate(candidates):
            z[i, list(nodes)] = 1
        y = np.array([record["objective"] for _, record in observations])
        mean, scale = float(y.mean()), max(float(y.std()), 1e-8)
        y = (y - mean) / scale

        def kernel(a, b):
            distance = np.maximum(a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - 2 * a @ b.T, 0)
            return np.exp(-distance / max(budget, 1))

        kxx = kernel(x, x) + np.eye(len(x)) * 1e-6
        kxz = kernel(x, z)
        chol = np.linalg.cholesky(kxx)
        weights = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        mu = kxz.T @ weights
        solved = np.linalg.solve(chol, kxz)
        sigma = np.sqrt(np.maximum(1 - np.sum(solved ** 2, axis=0), 1e-12))
        best_score = min(r["objective"] for r in cache.values())
        improvement = (best_score - mean) / scale - mu
        score = improvement / sigma
        cdf = np.asarray([0.5 * (1 + erf(v / np.sqrt(2))) for v in score])
        ei = improvement * cdf + sigma * np.exp(-score ** 2 / 2) / np.sqrt(2 * np.pi)
        evaluate(candidates[int(np.argmax(ei))])


def return_oracle(config, gamma=0.99, max_transition_evaluations=250000):
    """Exact finite-horizon DP over reachable masks using the core's real rewards.

    Previous reward metrics are determined by mask (or free-sag reference below
    the gate). This assumption is specific to the deterministic surrogate core.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1]")
    config = environment_config(config)
    core = FixtureCore(**config)
    core.reset()
    n, horizon = core.num_anchors, core.max_steps
    if n > 12:
        raise ValueError("Return oracle is restricted to at most 12 anchors")
    start = time.perf_counter()
    current = {0: (0.0, [], [])}
    transition_count = 0
    for t in range(horizon):
        following = {}
        for bits, (value, actions, rewards) in current.items():
            for action in range(core.action_catalog.action_dim):
                if transition_count >= max_transition_evaluations:
                    raise ValueError("DP transition limit exceeded; no optimum certified")
                core.reset()
                core.mask[:] = [(bits >> i) & 1 for i in range(n)]
                core.sim_apply()
                core.step_count = t
                if core.gravity_considered:
                    core.prev_err = core.sim_error()
                    core.prev_max_deformation = core.sim_max_deformation()
                _, reward, _, _ = core.step(action)
                new_bits = sum(1 << int(i) for i in np.flatnonzero(core.mask))
                total = value + gamma ** t * reward
                transition_count += 1
                if new_bits not in following or total > following[new_bits][0]:
                    following[new_bits] = (total, actions + [action], rewards + [reward])
        current = following
    bits, (value, actions, rewards) = max(current.items(), key=lambda entry: entry[1][0])
    return {"schema_version": 1, "environment": config, "source_hashes": source_hashes(),
        "scope": "finite_horizon_discounted_return", "gamma": gamma, "proven_optimum": True,
        "discounted_return": value, "undiscounted_return": sum(rewards),
        "actions": actions, "rewards": rewards, "final_nodes": [i for i in range(n) if bits & (1 << i)],
        "simulator_calls": transition_count, "seconds": time.perf_counter() - start}
