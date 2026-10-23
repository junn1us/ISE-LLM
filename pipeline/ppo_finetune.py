"""Shared provenance, collection and raw-policy evaluation. No legacy remapping."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import inspect
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np
import torch

from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.envs.student_env import SmartFixtureStudentEnv


def environment_config(overrides=None):
    params = inspect.signature(FixtureCore).parameters
    config = {name: p.default for name, p in params.items()}
    unknown = set(overrides or {}) - set(config)
    if unknown:
        raise ValueError(f"Unknown environment configuration: {sorted(unknown)}")
    config.update(overrides or {})
    return config


def action_schema(core):
    return [asdict(core.action_catalog.get(i)) for i in range(core.action_catalog.action_dim)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=encode) + "\n")


def source_hashes():
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ("ppo_finetune.py", "distill_student.py", "validation_runtime.py",
        "fixture_baselines.py", "fixture_search_advanced.py", "eval_policy.py", "run_validation_study.py", "validation_instances.py",
        "report_validation_study.py", "validate_physical_reference.py")]
    paths += [
             *sorted((root / "smartfixture_llm4teach").rglob("*.py"))]
    return {str(p.relative_to(root)): sha256(p) for p in paths if p.exists()}


def package_versions():
    versions = {}
    for package in ("numpy", "torch", "gymnasium", "matplotlib", "transformers", "accelerate", "huggingface_hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def archive_sources(destination, hashes):
    """Retain the actual source, not only hashes, even in non-Git workspaces."""
    root = Path(__file__).resolve().parent
    for relative, expected in hashes.items():
        source = root / relative
        if sha256(source) != expected:
            raise RuntimeError(f"Source changed during run setup: {relative}")
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def save_checkpoint(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    write_json(str(path) + ".json", {**metadata, "checkpoint_sha256": sha256(path)})


def read_metadata(path):
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    if not sidecar.exists():
        raise ValueError(f"Missing provenance for {path}; historical checkpoints need explicit audit, not guessed defaults")
    meta = json.loads(sidecar.read_text())
    if meta.get("schema_version") != 1 or meta.get("checkpoint_sha256") != sha256(path):
        raise ValueError("Unsupported checkpoint schema or checkpoint hash mismatch")
    current = source_hashes()
    for key in ("smartfixture_llm4teach/core/fixture_core.py", "smartfixture_llm4teach/core/action_space.py",
                "smartfixture_llm4teach/envs/student_env.py"):
        if meta.get("source_hashes", {}).get(key) != current[key]:
            raise ValueError(f"Environment source differs from checkpoint: {key}; audit/migrate explicitly")
    core = FixtureCore(**meta["environment"])
    if (meta["action_catalog"] != action_schema(core)
            or meta["obs_dim"] != core.reset().size
            or meta["act_dim"] != core.action_catalog.action_dim):
        raise ValueError("Checkpoint environment/action/observation schema mismatch")
    return meta


@contextmanager
def isolated_rng(seed):
    """Pretraining/evaluation must not perturb the subsequent PPO random stream."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng():
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def collect_teacher(env, teacher, n_samples, seed, episode_factory=None):
    if n_samples <= 0:
        raise ValueError("teacher_samples must be positive")
    rng = np.random.default_rng(seed)
    x, p, records = [], [], []
    episode, timestep = 0, 0
    obs = env.reset()
    for _ in range(n_samples):
        probs = np.asarray(teacher.get_teacher_distribution(), dtype=np.float64)
        if (probs.shape != (env.action_space.n,) or not np.isfinite(probs).all()
                or np.any(probs < 0) or not np.isclose(probs.sum(), 1.0, atol=1e-6)):
            raise ValueError("Invalid teacher distribution: shape, values or normalization")
        if env.core.step_count >= env.core.max_steps or not np.isfinite(obs).all():
            raise ValueError("Invalid/out-of-horizon teacher observation")
        x.append(obs.copy())
        p.append(probs.copy())
        records.append({"episode": episode, "timestep": timestep,
                        "instance_id": getattr(env.core, "instance_id", "fixed"),
                        **getattr(teacher, "last_query", {})})
        action = int(rng.choice(len(probs), p=probs / probs.sum()))
        obs, _, done, _ = env.step(action)
        timestep += 1
        if done:
            episode += 1
            timestep = 0
            if episode_factory is not None:
                env, teacher = episode_factory(episode)
            obs = env.reset()
    return np.asarray(x, dtype=np.float32), np.asarray(p, dtype=np.float32), records


def transfer_student(actor, student, observations):
    actor.pi_net.load_state_dict(student.net.state_dict(), strict=True)
    for key, tensor in actor.pi_net.state_dict().items():
        if not torch.equal(tensor, student.net.state_dict()[key]):
            raise RuntimeError("Student-to-actor weight transfer failed")
    with torch.no_grad():
        batch = torch.as_tensor(observations, dtype=torch.float32)
        if not torch.equal(actor.policy_logits(batch), student.net(batch)):
            raise RuntimeError("Student and initialized actor logits differ")


def rollout(actor, config, seed=0, deterministic=True):
    env = SmartFixtureStudentEnv(FixtureCore(**config))
    obs, total, actions = env.reset(), 0.0, []
    with isolated_rng(seed), torch.no_grad():
        for _ in range(env.core.max_steps):
            logits = actor.policy_logits(torch.as_tensor(obs, dtype=torch.float32))
            if logits.shape != (env.action_space.n,):
                raise ValueError("Actor action dimensions do not match environment")
            action = int(logits.argmax()) if deterministic else int(torch.distributions.Categorical(logits=logits).sample())
            obs, reward, done, info = env.step(action)
            actions.append(action)
            total += reward
            if done:
                break
    feasible = bool(info["feedback_available"])
    return {"return": total, "rmse": info["rmse"] if feasible else None,
            "max_deformation": info["max_deformation"] if feasible else None,
            "feasible": feasible, "fixtures": int(env.core.mask.sum()),
            "mask": env.core.mask.astype(int).tolist(), "actions": actions,
            "simulator_calls": len(actions), "controller": "raw_policy",
            "decoding": "greedy" if deterministic else "sample"}
