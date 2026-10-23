import json
import re
from typing import List

import numpy as np


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    z = logits / temp
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / (np.sum(exp) + 1e-8)


def _uniform_probs(num_candidates: int) -> np.ndarray:
    count = max(int(num_candidates), 1)
    return np.ones(count, dtype=np.float32) / count


def _load_teacher_json(raw_json_str: str):
    # Strip leading/trailing whitespace
    s = raw_json_str.strip()
    # Try straight JSON parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Model may have received '{"best_action_id":' as prefix and returned '42}'
    # Try completing it
    try:
        return json.loads('{"best_action_id":' + s)
    except json.JSONDecodeError:
        pass
    # Extract any {...} block
    match = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Extract bare integer — treat as best_action_id
    int_match = re.search(r"\b(\d+)\b", s)
    if int_match:
        return {"best_action_id": int(int_match.group(1))}
    raise ValueError(f"Cannot parse LLM output: {s[:120]!r}")


def parse_action_distribution(
    raw_json_str: str,
    num_candidates: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Parse teacher JSON into a probability vector over candidate actions.
    Accepts either:
      {"action_logits": [...]}
      {"action_probs":  [...]}  (will be normalized)
      {"best_action_id": <int>} (interpreted as one-hot over candidate index)
    Returns shape (num_candidates,)
    """
    try:
        obj = _load_teacher_json(raw_json_str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _uniform_probs(num_candidates)

    if "best_action_id" in obj:
        probs = np.zeros(max(num_candidates, 1), dtype=np.float32)
        try:
            best_idx = int(obj["best_action_id"])
        except (TypeError, ValueError):
            return _uniform_probs(num_candidates)
        if 0 <= best_idx < num_candidates:
            probs[best_idx] = 1.0
            return probs
        return _uniform_probs(num_candidates)

    if "action_probs" in obj:
        probs = np.array(obj["action_probs"], dtype=float)
        if probs.shape[0] < num_candidates:
            probs = np.pad(probs, (0, num_candidates - probs.shape[0]), mode="constant")
        elif probs.shape[0] > num_candidates:
            probs = probs[:num_candidates]
        total = float(np.sum(probs))
        if total <= 0:
            probs = np.ones(num_candidates, dtype=float) / max(num_candidates, 1)
        else:
            probs = probs / total
        return probs.astype(np.float32)

    logits = np.array(obj.get("action_logits", []), dtype=float)
    if logits.size == 0:
        return _uniform_probs(num_candidates)
    if logits.shape[0] < num_candidates:
        logits = np.pad(logits, (0, num_candidates - logits.shape[0]), mode="constant")
    elif logits.shape[0] > num_candidates:
        logits = logits[:num_candidates]
    probs = _softmax(logits, temperature=temperature)
    return probs.astype(np.float32)
