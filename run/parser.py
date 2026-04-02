import json
from typing import List

import numpy as np


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    z = logits / temp
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / (np.sum(exp) + 1e-8)


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
    Returns shape (num_candidates,)
    """
    obj = json.loads(raw_json_str)
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
    if logits.shape[0] < num_candidates:
        logits = np.pad(logits, (0, num_candidates - logits.shape[0]), mode="constant")
    elif logits.shape[0] > num_candidates:
        logits = logits[:num_candidates]
    probs = _softmax(logits, temperature=temperature)
    return probs.astype(np.float32)
