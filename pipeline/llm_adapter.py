import json
import re
from typing import Callable, Dict, List, Optional

import numpy as np


def default_llm_fn(ctx_json_str: str) -> str:
    """
    Stand-in LLM that returns JSON with action_logits over candidate actions.
    Heuristic: prefer add on high sag, remove on low sag, allow noop.
    """
    ctx = json.loads(ctx_json_str)
    candidates: List[Dict[str, object]] = ctx.get("candidate_actions", [])
    logits = np.zeros(len(candidates), dtype=float)

    for i, cand in enumerate(candidates):
        kind = cand.get("type", "noop")
        sag = cand.get("node_sag", 0.0)
        active = cand.get("node_active", 0)
        if kind == "add":
            logits[i] = float(sag) + 0.2
        elif kind == "remove":
            logits[i] = float(-sag) + (0.1 if active else -0.2)
        elif kind == "noop":
            logits[i] = 0.05
        else:
            logits[i] = 0.0

    logits += 0.01 * np.random.randn(len(candidates))
    return json.dumps({"action_logits": logits.tolist()})


class LLMClient:
    """
    Pluggable LLM client. Supply a callable that returns JSON with action_logits/probs.
    """

    def __init__(self, llm_fn: Callable[[str], str] = default_llm_fn):
        self.llm_fn = llm_fn

    def __call__(self, ctx_json_str: str) -> str:
        return self.llm_fn(ctx_json_str)


def _extract_json(text: str) -> str:
    """
    Best-effort JSON extraction from model output.
    Returns the first JSON object found, otherwise raises ValueError.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return match.group(0)


def build_hf_llm_fn(
    model_id: str = "openai/gpt-oss-120b",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    device_map: str = "auto",
    torch_dtype: Optional[str] = "auto",
    trust_remote_code: bool = True,
) -> Callable[[str], str]:
    """
    Build a callable that uses a Hugging Face causal LM to return JSON logits/probs.
    The callable signature matches llm_fn(ctx_json_str) -> raw_json_str.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise ImportError("transformers is required for HF LLM usage") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    def _llm_fn(ctx_json_str: str) -> str:
        prompt = (
            "You are a teacher policy for a discrete-action RL agent. "
            "Return STRICT JSON only. "
            "Schema: {\"action_logits\": [..]} or {\"action_probs\": [..]}. "
            "The list length must equal the number of candidate actions.\n"
            f"STATE_JSON:\n{ctx_json_str}\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        json_str = _extract_json(text)
        return json_str

    return _llm_fn
