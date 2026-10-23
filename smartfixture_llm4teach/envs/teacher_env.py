from typing import Callable, List, Optional
import json
import time

import numpy as np

from smartfixture_llm4teach.core.fixture_core import FixtureCore
from smartfixture_llm4teach.teacher.llm_adapter import LLMClient, default_llm_fn
from smartfixture_llm4teach.teacher.parser import parse_action_distribution
from smartfixture_llm4teach.teacher.translator import build_teacher_context


def strict_action_distribution(raw, num_candidates, temperature=1.0):
    """Validate actual JSON, candidate indices and exact vector sizes; never repair."""
    obj = json.loads(raw)
    keys = {"best_action_id", "action_probs", "action_logits"}
    if not isinstance(obj, dict) or len(keys.intersection(obj)) != 1:
        raise ValueError("Teacher must return exactly one action representation")
    if num_candidates <= 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid candidates or teacher temperature")
    if "best_action_id" in obj:
        index = obj["best_action_id"]
        if type(index) is not int or not 0 <= index < num_candidates:
            raise ValueError("best_action_id must be a valid candidate index")
        probs = np.zeros(num_candidates, dtype=np.float64)
        probs[index] = 1.0
    else:
        key = "action_probs" if "action_probs" in obj else "action_logits"
        probs = np.asarray(obj[key], dtype=np.float64)
        if probs.shape != (num_candidates,) or not np.isfinite(probs).all():
            raise ValueError("Teacher vector must have exactly one finite value per candidate")
        if key == "action_logits":
            probs = (probs - probs.max()) / temperature
            probs = np.exp(probs)
        elif np.any(probs < 0):
            raise ValueError("Teacher probabilities cannot be negative")
        if not np.isfinite(probs.sum()) or probs.sum() <= 0:
            raise ValueError("Teacher probabilities must have positive finite mass")
        probs = probs / probs.sum()
    return probs.astype(np.float32)


def build_strict_hf_llm_fn(model_id, max_new_tokens=256, temperature=0.2,
                         top_p=0.9, device_map="auto", cache_dir=None,
                         local_files_only=False):
    """HF adapter without the legacy adapter's hidden heuristic substitution.

    Construction/generation/parsing errors reach SmartFixtureTeacherEnv, which owns
    the sole, explicit fallback decision. Imports/model downloads are lazy.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from smartfixture_llm4teach.teacher.llm_adapter import _maybe_disable_transformers_cuda_warmup

    _maybe_disable_transformers_cuda_warmup()
    kwargs = {"cache_dir": cache_dir, "local_files_only": local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, **kwargs)
    model.eval()

    def llm_fn(context):
        prompt = (
            'Return strict JSON only: {"best_action_id": <int>}. Choose the best '
            'fixture action for the objective described in STATE_JSON. best_action_id is the zero-based '
            'INDEX in candidate_actions, not the global action_id. With fewer than '
            'the required fixtures do not consider gravity. STATE_JSON:\n' + context
        )
        if getattr(tokenizer, "chat_template", None):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                tokenize=True, return_tensors="pt", return_dict=True,
            )
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        count = int(inputs["input_ids"].shape[-1])
        generation = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generation.update(temperature=temperature, top_p=top_p)
        with torch.inference_mode():
            output = model.generate(**inputs, **generation)
        tokens = output[0][count:]
        llm_fn.last_usage = {"input_tokens": count, "output_tokens": len(tokens)}
        return tokenizer.decode(tokens, skip_special_tokens=True).strip()

    llm_fn.last_usage = {}
    llm_fn.provenance = {"adapter": "strict_hf", "model_id": model_id,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}
    return llm_fn


class SmartFixtureTeacherEnv:
    """
    Teacher / LLM environment: translates core state to JSON,
    queries LLM, and returns a probability distribution over the SAME
    discrete action set as the student.
    """

    def __init__(
        self,
        core: FixtureCore,
        llm_fn: Optional[Callable[[str], str]] = None,
        temperature: float = 1.0,
        *,
        strict: bool = False,
        allow_fallback: Optional[bool] = None,
    ):
        self.core = core
        self.temperature = float(temperature)
        self.strict = bool(strict)
        self.allow_fallback = not strict if allow_fallback is None else bool(allow_fallback)
        self.using_default = llm_fn is None or llm_fn is default_llm_fn
        if self.strict and self.using_default and not self.allow_fallback:
            raise ValueError("Strict LLM teacher requires llm_fn (or explicit allow_fallback=True)")
        self.llm = LLMClient(llm_fn or default_llm_fn)
        self.last_context = None
        self.last_candidate_action_ids: List[int] = []
        self.last_query = {}

    def reset(self):
        return self.core.reset()

    def step(self, action_idx: int):
        return self.core.step(int(action_idx))

    def get_teacher_distribution(
        self,
        topk_add: int = 5,
        topk_remove: int = 3,
        include_noop: bool = True,
    ) -> np.ndarray:
        candidate_action_ids = self.core.candidate_actions(
            topk_add=topk_add,
            topk_remove=topk_remove,
            include_noop=include_noop,
        )
        ctx = build_teacher_context(self.core, candidate_action_ids)
        started = time.perf_counter()
        fallback = self.using_default
        reason = "no_llm_callable" if fallback else None
        try:
            raw = self.llm(json.dumps(ctx))
            parser = strict_action_distribution if self.strict else parse_action_distribution
            probs_subset = parser(raw, len(candidate_action_ids), self.temperature)
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(f"Teacher query failed without fallback: {exc}") from exc
            fallback = True
            reason = f"{type(exc).__name__}: {exc}"
            raw = default_llm_fn(json.dumps(ctx))
            probs_subset = strict_action_distribution(raw, len(candidate_action_ids), self.temperature)
        usage = getattr(self.llm.llm_fn, "last_usage", {})
        self.last_query = {
            "source": "heuristic" if fallback else "llm",
            "fallback": fallback,
            "fallback_reason": reason,
            "query_seconds": time.perf_counter() - started,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "context": ctx,
            "raw_reply": raw,
            "model_provenance": getattr(self.llm.llm_fn, "provenance", {}),
        }

        # Scatter into full action vector
        full = np.zeros(self.core.action_catalog.action_dim, dtype=np.float32)
        for idx, action_id in enumerate(candidate_action_ids):
            full[int(action_id)] = probs_subset[idx]

        # Normalize to ensure valid distribution
        total = float(full.sum())
        if total > 0:
            full = full / total
        else:
            full[:] = 1.0 / max(self.core.action_catalog.action_dim, 1)

        self.last_context = ctx
        self.last_candidate_action_ids = candidate_action_ids
        return full
