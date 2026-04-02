import json
import os
import re
import time
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
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> Callable[[str], str]:
    """
    Build a callable that uses a Hugging Face causal LM to return JSON logits/probs.
    The callable signature matches llm_fn(ctx_json_str) -> raw_json_str.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise ImportError("transformers is required for HF LLM usage") from exc

    try:
        import torch
    except Exception:
        torch = None

    print(f"[HF LLM] loading model: {model_id}")
    if torch is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"[HF LLM] CUDA visible GPUs: {gpu_count} -> {gpu_names}")
    else:
        print("[HF LLM] CUDA not available; using CPU or non-CUDA backend")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if hasattr(model, "hf_device_map"):
        print(f"[HF LLM] device map: {model.hf_device_map}")
    elif hasattr(model, "device"):
        print(f"[HF LLM] single device: {model.device}")

    def _llm_fn(ctx_json_str: str) -> str:
        prompt = (
            "You are a teacher policy for a discrete-action RL agent. "
            "Return STRICT JSON only. "
            "Schema: {\"action_logits\": [..]} or {\"action_probs\": [..]}. "
            "The list length must equal the number of candidate actions.\n"
            f"STATE_JSON:\n{ctx_json_str}\n"
        )

        # gpt-oss model cards recommend using chat templates (Harmony format).
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
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


# ---------------------------------------------------------------------------
# Memory-efficient MoE patch  (required for ≤32 GB unified memory)
# ---------------------------------------------------------------------------

def _patch_mlp_for_low_memory(chunk_size: int = 4):
    """
    Monkey-patch ``gpt_oss.torch.model.MLPBlock.forward`` so that the huge
    expert-weight fancy-indexing is done in small token-chunks instead of the
    full sequence at once.

    Without this patch the MoE layer allocates
        seq_len × 4 × 5760 × 2880 × 2 bytes
    which is ~132 MB **per token**.  A 1 500-token prompt would need ~197 GiB.

    With chunk_size=4 the peak buffer drops to ~530 MB regardless of seq_len.
    """
    import torch
    import torch.distributed as dist
    from gpt_oss.torch.model import MLPBlock, swiglu

    def _chunked_forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        seq_len = t.shape[0]
        out = torch.zeros_like(t)

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            ti = t[start:end]
            ei = expert_indices[start:end]
            ew = expert_weights[start:end]

            # MLP #1
            m1w = self.mlp1_weight[ei]
            m1b = self.mlp1_bias[ei]
            ti_e = torch.einsum("beck,bk->bec", m1w, ti) + m1b
            ti_e = swiglu(ti_e, limit=self.swiglu_limit)

            # MLP #2
            m2w = self.mlp2_weight[ei]
            m2b = self.mlp2_bias[ei]
            ti_e = torch.einsum("beck,bek->bec", m2w, ti_e)
            if self.world_size > 1:
                dist.all_reduce(ti_e, op=dist.ReduceOp.SUM)
            ti_e += m2b

            # Weighted sum of experts
            out[start:end] = torch.einsum("bec,be->bc", ti_e, ew)

        return x + out

    MLPBlock.forward = _chunked_forward
    print(f"[patch] MLPBlock.forward → chunked  (chunk_size={chunk_size}, "
          f"peak MoE buf ≈ {chunk_size * 4 * 5760 * 2880 * 2 / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Fix for MPS: torch.pow(scalar, mps_tensor) hangs on Apple Silicon
# ---------------------------------------------------------------------------

def _patch_rope_for_mps():
    """
    Monkey-patch ``RotaryEmbedding._compute_concentration_and_inv_freq`` to
    compute the expensive ``self.base ** (...)`` on CPU and then move the result
    to MPS.  The MPS backend's scalar-tensor ``pow`` kernel is catastrophically
    slow (minutes) and makes the model unusable.
    """
    import torch
    from gpt_oss.torch.model import RotaryEmbedding
    import math

    _orig = RotaryEmbedding._compute_concentration_and_inv_freq

    def _fast_compute(self):
        # Compute on CPU to avoid MPS pow bug
        target_device = self.device
        self.device = torch.device("cpu")
        concentration, inv_freq = _orig(self)
        self.device = target_device
        # Move result to target device
        inv_freq = inv_freq.to(target_device)
        return concentration, inv_freq

    RotaryEmbedding._compute_concentration_and_inv_freq = _fast_compute
    print("[patch] RotaryEmbedding → CPU-computed inv_freq (MPS pow fix)")


# ---------------------------------------------------------------------------
# KV-cache patched generator  (makes generation O(N) instead of O(N²))
# ---------------------------------------------------------------------------

def _make_kvcache_generator(checkpoint_dir: str, device):
    """
    Build a ``generate()`` function that uses per-layer KV caches so that
    each new token only does a single-token forward pass through the model.

    Includes: pre-computed RoPE tables, GQA-aware attention, sink tokens,
    sliding window, and chunked MoE.
    """
    import math
    import torch
    import torch.distributed as dist
    from gpt_oss.torch.model import (
        Transformer, _apply_rotary_emb, swiglu,
    )

    model = Transformer.from_checkpoint(checkpoint_dir, device=device)
    n_layers = len(model.block)

    # Pre-compute RoPE tables once (max 512 positions covers all prompts + generation)
    MAX_SEQ = 512
    _rope_cache = {}
    for layer_idx, block in enumerate(model.block):
        cos, sin = block.attn.rope._compute_cos_sin(MAX_SEQ)
        _rope_cache[layer_idx] = (cos, sin)
    print(f"[kv-cache] RoPE tables pre-computed for {MAX_SEQ} positions")

    # ---- per-call state ----
    class _KVState:
        def __init__(self):
            self.caches = [{"k": None, "v": None} for _ in range(n_layers)]
            self.seq_len = 0

    def _attn_forward_kv(attn, layer_idx, x_slice, kv, seq_offset):
        """AttentionBlock forward for new tokens with KV cache + pre-computed RoPE."""
        t = attn.norm(x_slice)
        qkv = attn.qkv(t)

        n_new = t.shape[0]
        n_kv = attn.num_key_value_heads
        q_mult = attn.num_attention_heads // n_kv
        hd = attn.head_dim

        q = qkv[:, :attn.num_attention_heads * hd].contiguous()
        kv_start = attn.num_attention_heads * hd
        kv_mid = (attn.num_attention_heads + n_kv) * hd
        kv_end = (attn.num_attention_heads + 2 * n_kv) * hd
        k_new = qkv[:, kv_start:kv_mid].contiguous()
        v_new = qkv[:, kv_mid:kv_end].contiguous()

        # Reshape
        q = q.view(n_new, n_kv, q_mult, hd)
        k_new = k_new.view(n_new, n_kv, hd)
        v_new = v_new.view(n_new, n_kv, hd)

        # RoPE using pre-computed tables (no recomputation!)
        cos_all, sin_all = _rope_cache[layer_idx]
        cos_new = cos_all[seq_offset:seq_offset + n_new]
        sin_new = sin_all[seq_offset:seq_offset + n_new]

        q_flat = q.view(n_new, -1, hd)
        q_flat = _apply_rotary_emb(q_flat, cos_new, sin_new)
        q = q_flat.view(n_new, n_kv, q_mult, hd)

        k_flat = k_new.view(n_new, -1, hd)
        k_flat = _apply_rotary_emb(k_flat, cos_new, sin_new)
        k_new = k_flat.view(n_new, n_kv, hd)

        # Append to KV cache
        if kv["k"] is None:
            kv["k"] = k_new
            kv["v"] = v_new
        else:
            kv["k"] = torch.cat([kv["k"], k_new], dim=0)
            kv["v"] = torch.cat([kv["v"], v_new], dim=0)

        K_all = kv["k"]
        V_all = kv["v"]
        S_total = K_all.shape[0]

        # Expand K, V for GQA
        K_exp = K_all[:, :, None, :].expand(-1, -1, q_mult, -1)
        V_exp = V_all[:, :, None, :].expand(-1, -1, q_mult, -1)

        # Attention scores
        QK = torch.einsum("qhmd,khmd->hmqk", q, K_exp)
        QK *= attn.sm_scale

        # Causal + sliding window mask
        q_pos = torch.arange(seq_offset, seq_offset + n_new, device=x_slice.device)
        k_pos = torch.arange(S_total, device=x_slice.device)
        causal = q_pos[:, None] < k_pos[None, :]
        if attn.sliding_window > 0:
            causal = causal | ((q_pos[:, None] - k_pos[None, :]) >= attn.sliding_window)
        QK = QK.masked_fill(causal[None, None, :, :], -float("inf"))

        # Sink tokens
        sinks = attn.sinks.reshape(n_kv, q_mult, 1, 1).expand(-1, -1, n_new, 1)
        W = torch.softmax(torch.cat([QK, sinks], dim=-1), dim=-1)
        W = W[..., :-1]

        attn_out = torch.einsum("hmqk,khmd->qhmd", W, V_exp).reshape(n_new, -1)
        attn_out = attn.out(attn_out)
        return x_slice + attn_out

    @torch.inference_mode()
    def generate(prompt_tokens, stop_tokens, temperature=0.0,
                 max_tokens=30):
        """Generate with KV cache — only 1 token through the model per step."""
        state = _KVState()

        # ---- prefill: run full prompt through model ----
        tokens_t = torch.as_tensor(prompt_tokens, dtype=torch.int32, device=device)
        x = model.embedding(tokens_t)
        for layer_idx, block in enumerate(model.block):
            x = _attn_forward_kv(block.attn, layer_idx, x, state.caches[layer_idx], 0)
            x = block.mlp(x)
        state.seq_len = len(prompt_tokens)
        x = model.norm(x)
        logits = model.unembedding(x)[-1]

        tokens = list(prompt_tokens)
        generated = 0

        while max_tokens == 0 or generated < max_tokens:
            if temperature == 0.0:
                tok = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                tok = torch.multinomial(probs, num_samples=1).item()
            tokens.append(tok)
            generated += 1
            yield tok
            if tok in stop_tokens:
                break

            # ---- decode step: only the new token ----
            tok_t = torch.as_tensor([tok], dtype=torch.int32, device=device)
            x = model.embedding(tok_t)
            for layer_idx, block in enumerate(model.block):
                x = _attn_forward_kv(block.attn, layer_idx, x,
                                     state.caches[layer_idx], state.seq_len)
                x = block.mlp(x)
            state.seq_len += 1
            x = model.norm(x)
            logits = model.unembedding(x)[-1]

    print(f"[kv-cache] generator ready  ({n_layers} layers, device={device})")
    return generate


# ---------------------------------------------------------------------------
# Native gpt-oss-20b adapter  (openai_harmony + gpt_oss.torch.model)
# ---------------------------------------------------------------------------

def build_gptoss_llm_fn(
    checkpoint_dir: str,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    log_dir: Optional[str] = None,
) -> Callable[[str], str]:
    """
    Build a teacher LLM callable that uses the locally downloaded gpt-oss-20b
    checkpoint via the native ``gpt_oss`` package (pure-PyTorch, MPS-compatible).

    Uses a custom KV-cache patched generator for O(N) generation instead of
    the upstream O(N²) implementation.

    The returned callable has the signature::

        llm_fn(ctx_json_str: str) -> raw_json_str

    It also logs every (prompt, response) pair to ``<log_dir>/llm_log.jsonl``.
    """
    import torch
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    # ---- Apply memory-efficient MoE patch BEFORE loading the model ----
    _patch_mlp_for_low_memory(chunk_size=4)

    # ---- Fix MPS pow() hang in RoPE ----
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _patch_rope_for_mps()

    from gpt_oss.torch.utils import init_distributed
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        Role,
        load_harmony_encoding,
    )

    device = init_distributed()
    print(f"[gpt-oss-20b] loading with KV-cache generator from {checkpoint_dir} on {device}")
    generate_fn = _make_kvcache_generator(checkpoint_dir, device)
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Prepare log file
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        _log_path = os.path.join(log_dir, "llm_log.jsonl")
    else:
        _log_path = None

    _call_counter = [0]

    def _llm_fn(ctx_json_str: str) -> str:
        _call_counter[0] += 1
        call_id = _call_counter[0]

        # ---- build COMPACT prompt ----
        # Strip the full 100-anchor state down to just what the model needs:
        # key stats + candidate actions.  Keeps prompt < 200 tokens.
        ctx = json.loads(ctx_json_str)
        cands = ctx.get("candidate_actions", [])
        n_cands = len(cands)

        lines = [
            "Fixture-placement teacher. Return ONLY JSON.",
            f"stage={ctx.get('stage',0)} budget={ctx.get('budget_left',5)} "
            f"dmax={ctx.get('d_max_mm',0):.3f} dmean={ctx.get('d_mean_mm',0):.3f}",
            "Candidates:",
        ]
        for ca in cands:
            if ca["type"] == "noop":
                lines.append(f"  {ca['action_id']}:noop")
            else:
                sag = ca.get("node_sag") or 0.0
                active = ca.get("node_active", 0)
                lines.append(
                    f"  {ca['action_id']}:{ca['type']} n{ca['node_id']} "
                    f"sag={sag:.3f} act={active}"
                )
        lines.append(
            f'Reply: {{"action_logits":[{n_cands} floats, higher=better]}}'
        )
        compact_prompt = "\n".join(lines)

        # Harmony-encode (user message only — skip system to save tokens)
        user_msg = Message.from_role_and_content(Role.USER, compact_prompt)
        conversation = Conversation.from_messages([user_msg])
        prompt_tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        stop_tokens = encoding.stop_tokens_for_assistant_actions()

        # ---- generate ----
        t0 = time.time()
        output_tokens: List[int] = []
        for tok in generate_fn(
            prompt_tokens,
            stop_tokens,
            temperature=temperature,
            max_tokens=max_new_tokens,
        ):
            output_tokens.append(tok)
        elapsed = time.time() - t0
        raw_text = encoding.decode(output_tokens)

        # ---- log ----
        if _log_path:
            entry = {
                "call_id": call_id,
                "prompt": compact_prompt,
                "raw_output": raw_text,
                "n_prompt_tokens": len(prompt_tokens),
                "n_output_tokens": len(output_tokens),
                "elapsed_s": round(elapsed, 2),
            }
            with open(_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        print(
            f"[gpt-oss-20b] call #{call_id}: "
            f"{len(prompt_tokens)} prompt → {len(output_tokens)} out  "
            f"({elapsed:.1f}s)  {raw_text[:80]!r}"
        )

        # ---- parse JSON ----
        try:
            json_str = _extract_json(raw_text)
        except ValueError:
            # Fallback: return uniform logits so training doesn't crash
            json_str = json.dumps({"action_logits": [0.0] * max(n_cands, 1)})
            print(f"[gpt-oss-20b] WARNING #{call_id}: no JSON, uniform fallback")
        return json_str

    return _llm_fn
