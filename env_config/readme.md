# Configuration and Environment Logic

This part of the repository defines the fixture-placement environment and converts environment state into teacher actions.

## Data Flow

```text
environment state
    -> translator context/prompt
    -> teacher response (JSON logits)
    -> parsed action probabilities
    -> environment step
```

## Files

| File | Purpose |
|---|---|
| `env_fixture.py` | Standalone `DiscreteFixtureEnv` surrogate for fixture placement on a 9×9 sheet. It configures candidate anchors, applies fixture effects, builds observations, computes RMSE-based rewards, and exposes `reset()`/`step()` methods. |
| `fixture_core.py` | Shared `FixtureCore` transition kernel. It owns sheet geometry, material settings, fixture state, action constraints, deformation calculations, rewards, and step metadata. |
| `teacher_iface.py` | Lightweight teacher adapter containing context generation, a heuristic mock teacher, and probability parsing. Useful for local tests without an external LLM. |
| `translator.py` | Converts an observation, fixture mask, anchor coordinates, and step information into a structured teacher prompt requesting strict JSON action logits. |
| `parse_teacher_reply.py` | Parses the teacher's JSON logits and applies softmax to produce a normalized action-probability vector. |

## Interface Notes

- `env_fixture.py` supports `boundary`, `stride`, `twelve`, and `all` anchor modes.
- Observations contain deformation features; the translator adds fixture and episode context for the teacher.
- Teacher responses must contain an `action_logits` array.
- The standalone environment may include a NOOP action, so confirm whether a given interface expects `A` anchor actions or `A + 1` actions before training.

The newer package-based pipeline uses the canonical modules under `smartfixture_llm4teach/`; these root-level files remain useful for the standalone ablation and compatibility workflows.
