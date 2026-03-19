from typing import Callable, List, Optional

import numpy as np

# nanonical SmartFixture dynamics/state core shared by student and teacher interfaces
from smartfixture_llm4teach.core.fixture_core import FixtureCore

# LLM teacher client relation and default heuristic teacher function 
from smartfixture_llm4teach.teacher.llm_adapter import LLMClient, default_llm_fn

# parse teacher JSON reply into a normalized probability distribution over candidate actions
from smartfixture_llm4teach.teacher.parser import parse_action_distribution

# builds teacher-facing state context and streamline it into a compact prompt payload
from smartfixture_llm4teach.teacher.translator import build_teacher_context, build_teacher_prompt


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
    ):
        self.core = core
        self.temperature = float(temperature)
        self.llm = LLMClient(llm_fn or default_llm_fn)
        self.last_context = None
        self.last_candidate_action_ids: List[int] = []

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
        prompt = build_teacher_prompt(ctx)
        raw = self.llm(prompt)

        probs_subset = parse_action_distribution(
            raw,
            num_candidates=len(candidate_action_ids),
            temperature=self.temperature,
        )

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
