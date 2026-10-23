from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ActionSpec:
    kind: str  # "add" | "remove" | "noop"
    node_id: Optional[int] = None


class ActionCatalog:
    """
    Canonical action catalog shared by student + teacher.
    Indexing:
      0..A-1      : add@node_id
      A..2A-1     : remove@node_id
      2A          : noop
    """

    def __init__(self, num_anchors: int):
        if num_anchors <= 0:
            raise ValueError("num_anchors must be > 0")
        self.num_anchors = int(num_anchors)
        self._actions: List[ActionSpec] = []
        for i in range(self.num_anchors):
            self._actions.append(ActionSpec("add", i))
        for i in range(self.num_anchors):
            self._actions.append(ActionSpec("remove", i))
        self._actions.append(ActionSpec("noop", None))

    @property
    def action_dim(self) -> int:
        return len(self._actions)

    def get(self, idx: int) -> ActionSpec:
        return self._actions[int(idx)]

    def index(self, kind: str, node_id: Optional[int] = None) -> int:
        if kind == "noop":
            return 2 * self.num_anchors
        if node_id is None:
            raise ValueError("node_id required for add/remove")
        node_id = int(node_id)
        if not (0 <= node_id < self.num_anchors):
            raise ValueError("node_id out of range")
        if kind == "add":
            return node_id
        if kind == "remove":
            return self.num_anchors + node_id
        raise ValueError(f"Unknown kind: {kind}")

    def describe(self, idx: int) -> str:
        action = self.get(idx)
        if action.kind == "noop":
            return "noop"
        return f"{action.kind}@{action.node_id}"
