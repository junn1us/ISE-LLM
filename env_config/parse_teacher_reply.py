import json
import numpy as np

def parse_teacher_reply(raw_text):
    """
    raw_text: string from LLM completion (should be strict JSON)
    returns probs: np.array shape [A+1] with sum ~1
    """
    # basic safe parse
    data = json.loads(raw_text)
    logits = np.array(data["action_logits"], dtype=np.float32)
    # softmax
    z = logits - logits.max()
    probs = np.exp(z) / np.sum(np.exp(z))
    return probs  # shape [A+1]
