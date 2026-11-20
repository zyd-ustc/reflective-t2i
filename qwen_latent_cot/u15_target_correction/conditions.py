"""Frozen text conditioning template shared by training and closed-loop inference."""


def compact(text):
    return " ".join(str(text or "").split())


def empty_feedback_prompt():
    """Canonical unconditional prefix input shared by prompt dropout and edit CFG."""
    return None


def feedback_prompt(feedback, *, allow_empty=False):
    """Fixed feedback-only template shared by training and inference."""
    request = compact(feedback)
    if not request:
        if allow_empty:
            return empty_feedback_prompt()
        raise ValueError("A non-empty feedback instruction is required")
    return "Edit request: " + request
