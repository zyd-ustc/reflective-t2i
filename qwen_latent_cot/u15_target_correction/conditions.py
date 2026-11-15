"""The frozen text template shared by training, cache production, and inference."""


def compact(text):
    return " ".join(str(text or "").split())


def empty_edit_prompt():
    """Canonical unconditional edit prefix input shared by dropout and edit CFG."""
    return None


def edit_prompt(edit_request, *, allow_empty=False):
    """The editor's only text condition: the short instruction, nothing else.

    No target scene, no preserve list, no preview caption. Deployment hands the
    diagnoser's short instruction and the state; anything more here would be
    information the closed loop never has, so the editor would learn to read a
    condition that disappears at test time.

    When ``allow_empty=True``, an empty or ``None`` request resolves to
    ``empty_edit_prompt()`` so training prompt dropout and inference edit CFG
    share the exact same unconditional prefix.
    """
    request = compact(edit_request)
    if not request:
        if allow_empty:
            return empty_edit_prompt()
        raise ValueError("An edit request is required")
    return "Edit request: " + request
