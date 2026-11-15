"""Edit CFG must keep the conditional branch: joint_prefix never selects the empty prefix."""

import pytest
import torch

from qwen_latent_cot.u15_target_correction.field import edit_state, rollout_prefix


class _PrefixEditor:
    """A stand-in whose edit only happens when the conditional prefix is used."""

    def __init__(self):
        self.calls = 0

    def image_clean(self, anchor, t, prefix):
        return torch.zeros_like(anchor)

    def state_velocity(self, state, t, s, joint_prefix, base_prediction):
        self.calls += 1
        return torch.full_like(state, 1.5 if joint_prefix == "conditional" else 0.0)

    def guided_state_velocity(self, state, t, s, joint_prefix, empty_prefix, base_prediction, cfg_scale=1.0):
        cond = self.state_velocity(state, t, s, joint_prefix, base_prediction)
        uncond = self.state_velocity(state, t, s, empty_prefix, base_prediction)
        return uncond + float(cfg_scale) * (cond - uncond)


def _prefixes():
    return {"image": "image", "empty": "empty", "joint": "conditional"}


def _anchor():
    return torch.zeros(1, 3, 2, 2)


def test_cfg_keeps_the_instruction():
    editor = _PrefixEditor()
    final, _ = edit_state(
        editor, _anchor(), 0.36, _prefixes(), 4,
        base_prediction=_anchor(), cfg_scale=2.0,
    )
    # Four Euler steps of unit progress at speed 1.5, extrapolated by w=2.
    assert float(final.abs().max()) > 0.0


def test_cfg_scale_one_matches_plain_state_velocity():
    editor = _PrefixEditor()
    plain, _ = edit_state(editor, _anchor(), 0.36, _prefixes(), 4, base_prediction=_anchor())
    scaled, _ = edit_state(
        editor, _anchor(), 0.36, _prefixes(), 4, base_prediction=_anchor(), cfg_scale=1.0
    )
    torch.testing.assert_close(plain, scaled)


def test_passing_the_empty_prefix_as_joint_is_rejected():
    """This exact mistake once zeroed every collected q_K."""
    for fn in (
        lambda ed: edit_state(ed, _anchor(), 0.36, _prefixes(), 4,
                              base_prediction=_anchor(), cfg_scale=2.0, joint_prefix="empty"),
        lambda ed: rollout_prefix(ed, _anchor(), 0.36, _prefixes(), 2, 4,
                                  base_prediction=_anchor(), cfg_scale=2.0, joint_prefix="empty"),
    ):
        with pytest.raises(ValueError, match="empty"):
            fn(_PrefixEditor())


def test_rollout_prefix_applies_the_condition():
    editor = _PrefixEditor()
    state, _ = rollout_prefix(editor, _anchor(), 0.36, _prefixes(), 4, 4,
                              base_prediction=_anchor(), cfg_scale=1.0)
    assert float(state.abs().max()) > 0.0
