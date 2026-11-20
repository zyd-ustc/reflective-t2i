"""Canvas geometry: the shared field of view must keep the scene's shape."""

import json

import pytest
import torch
from PIL import Image

from qwen_latent_cot.u15_target_correction import (
    CURRENT_GEOMETRY_PROTOCOL,
    GEOMETRY_PREPROCESSING,
    GEOMETRY_PROTOCOL,
)
from qwen_latent_cot.u15_target_correction.checkpoint import geometry_gate
from qwen_latent_cot.u15_target_correction.images import _rect_for_ratio, load_canvas, load_pair


def _shape(rect):
    return rect[2] - rect[0], rect[3] - rect[1]


@pytest.mark.parametrize("ratio", [1.0, 1.3, 1.5, 16 / 9, 2.0, 3.0, 4.0])
def test_rect_keeps_the_aspect_ratio_on_a_square_canvas(ratio):
    width = height = 512
    left, top, right, bottom = _rect_for_ratio(ratio, width, height)
    assert (right - left) / (bottom - top) == pytest.approx(ratio, rel=0.01)
    assert 0 <= left < right <= width and 0 <= top < bottom <= height


def test_wide_images_are_not_stretched_to_the_square():
    """The v1 bug computed width * ratio, which clamped every landscape to 512x512."""
    wide, tall = _shape(_rect_for_ratio(16 / 9, 512, 512)), _shape(_rect_for_ratio(9 / 16, 512, 512))
    assert wide == (512, 288)
    assert tall == (288, 512)


@pytest.mark.parametrize("ratio", [0.5, 0.75, 1.0, 2.0, 3.0])
def test_rect_fits_inside_a_non_square_canvas(ratio):
    width, height = 640, 384
    left, top, right, bottom = _rect_for_ratio(ratio, width, height)
    assert 0 <= left < right <= width and 0 <= top < bottom <= height
    assert (right - left) / (bottom - top) == pytest.approx(ratio, rel=0.01)


def test_load_canvas_letters_boxes_a_wide_scene(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (800, 450)).save(path)
    canvas, valid, rect = load_canvas(path, 512, 512, "cpu")
    assert canvas.shape == (1, 3, 512, 512)
    assert _shape(rect) == (512, 288)
    # The area outside the shared field of view stays invalid.
    assert float(valid[0, :, 0, 0]) == 0.0
    assert float(valid[0, :, 256, 256]) == 1.0


def test_load_pair_gives_a_wide_pair_the_same_rect(tmp_path):
    source, target = tmp_path / "s.png", tmp_path / "t.png"
    Image.new("RGB", (1600, 900)).save(source)
    Image.new("RGB", (800, 450)).save(target)
    a, b, valid, rect = load_pair(
        dict(source_image=str(source), target_image=str(target)), 512, 512, "cpu"
    )
    assert _shape(rect) == (512, 288)
    assert a.shape == b.shape == (1, 3, 512, 512)
    assert valid.shape == (1, 1, 512, 512)
    valid_cells = float(valid.sum())
    assert valid_cells == pytest.approx(512 * 288)


def test_geometry_protocol_records_the_preprocessing_change():
    assert GEOMETRY_PROTOCOL == CURRENT_GEOMETRY_PROTOCOL == "shared_field_of_view_v2"
    assert "shared_field_of_view_v1" in GEOMETRY_PREPROCESSING
    assert "shared_field_of_view_v2" in GEOMETRY_PREPROCESSING


def test_geometry_gate_accepts_only_the_current_preprocessing():
    assert geometry_gate({"geometry": CURRENT_GEOMETRY_PROTOCOL}) == CURRENT_GEOMETRY_PROTOCOL
    with pytest.raises(ValueError, match="predates"):
        geometry_gate({"geometry": "shared_field_of_view_v1"}, allow_legacy=False)
    # A checkpoint written before geometry was recorded used the v1 preprocessing.
    with pytest.raises(ValueError, match="shared_field_of_view_v1"):
        geometry_gate({}, allow_legacy=False)
    assert geometry_gate({}, allow_legacy=True) == "shared_field_of_view_v1"


def test_legacy_geometry_requires_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("U15_ALLOW_LEGACY_GEOMETRY", raising=False)
    with pytest.raises(ValueError, match="predates"):
        geometry_gate({"geometry": "shared_field_of_view_v1"})
    monkeypatch.setenv("U15_ALLOW_LEGACY_GEOMETRY", "1")
    assert geometry_gate({"geometry": "shared_field_of_view_v1"}) == "shared_field_of_view_v1"


def _tiny_editor(rt, cfg):
    from qwen_latent_cot.u15_target_correction.model import IsochronicManifoldEditor
    return IsochronicManifoldEditor(rt, cfg).train()


def test_checkpoint_records_geometry_and_reloads_under_it(tmp_path):
    from types import SimpleNamespace

    from qwen_latent_cot.u15_target_correction.checkpoint import (
        canonical_parameters,
        editor_identity,
        load_init_weights,
        save,
    )
    from tests.test_isochronic_manifold import DummyRuntime, make_cfg

    rt, cfg = DummyRuntime(), make_cfg(width=16, height=16, lora_rank=4, activation_checkpointing=False)
    editor = _tiny_editor(rt, cfg)
    opt = torch.optim.AdamW(canonical_parameters(editor).values(), lr=1e-4)
    sampler = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda s: None)
    identity = editor_identity(rt, cfg)
    assert identity["geometry"] == CURRENT_GEOMETRY_PROTOCOL

    ckpt = tmp_path / "stage1"
    save(ckpt, editor, opt, sampler, 1, cfg, identity, rt.device,
         data_version="v", input_domains=["t"], total_steps=1, budget=1)
    stored = json.loads((ckpt / "manifest.json").read_text())
    assert stored["identity"]["geometry"] == CURRENT_GEOMETRY_PROTOCOL

    # A checkpoint carrying the current geometry loads without any opt-in.
    load_init_weights(ckpt, _tiny_editor(rt, cfg), identity)

    # Rewriting the recorded geometry to the v1 preprocessing must be refused.
    stored["identity"]["geometry"] = "shared_field_of_view_v1"
    (ckpt / "manifest.json").write_text(json.dumps(stored))
    from qwen_latent_cot.u15_target_correction.checkpoint import load_init_weights as load
    with pytest.raises(ValueError, match="predates"):
        load(ckpt, _tiny_editor(rt, cfg), identity)
