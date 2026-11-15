"""Stage 1/2 image editing and UND-guided T2I feedback."""

import torch

from .diagnosis import review_reflection
from .field import NativeT2I, TailT2I, edit_state, euler, phase_grid, preview_image, segment_times
from .images import load_canvas


def _edit(editor, state, time, instruction, steps, *, edit_cfg_scale=1.0):
    prefixes = editor.prefixes(instruction)
    base = editor.image_clean(state, time, prefixes['image'])
    # edit_state reads prefixes['joint'] as the conditional branch and
    # prefixes['empty'] as the unconditional one. Never hand the empty prefix
    # to joint_prefix: that zeroes the CFG term and drops the instruction.
    return edit_state(
        editor,
        state,
        time,
        prefixes,
        steps,
        base_prediction=base,
        cfg_scale=float(edit_cfg_scale),
    )[0]


def _run_edit(editor, state, time, instruction, steps, edit_cfg_scale):
    if abs(float(edit_cfg_scale) - 1.0) > 1e-6:
        return _edit(editor, state, time, instruction, steps, edit_cfg_scale=float(edit_cfg_scale))
    return _edit(editor, state, time, instruction, steps)


def _noised_source(rt, cfg, image_path, seed):
    image, _, _ = load_canvas(image_path, rt.width, rt.height, rt.device)
    time = float(cfg['read_time'])
    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn(image.shape, device=rt.device, generator=generator) * rt.noise_scale()
    return image, time * image + (1 - time) * noise, time


def _decode(rt, cfg, state):
    grid = phase_grid(rt, cfg)
    native = NativeT2I(rt, None, cfg)
    return grid, native, euler(state, segment_times(grid, float(cfg['read_time']), 1.0), native)


def _tail_field(rt, prompt, cfg, tail=None):
    native = NativeT2I(rt, prompt, cfg)
    return native if tail is None else TailT2I(tail, prompt, cfg)


@torch.no_grad()
def edit_pair(
    rt,
    editor,
    cfg,
    image_path,
    instruction,
    seed,
    load_trained=None,
    trained_editor=None,
    *,
    edit_cfg_scale=None,
    tail=None,
):
    """Edit one shared noised source.

    ``baseline`` continues that state with no transformer. A second editor,
    either ``trained_editor`` or the same editor after ``load_trained``, produces
    ``result``; the first editor produces ``native_result``. Neither is an
    official native image editor or the untouched source.
    """
    if load_trained and trained_editor is not None:
        raise ValueError('use one trained editor')
    cfg_scale = float(cfg.get('edit_cfg_scale', 1.0) if edit_cfg_scale is None else edit_cfg_scale)
    image, state, time = _noised_source(rt, cfg, image_path, seed)
    grid, native, baseline = _decode(rt, cfg, state.clone())
    tail_field = _tail_field(rt, None, cfg, tail)
    native_edited = None
    loaded_step = None
    trained = editor
    if load_trained or trained_editor is not None:
        native_edited = _run_edit(editor, state.clone(), time, instruction, int(cfg['edit_steps']), cfg_scale)
        if load_trained:
            loaded = load_trained()
            loaded_step = loaded.get('step') if isinstance(loaded, dict) else None
        else:
            trained = trained_editor
    edited = _run_edit(trained, state.clone(), time, instruction, int(cfg['edit_steps']), cfg_scale)
    images = {
        'source': image,
        'baseline': baseline,
        'edited_preview': preview_image(edited, time, grid, tail_field),
        'result': euler(edited, segment_times(grid, time, 1.0), tail_field),
    }
    states = {'result': edited}
    if native_edited is not None:
        images['native_edited_preview'] = preview_image(native_edited, time, grid, native)
        images['native_result'] = euler(native_edited, segment_times(grid, time, 1.0), native)
        states['native_result'] = native_edited
    return images, states, dict(
        mode='edit', instruction=instruction, seed=seed, read_time=time,
        edit_cfg_scale=cfg_scale, tail_generator=tail is not None,
        continuation='tail generator T2I' if tail is not None else 'unconditional native T2I',
        state_initialization='linear source and noise',
        baseline='noised source continued without the state transformer',
        loaded_checkpoint_step=loaded_step)


@torch.no_grad()
def edit_image(rt, editor, cfg, image_path, instruction, seed, *, edit_cfg_scale=None, tail=None):
    images, states, record = edit_pair(
        rt, editor, cfg, image_path, instruction, seed,
        edit_cfg_scale=edit_cfg_scale, tail=tail,
    )
    return images, states['result'], record


@torch.no_grad()
def t2i_feedback(rt, editor, cfg, prompt, seed, *, edit_cfg_scale=None, tail=None):
    cfg_scale = float(cfg.get('edit_cfg_scale', 1.0) if edit_cfg_scale is None else edit_cfg_scale)
    grid = phase_grid(rt, cfg)
    time = float(cfg['read_time'])
    native = NativeT2I(rt, prompt, cfg)
    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * rt.noise_scale()
    state = euler(noise, segment_times(grid, 0, time), native)
    preview = preview_image(state, time, grid, native)
    review = review_reflection(rt, preview, prompt, cfg)['reflection']
    if review['outcome'] in ('failed_parse', 'failed_unconverted'):
        raise ValueError('UND reflection could not produce a decision: ' + review['raw'])
    edited = None
    images = {'preview': preview}
    tail_field = native
    if review['outcome'] == 'edit':
        edited = _run_edit(editor, state, time, review['instruction'], int(cfg['edit_steps']), cfg_scale)
        tail_field = _tail_field(rt, prompt, cfg, tail)
        images['edited_preview'] = preview_image(edited, time, grid, tail_field)
    result = euler(edited if edited is not None else state, segment_times(grid, time, 1), tail_field)
    images['result'] = result
    return images, edited, dict(mode='t2i', prompt=prompt, seed=seed, read_time=time,
                                edit_cfg_scale=cfg_scale, tail_generator=tail is not None,
                                reflection=review)
