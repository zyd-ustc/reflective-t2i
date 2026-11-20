"""CLI for training and running the strict isochronic manifold-transition operator."""

import argparse
import json
from pathlib import Path

from .config import load_config


def _model_args(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--distributed", action="store_true")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    _model_args(train)
    train.add_argument("--linear", nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--init-checkpoint")
    train.add_argument("--steps", type=int)

    for name in ("edit", "t2i"):
        command = commands.add_parser(name)
        _model_args(command)
        command.add_argument("--output", required=True)
        command.add_argument("--seed", type=int, default=17)
        command.add_argument("--checkpoint-mode", choices=("trained", "native"), required=True)
        command.add_argument("--checkpoint")
        command.add_argument("--edit-cfg-scale", type=float, default=None)
        if name == "edit":
            command.add_argument("--image", required=True)
            command.add_argument("--instruction", required=True)
        else:
            command.add_argument("--prompt", required=True)
    return parser


def _load_editor(rt, cfg, mode, checkpoint):
    from .checkpoint import editor_identity, load_stage_weights
    from .model import build_editor

    if (mode == "trained") != bool(checkpoint):
        raise ValueError("trained mode requires --checkpoint; native mode must omit it")
    editor = build_editor(rt, cfg).eval()
    meta = load_stage_weights(checkpoint, editor, editor_identity(rt, cfg)) if checkpoint else None
    return editor, meta


def main():
    args = _parser().parse_args()
    from . import distributed as dist
    from .accelerator import set_device

    cfg = load_config(args.config)
    if args.command != "train" and args.distributed:
        raise ValueError("Inference runs on one device")
    device = dist.initialize(args.device) if args.distributed else set_device(args.device)
    if args.command == "train":
        from .train import train

        train(args, cfg, device)
    else:
        import torch

        from .images import save_image
        from .inference import edit_image, t2i_feedback
        from .loading import load_native

        if (args.checkpoint_mode == "trained") != bool(args.checkpoint):
            raise ValueError("trained mode requires --checkpoint; native mode must omit it")
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        rt = load_native(args.model, args.source, cfg, device, False)
        editor, meta = _load_editor(rt, cfg, args.checkpoint_mode, args.checkpoint)
        if args.command == "edit":
            images, state, record = edit_image(
                rt, editor, cfg, args.image, args.instruction, args.seed,
                edit_cfg_scale=args.edit_cfg_scale,
            )
        else:
            images, state, record = t2i_feedback(
                rt, editor, cfg, args.prompt, args.seed,
                edit_cfg_scale=args.edit_cfg_scale,
            )
        for name, image in images.items():
            save_image(output / (name + ".png"), image.clamp(-1.0, 1.0))
        if state is not None:
            torch.save(state.detach().float().cpu(), output / "edited_state.pt")
        record.update(
            checkpoint_mode=args.checkpoint_mode,
            checkpoint=args.checkpoint,
            checkpoint_step=meta.get("step") if meta else None,
        )
        (output / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    if args.distributed:
        dist.shutdown()


if __name__ == "__main__":
    main()
