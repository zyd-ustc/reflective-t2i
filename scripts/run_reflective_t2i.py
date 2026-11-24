"""Unified CLI for reflective_t2i_v1: DDP training, read_progress probe, pair eval, and 5-arm inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_latent_cot.reflective_t2i.core import (
    IsochronicTransitionV2,
    LatentState,
    PipelineConfig,
    ReflectionPolicyConfig,
    ReflectiveT2IPipeline,
    TransitionConfig,
)
from qwen_latent_cot.reflective_t2i.evaluation import (
    compute_hf_grad,
    compute_oob_frac,
    probe_read_progress,
    score_edit_pair_vlm,
)
from qwen_latent_cot.reflective_t2i.evaluation.benchmark import save_strip
from qwen_latent_cot.reflective_t2i.generators.diffusers import build_generator
from qwen_latent_cot.reflective_t2i.observers import LocalHFObserver, OpenAIVLMObserver
from qwen_latent_cot.reflective_t2i.training import LossConfig, RolloutConfig
from qwen_latent_cot.reflective_t2i.training.dataset import ReflectiveEditPairDataset
from qwen_latent_cot.reflective_t2i.training.train import (
    TrainConfig,
    cleanup_distributed,
    init_distributed,
    train_corrector,
)


def _setup_device(device: str) -> str:
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(device)
    return device


def _build_observer(cfg: dict, *, base_url: str | None = None, device: str = "npu:0"):
    ocfg = cfg.get("observer", {})
    backend = ocfg.get("backend", "openai_vlm")
    if backend == "openai_vlm":
        if not base_url:
            raise ValueError("--observer-url is required for openai_vlm backend")
        return OpenAIVLMObserver(base_url=base_url, model=ocfg["model_id"])
    if backend == "local_hf":
        return LocalHFObserver(ocfg["model_id"], device=device)
    raise ValueError(f"Unsupported observer backend: {backend}")


def cmd_train(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tcfg = cfg["training"]
    gcfg = cfg["generator"]
    rcfg = cfg["reflection"]
    ccfg = cfg["corrector"]

    # Apply explicit CLI overrides cleanly before building model & dataset
    if args.width is not None:
        gcfg["width"] = int(args.width)
    if args.height is not None:
        gcfg["height"] = int(args.height)
    if args.read_progress is not None:
        rcfg["read_progress"] = float(args.read_progress)
    if args.stop_progress is not None:
        rcfg["stop_progress"] = float(args.stop_progress)
    if args.lora_rank is not None:
        ccfg["rank"] = int(args.lora_rank)
    if args.layer_scope is not None:
        ccfg["layer_scope"] = args.layer_scope
    if args.loss_mode is not None:
        tcfg["loss_mode"] = args.loss_mode
    if args.soar_prob is not None:
        tcfg["soar_rollout_prob"] = float(args.soar_prob)
    if args.dropout_target is not None:
        tcfg["dropout_target"] = args.dropout_target
    if args.bridge_start_prob is not None:
        tcfg["bridge_start_prob"] = float(args.bridge_start_prob)
    if args.condition_mode is not None:
        ccfg["condition_mode"] = args.condition_mode

    device, rank, _, world = init_distributed(args.device)
    try:
        adapter = build_generator(cfg, device, attach_corrector=True)
        dataset = ReflectiveEditPairDataset(
            args.manifest or tcfg["manifest_glob"],
            width=int(gcfg["width"]),
            height=int(gcfg["height"]),
            split="train",
            max_records=args.max_records if args.max_records is not None else tcfg.get("pilot_pairs"),
            seed=int(args.seed),
            verify_files=False,
        )
        grad_accum = int(args.grad_accum if args.grad_accum is not None else tcfg.get("grad_accum", 1))
        train_cfg = TrainConfig(
            lr=float(args.lr if args.lr is not None else tcfg.get("lr", 3e-5)),
            lr_final=float(tcfg.get("lr_final", 5e-6)),
            warmup_steps=int(args.warmup_steps if args.warmup_steps is not None else tcfg.get("warmup_steps", 200)),
            grad_accum=grad_accum,
            condition_mode=ccfg.get("condition_mode", "native_prompt_plus_feedback"),
            save_every=int(args.save_every),
            keep_checkpoints=int(tcfg.get("keep_checkpoints", 4)),
            log_every=int(args.log_every),
            seed=int(args.seed),
            rollout=RolloutConfig(
                read_progress=float(rcfg["read_progress"]),
                train_max_progress=float(rcfg["stop_progress"]),
                bridge_start_prob=float(tcfg.get("bridge_start_prob", 0.25)),
                soar_rollout_prob=float(tcfg.get("soar_rollout_prob", 0.15)),
                soar_warmup_steps=int(args.soar_warmup_steps if args.soar_warmup_steps is not None else tcfg.get("soar_warmup_steps", 0)),
                soar_cps_prob=float(tcfg.get("soar_cps_prob", 0.0)),
                soar_cps_eta=float(tcfg.get("soar_cps_eta", 0.30)),
                edit_prompt_dropout=float(tcfg.get("edit_prompt_dropout", 0.10)),
                dropout_target=str(tcfg.get("dropout_target", "source_identity")),
            ),
            loss=LossConfig(
                mode=str(tcfg.get("loss_mode", "delta_weighted_clean_mse_plus_cosine")),
                delta_weight_lambda=float(tcfg.get("delta_weight_lambda", 1.0)),
                delta_weight_max=float(tcfg.get("delta_weight_max", 3.0)),
                cosine_lambda=float(tcfg.get("cosine_lambda", 0.02)),
            ),
        )
        summary = train_corrector(
            adapter,
            dataset,
            train_cfg,
            output_dir=args.out,
            max_steps=int(args.max_steps),
            rank=rank,
            world_size=world,
            resume_from=args.resume,
            config_payload=cfg,
        )
        if rank == 0:
            print(json.dumps(summary, indent=2), flush=True)
    finally:
        cleanup_distributed()


def cmd_probe_progress(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.width is not None:
        cfg["generator"]["width"] = int(args.width)
    if args.height is not None:
        cfg["generator"]["height"] = int(args.height)
    device = _setup_device(args.device)
    adapter = build_generator(cfg, device, attach_corrector=False)
    prompts = [
        "A photo of three red apples and two green pears on a wooden kitchen table",
        "A white ceramic mug to the left of a blue hardcover book on a desk",
        "Four yellow rubber ducks swimming in a clear pond beside a gray stone",
        "A black cat sitting on top of a red velvet armchair near a sunny window",
        "Five glazed donuts arranged in a row on a white plate",
        "A small brass telescope to the right of a green potted cactus on a windowsill",
    ][: int(args.num_prompts)]
    candidates = [float(x) for x in args.candidates.split(",")]
    report = probe_read_progress(
        adapter,
        prompts,
        candidate_progress=candidates,
        guidance_scale=float(cfg["generator"].get("guidance_scale", 4.0)),
        lookahead_steps=int(cfg["reflection"].get("lookahead_steps", 4)),
        seed=int(args.seed),
        out_dir=args.out,
    )
    print(json.dumps(report, indent=2), flush=True)


def _bucket_action(inst: str, sem: str) -> str:
    low = f"{inst} {sem}".lower()
    if low.startswith("add ") or "object_add" in low or "count_add" in low or "count_increase" in low:
        return "add"
    if low.startswith("remove ") or "object_remove" in low or "count_remove" in low or "count_reduce" in low:
        return "remove"
    if low.startswith(("move ", "relocate ", "swap ")) or "spatial" in low:
        return "move"
    return "attr"


def cmd_eval_pairs(args):
    """Generate edited outputs on unseen val pairs (stratified by add/remove/attr/move)."""
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.width is not None:
        cfg["generator"]["width"] = int(args.width)
    if args.height is not None:
        cfg["generator"]["height"] = int(args.height)
    if args.read_progress is not None:
        cfg["reflection"]["read_progress"] = float(args.read_progress)
    if args.stop_progress is not None:
        cfg["reflection"]["stop_progress"] = float(args.stop_progress)
    if args.lora_rank is not None:
        cfg["corrector"]["rank"] = int(args.lora_rank)
    if args.layer_scope is not None:
        cfg["corrector"]["layer_scope"] = args.layer_scope
    if args.anchor_mode is not None:
        cfg["corrector"]["anchor_mode"] = args.anchor_mode
    if args.edit_cfg_scale is not None:
        cfg["corrector"]["edit_cfg_scale"] = float(args.edit_cfg_scale)

    device = _setup_device(args.device)
    adapter = build_generator(cfg, device, attach_corrector=True)
    if args.checkpoint:
        sd = torch.load(Path(args.checkpoint) / "corrector.pt", map_location="cpu", weights_only=True)
        adapter.corrector.load_state_dict_trainable(sd, strict=True)

    ds = ReflectiveEditPairDataset(
        args.manifest or cfg["training"]["manifest_glob"],
        width=int(cfg["generator"]["width"]),
        height=int(cfg["generator"]["height"]),
        split="val",
        max_records=None,
        seed=int(args.seed),
        verify_files=True,
    )
    # Deterministic stratified selection across add / remove / attr / move
    by_act: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(ds.rows):
        by_act[_bucket_action(row["edit_instruction"], row["semantic_task"])].append(idx)
    rng = random.Random(int(args.seed))
    chosen: list[tuple[int, str]] = []
    quotas = {"add": int(args.n_add), "remove": int(args.n_remove), "attr": int(args.n_attr), "move": int(args.n_move)}
    for act, q in quotas.items():
        pool = list(by_act.get(act, []))
        rng.shuffle(pool)
        chosen.extend((i, act) for i in pool[:q])

    mine = [item for k, item in enumerate(chosen) if k % int(args.world) == int(args.rank)]
    out_dir = Path(args.out).resolve()
    img_dir = out_dir / "images"
    strip_dir = out_dir / "strips"
    img_dir.mkdir(parents=True, exist_ok=True)
    strip_dir.mkdir(parents=True, exist_ok=True)

    p0 = float(cfg["reflection"]["read_progress"])
    p_stop = float(cfg["reflection"]["stop_progress"])
    ccfg = cfg["corrector"]
    gcfg = cfg["generator"]
    transition = IsochronicTransitionV2(
        TransitionConfig(
            read_progress=p0,
            stop_progress=p_stop,
            anchor_mode=ccfg.get("anchor_mode", "conditional_native"),
            edit_cfg_scale=float(ccfg.get("edit_cfg_scale", 1.4)),
            trust_region_rho=float(ccfg.get("trust_region_rho", 0.65)),
            noise_refresh=float(ccfg.get("noise_refresh", 0.0)),
        )
    )
    null_cond = adapter.encode_prompt(None)
    shard_path = out_dir / f"eval_rank{int(args.rank):02d}.jsonl"

    import numpy as np

    with torch.no_grad():
        for pos, (ds_idx, act) in enumerate(mine, 1):
            sample = ds[ds_idx]
            uid = sample["uid"]
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in uid)
            src_rgb = sample["source_rgb"].unsqueeze(0).to(device)
            tar_rgb = sample["target_rgb"].unsqueeze(0).to(device)
            valid = sample["valid"].unsqueeze(0).to(device)

            src_clean = adapter.encode_clean(src_rgb)
            gen = torch.Generator(device=device).manual_seed(int(args.seed) + ds_idx)
            eps = adapter.geometry.sample_noise(tuple(src_clean.shape), device=device, generator=gen)
            z_p0 = adapter.geometry.interpolate(src_clean, eps, p0)
            state_p0 = LatentState(tensor=z_p0, progress=p0, valid_mask=valid)

            orig_text = sample.get("original_prompt") or None
            tail_text = sample.get("target_caption") or orig_text
            orig_cond = adapter.encode_prompt(orig_text) if orig_text else null_cond
            tail_cond = adapter.encode_prompt(tail_text) if tail_text else null_cond
            edit_cond = adapter.encode_edit_condition(
                feedback=sample["edit_instruction"],
                original_prompt=orig_text,
                target_rewrite=tail_text,
                condition_mode=ccfg.get("condition_mode", "native_prompt_plus_feedback"),
            )

            tr = transition.apply(adapter, state_p0, edit_cond, orig_cond, null_cond, generator=gen)
            final_lat = adapter.continue_native(
                tr.state,
                tail_cond,
                null_cond,
                float(gcfg.get("guidance_scale", 4.0)) if tail_text else 1.0,
                cfg_interval=tuple(float(v) for v in gcfg.get("cfg_interval", (0.0, 0.80))),
            )

            src_pil = Image.fromarray(((sample["source_rgb"].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy())
            tar_pil = Image.fromarray(((sample["target_rgb"].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy())
            edit_pil = adapter.decode_preview(final_lat)
            clean_pil = adapter.decode_preview(tr.clean_target_est)

            src_path = img_dir / f"{safe}_src.png"
            edit_path = img_dir / f"{safe}_edit.png"
            src_pil.save(src_path)
            edit_pil.save(edit_path)
            if pos <= 8:
                save_strip(
                    [(src_pil, "SOURCE"), (tar_pil, "GT_TARGET"), (clean_pil, f"CLEAN({p_stop})"), (edit_pil, "RESULT")],
                    strip_dir / f"{safe}.jpg",
                )

            edit_rgb = torch.from_numpy(np.asarray(edit_pil, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
            delta_gt = (tar_rgb - src_rgb).abs().mean(dim=1, keepdim=True)
            edit_region = ((delta_gt > 0.12).float() * valid)
            bg_region = ((delta_gt <= 0.12).float() * valid)
            err = (edit_rgb - tar_rgb).abs().mean(dim=1, keepdim=True)
            edit_l1 = float((err * edit_region).sum().item() / max(edit_region.sum().item(), 1.0))
            bg_l1 = float(((edit_rgb - src_rgb).abs().mean(dim=1, keepdim=True) * bg_region).sum().item() / max(bg_region.sum().item(), 1.0))
            hf_src = compute_hf_grad(src_rgb)
            hf_edit = compute_hf_grad(edit_rgb)

            rec = {
                "uid": uid,
                "action": act,
                "instruction": sample["edit_instruction"],
                "src_path": str(src_path),
                "edit_path": str(edit_path),
                "edit_l1": round(edit_l1, 5),
                "bg_l1": round(bg_l1, 5),
                "hf_edit": round(hf_edit, 5),
                "hf_ratio": round(hf_edit / max(hf_src, 1e-5), 4),
                "edit_clean_rms": tr.telemetry["edit_clean_rms"],
            }
            with shard_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if pos == 1 or pos % 10 == 0 or pos == len(mine):
                print(f"[eval-pairs r{args.rank}] ({pos}/{len(mine)}) {act} edit_l1={edit_l1:.4f} bg_l1={bg_l1:.4f} hf_ratio={rec['hf_ratio']:.3f}", flush=True)


def cmd_score_pairs(args):
    """Score eval-pairs outputs side-by-side with independent local Qwen3-VL-8B-Instruct."""
    out_dir = Path(args.out).resolve()
    rows = []
    for p in sorted(out_dir.glob("eval_rank*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    mine = [r for i, r in enumerate(rows) if i % int(args.world) == int(args.rank)]
    obs = LocalHFObserver(args.judge_model, device=args.device)
    shard_path = out_dir / f"score_rank{int(args.rank):02d}.jsonl"
    for pos, r in enumerate(mine, 1):
        with Image.open(r["src_path"]) as sim, Image.open(r["edit_path"]) as eim:
            sc = score_edit_pair_vlm(obs, sim, eim, r["instruction"])
        rec = {**r, **sc}
        with shard_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if pos == 1 or pos % 10 == 0 or pos == len(mine):
            print(f"[score-pairs r{args.rank}] ({pos}/{len(mine)}) {r['action']} followed={sc['edit_followed']} strict={sc['strict_pass']}", flush=True)


def cmd_infer(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = _setup_device(args.device)
    adapter = build_generator(cfg, device, attach_corrector=True)
    if args.checkpoint:
        sd = torch.load(Path(args.checkpoint) / "corrector.pt", map_location="cpu", weights_only=True)
        adapter.corrector.load_state_dict_trainable(sd, strict=True)

    observer = None
    if args.arm in ("B_rewrite_resample", "C_reflective"):
        observer = _build_observer(cfg, base_url=args.observer_url, device=args.observer_device or device)

    gcfg, rcfg, ccfg = cfg["generator"], cfg["reflection"], cfg["corrector"]
    pipe = ReflectiveT2IPipeline(
        adapter,
        observer,
        PipelineConfig(
            read_progress=float(rcfg["read_progress"]),
            stop_progress=float(rcfg["stop_progress"]),
            lookahead_steps=int(rcfg.get("lookahead_steps", 4)),
            guidance_scale=float(gcfg.get("guidance_scale", 4.0)),
            cfg_interval=tuple(float(v) for v in gcfg.get("cfg_interval", (0.0, 0.80))),
            max_corrections=int(rcfg.get("max_corrections", 1)),
            condition_mode=ccfg.get("condition_mode", "native_prompt_plus_feedback"),
            transition=TransitionConfig(
                read_progress=float(rcfg["read_progress"]),
                stop_progress=float(rcfg["stop_progress"]),
                anchor_mode=ccfg.get("anchor_mode", "conditional_native"),
                edit_cfg_scale=float(ccfg.get("edit_cfg_scale", 1.4)),
                trust_region_rho=float(ccfg.get("trust_region_rho", 0.65)),
                noise_refresh=float(ccfg.get("noise_refresh", 0.0)),
            ),
            reflection=ReflectionPolicyConfig(
                focused_recheck_actions=tuple(rcfg.get("focused_recheck_actions", ("add", "remove"))),
                confidence_threshold=float(rcfg.get("confidence_threshold", 0.75)),
                contradiction_gate=bool(rcfg.get("contradiction_gate", True)),
            ),
        ),
    )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    res = pipe.run(
        args.prompt,
        seed=int(args.seed),
        arm=args.arm,
        oracle_instruction=args.oracle_instruction,
        shuffled_instruction=args.shuffled_instruction,
    )
    res.image.save(out_dir / f"{args.arm}.png")
    for idx, prev in enumerate(res.preview_images):
        prev.save(out_dir / f"preview_{idx}.png")
    trace = {
        "arm": res.arm,
        "prompt": args.prompt,
        "seed": int(args.seed),
        "telemetry": res.telemetry,
        "decisions": [d.__dict__ for d in res.decisions],
        "transitions": [t.telemetry for t in res.transitions],
    }
    (out_dir / f"{args.arm}_trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(trace, indent=2, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description="reflective_t2i_v1 unified runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("train")
    st.add_argument("--config", required=True)
    st.add_argument("--out", required=True)
    st.add_argument("--manifest", default=None)
    st.add_argument("--max-records", type=int, default=None)
    st.add_argument("--max-steps", type=int, default=1000)
    st.add_argument("--save-every", type=int, default=250)
    st.add_argument("--log-every", type=int, default=20)
    st.add_argument("--grad-accum", type=int, default=None)
    st.add_argument("--lr", type=float, default=None)
    st.add_argument("--warmup-steps", type=int, default=None)
    st.add_argument("--width", type=int, default=None)
    st.add_argument("--height", type=int, default=None)
    st.add_argument("--read-progress", type=float, default=None)
    st.add_argument("--stop-progress", type=float, default=None)
    st.add_argument("--lora-rank", type=int, default=None)
    st.add_argument("--layer-scope", default=None)
    st.add_argument("--loss-mode", default=None)
    st.add_argument("--soar-prob", type=float, default=None)
    st.add_argument("--soar-warmup-steps", type=int, default=None)
    st.add_argument("--dropout-target", default=None)
    st.add_argument("--bridge-start-prob", type=float, default=None)
    st.add_argument("--condition-mode", default=None)
    st.add_argument("--resume", default=None)
    st.add_argument("--device", default="npu:0")
    st.add_argument("--seed", type=int, default=31)
    st.set_defaults(func=cmd_train)

    sp = sub.add_parser("probe-progress")
    sp.add_argument("--config", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--candidates", default="0.24,0.30,0.36")
    sp.add_argument("--num-prompts", type=int, default=6)
    sp.add_argument("--width", type=int, default=512)
    sp.add_argument("--height", type=int, default=512)
    sp.add_argument("--device", default="npu:0")
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_probe_progress)

    se = sub.add_parser("eval-pairs")
    se.add_argument("--config", required=True)
    se.add_argument("--checkpoint", default=None)
    se.add_argument("--manifest", default=None)
    se.add_argument("--out", required=True)
    se.add_argument("--width", type=int, default=512)
    se.add_argument("--height", type=int, default=512)
    se.add_argument("--read-progress", type=float, default=None)
    se.add_argument("--stop-progress", type=float, default=None)
    se.add_argument("--lora-rank", type=int, default=None)
    se.add_argument("--layer-scope", default=None)
    se.add_argument("--anchor-mode", default=None)
    se.add_argument("--edit-cfg-scale", type=float, default=None)
    se.add_argument("--n-add", type=int, default=40)
    se.add_argument("--n-remove", type=int, default=20)
    se.add_argument("--n-attr", type=int, default=20)
    se.add_argument("--n-move", type=int, default=20)
    se.add_argument("--rank", type=int, default=0)
    se.add_argument("--world", type=int, default=1)
    se.add_argument("--device", default="npu:0")
    se.add_argument("--seed", type=int, default=20261122)
    se.set_defaults(func=cmd_eval_pairs)

    ss = sub.add_parser("score-pairs")
    ss.add_argument("--out", required=True)
    ss.add_argument("--judge-model", default="/data/zyd_workspace/models/Qwen3-VL-8B-Instruct")
    ss.add_argument("--rank", type=int, default=0)
    ss.add_argument("--world", type=int, default=1)
    ss.add_argument("--device", default="npu:0")
    ss.set_defaults(func=cmd_score_pairs)

    si = sub.add_parser("infer")
    si.add_argument("--config", required=True)
    si.add_argument("--checkpoint", default=None)
    si.add_argument("--prompt", required=True)
    si.add_argument("--seed", type=int, default=42)
    si.add_argument("--arm", default="C_reflective")
    si.add_argument("--observer-url", default=None)
    si.add_argument("--observer-device", default=None)
    si.add_argument("--oracle-instruction", default=None)
    si.add_argument("--shuffled-instruction", default=None)
    si.add_argument("--device", default="npu:0")
    si.add_argument("--out", required=True)
    si.set_defaults(func=cmd_infer)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
