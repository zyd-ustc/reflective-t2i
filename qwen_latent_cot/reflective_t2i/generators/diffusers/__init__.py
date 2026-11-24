"""Diffusers & S3-DiT / MMDiT GeneratorAdapters."""

from .sd35 import SD35Adapter, SD35Config
from .z_image import ZImageAdapter, ZImageConfig

__all__ = ["SD35Adapter", "SD35Config", "ZImageAdapter", "ZImageConfig", "build_generator"]


def build_generator(
    cfg: dict, device: str, *, attach_corrector: bool = True, default_rank: int = 16,
    default_scope: str = "middle_third", default_sd_size: int = 1024, default_zimage_src: str | None = None,
) -> SD35Adapter | ZImageAdapter:
    g = cfg["generator"]
    c = cfg.get("corrector", {})
    common = dict(
        dtype=str(g.get("dtype", "bfloat16")),
        lora_rank=int(c.get("rank", default_rank)),
        layer_scope=c.get("layer_scope", default_scope),
    )
    if g["adapter"] == "z_image_s3dit":
        adapter = ZImageAdapter
        config = ZImageConfig(
            model_path=g["model_path"], zimage_core_src=g.get("zimage_core_src", default_zimage_src),
            width=int(g.get("width", 512)), height=int(g.get("height", 512)),
            num_inference_steps=int(g.get("num_inference_steps", 30)),
            adapt_noise_refiner=bool(c.get("adapt_noise_refiner", False)),
            target_suffixes=tuple(c.get("target_suffixes", ZImageConfig.target_suffixes)), **common,
        )
    elif g["adapter"] == "diffusers_sd35":
        adapter = SD35Adapter
        config = SD35Config(
            model_id=g["model_id"], width=int(g.get("width", default_sd_size)),
            height=int(g.get("height", default_sd_size)), num_inference_steps=int(g.get("num_inference_steps", 40)),
            include_ff=bool(c.get("include_ff", False)), **common,
        )
    else:
        raise ValueError(f"Unsupported generator adapter: {g['adapter']}")
    return adapter.from_pretrained(config, device=device, attach_corrector=attach_corrector)
