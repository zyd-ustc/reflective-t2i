"""Local HuggingFace Transformers VLM Observer (Qwen3-VL / Qwen3.5 / InternVL3.5 on NPU/GPU)."""

from __future__ import annotations

from PIL import Image
import torch

from ..core.contracts import ObserverBackend


class LocalHFObserver(ObserverBackend):
    """Runs a local multimodal chat model deterministically (temperature=0 / greedy)."""

    def __init__(self, model_path: str, *, device: str = "npu:0", dtype: str = "bfloat16") -> None:
        if str(device).startswith("npu"):
            import torch_npu  # noqa: F401
            torch.npu.set_device(device)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype == "bfloat16" and self.device.type != "cpu" else torch.float32

        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[import-not-found]

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=self.dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def query_image(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        pil_img = image.convert("RGB")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], images=[pil_img], return_tensors="pt").to(self.device)
        gen_kwargs = {"max_new_tokens": int(max_new_tokens), "do_sample": temperature > 0.0}
        if temperature > 0.0:
            gen_kwargs["temperature"] = float(temperature)
        out = self.model.generate(**inputs, **gen_kwargs)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0].strip()
