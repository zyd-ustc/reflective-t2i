"""Make the verified image wrong: the editor service that consumes plan instructions.

The service is a vLLM-Omni ``/v1/images/edits`` endpoint. Its reply is checked
twice: the picture must decode, and it must keep the source's aspect ratio,
because the training pair shares one canvas rectangle and the loader refuses a
pair whose ratios differ by more than 1%.
"""

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
import uuid

from PIL import Image

# The training pair shares one canvas rectangle and the loader refuses a pair
# whose aspect ratios differ by more than 1%. Duplicated here so this module
# stays importable without the training stack (no torch) for the offline tests.
RATIO_TOLERANCE = 0.01


class EditError(RuntimeError):
    """The service failed, or returned something that cannot be a training source."""


def _multipart(fields, files):
    boundary = "----dsh" + uuid.uuid4().hex
    body = io.BytesIO()
    for key, value in fields:
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                    % (boundary, key, value)).encode())
    for key, path in files:
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                    "Content-Type: image/png\r\n\r\n" % (boundary, key, path.name)).encode())
        body.write(path.read_bytes())
        body.write(b"\r\n")
    body.write(("--%s--\r\n" % boundary).encode())
    return body.getvalue(), "multipart/form-data; boundary=" + boundary


class EditClient:
    def __init__(self, base_url, model, steps=20, cfg_scale=1.0, timeout=300, transport=None, workers=1):
        self.base_url = base_url.rstrip("/")
        self.workers = max(1, int(workers))
        self.model = model
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.timeout = int(timeout)
        self.transport = transport or self._http_transport

    def _http_transport(self, fields, files):
        body, content_type = _multipart(fields, files)
        request = urllib.request.Request(
            self.base_url + "/v1/images/edits", data=body, headers={"Content-Type": content_type})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def edit(self, source, instruction, seed=42):
        """The edited image bytes plus the service's own timing, or EditError."""
        fields = [("model", self.model), ("prompt", instruction), ("num_inference_steps", self.steps),
                  ("true_cfg_scale", self.cfg_scale), ("seed", seed), ("response_format", "b64_json")]
        started = time.time()
        try:
            reply = self.transport(fields, [("image", source)])
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise EditError("the editor service failed: %s" % exc) from exc
        try:
            image = Image.open(io.BytesIO(base64.b64decode(reply["data"][0]["b64_json"]))).convert("RGB")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EditError("the editor reply carries no decodable image: %s" % exc) from exc
        source_size = Image.open(source).size
        ratio = image.size[0] / image.size[1]
        source_ratio = source_size[0] / source_size[1]
        if abs(ratio - source_ratio) / source_ratio > RATIO_TOLERANCE:
            raise EditError("the edited image changed the aspect ratio: %s -> %s"
                            % (source_size, image.size))
        return dict(image=image, seconds=round(time.time() - started, 2), seed=seed,
                    size=list(image.size))

    def edit_many(self, jobs):
        """``(key, source, instruction, seed)`` tuples; failures come back in place.

        The edit server batches requests itself, so this only keeps the socket
        busy; ``workers`` above one does not start extra servers.
        """
        def run(job):
            key, source, text, seed = job
            try:
                return key, self.edit(source, text, seed), None
            except EditError as exc:
                return key, None, str(exc)

        results = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for key, value, error in pool.map(run, jobs):
                results[key] = (value, error)
        return results
