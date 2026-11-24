"""Generator-agnostic ReflectionPolicy with single-pass structured JSON and bidirectional gating.

Addresses two empirical bottlenecks identified in Step 1 (180-case GenEval2 ablation):
  1. High Damage Rate on false `remove` triggers (39%-87%): adds prompt-blind focused
     recheck on BOTH `add` and `remove`, plus a confidence gate and deterministic
     arithmetic consistency gate.
  2. High latency from 2-hop describe -> text-only judge: uses a single visual structured
     JSON pass that keeps prompt-blind object enumeration first, only triggering a focused
     visual recheck when `match=false` on `add`/`remove`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any

from PIL import Image

from .contracts import EditAction, ObserverBackend, ReflectionDecision


NUMBER_WORDS = {
    "zero": 0, "no": 0, "none": 0,
    "one": 1, "a": 1, "an": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

STRUCTURED_AUDIT_SYSTEM = (
    "You are a strict visual auditor for text-to-image generation. "
    "Step 1: First inspect ONLY the image and enumerate every visible object with its exact count, "
    "color/material, and rough position. Do not hallucinate missing items. "
    "Step 2: Compare your visual inventory against the target prompt. "
    "Step 3: Output ONLY a single valid JSON object matching the requested schema. "
    "Never mix an addition and a removal in one instruction. "
    "Prefer coarse spatial hints (e.g. 'open space on the right') over pixel coordinates."
)

STRUCTURED_AUDIT_TEMPLATE = (
    "Target prompt: {prompt}\n\n"
    "Return ONLY valid JSON with these exact keys:\n"
    "{{\n"
    '  "observed_objects": [{{"name": "<noun>", "count": <int>, "attributes": "<color/material>", "location": "<rough area>"}}],\n'
    '  "missing_or_wrong": ["<short discrepancy>"],\n'
    '  "match": <true or false>,\n'
    '  "action": "<none | add | remove | attr | move | other>",\n'
    '  "instruction": "<one atomic edit instruction or empty string if match is true>",\n'
    '  "target_rewrite": "<clean declarative target prompt describing the desired image>",\n'
    '  "coarse_region_hint": "<coarse empty/target region hint or empty string>",\n'
    '  "confidence": <float in 0.0..1.0>\n'
    "}}\n\n"
    "Rules for instruction when match is false:\n"
    "- For count changes, always state endpoints: 'Add K <obj> so that it contains exactly N <obj> instead of the current M, "
    "placing the new ones in <coarse empty region>; keep everything else unchanged.' or "
    "'Remove K <obj> so that it contains exactly N <obj> instead of the current M, filling the vacated areas naturally; "
    "keep everything else unchanged.'\n"
    "- If the image already satisfies all constraints (even if rough/early preview), set match=true, action='none', instruction=''."
)

FOCUSED_RECHECK_SYSTEM = (
    "You count visible instances of a single requested subject in a picture. "
    "Report ONLY what is visibly present, including small, distant, or partially emerged objects. "
    "Return ONLY valid JSON."
)

FOCUSED_RECHECK_TEMPLATE = (
    "Carefully scan the entire image for every visible instance of '{subject}' "
    "(including small, background, or partially formed ones). "
    "Return ONLY JSON: "
    '{{"instances": ["<location and color of each visible instance>"], "count": <exact integer>, "uncertain": <true or false>}}'
)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", str(text or ""), flags=re.I).strip()


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = _strip_think(raw_text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(cleaned[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _parse_int_token(token: str) -> int | None:
    low = str(token or "").strip().lower()
    if low.isdigit():
        return int(low)
    return NUMBER_WORDS.get(low)


def _norm_noun(phrase: str) -> str:
    words = re.findall(r"[a-z]+", str(phrase or "").lower())
    stop = {
        "the", "a", "an", "of", "in", "on", "at", "from", "to",
        "image", "picture", "frame", "background", "foreground",
        "left", "right", "above", "below",
    }
    words = [w for w in words if w not in stop]
    norm = []
    for w in words:
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("ves"):
            w = w[:-3] + "f"
        elif len(w) > 3 and w.endswith("es") and not w.endswith(("ses", "xes", "zes", "ches", "shes")):
            w = w[:-1]
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "glass", "scissors")):
            w = w[:-1]
        norm.append(w)
    return " ".join(norm)


def classify_action(instruction: str, declared: str | None = None) -> EditAction:
    valid = {"none", "add", "remove", "attr", "move", "other"}
    if declared and declared.strip().lower() in valid and declared.strip().lower() != "other":
        return declared.strip().lower()  # type: ignore[return-value]
    low = " ".join(str(instruction or "").split()).lower()
    if not low:
        return "none"
    if low.startswith("add "):
        return "add"
    if low.startswith("remove "):
        return "remove"
    if low.startswith(("move ", "relocate ", "reposition ", "swap ", "rearrange ")):
        return "move"
    if low.startswith(("change ", "replace ", "recolor ", "make ")):
        return "attr"
    return "other"


def extract_count_subject_and_endpoints(instruction: str) -> dict[str, Any] | None:
    """Parse canonical Add/Remove endpoint instructions if present."""
    text = " ".join(str(instruction or "").split()).strip()
    ep_re = re.compile(
        r"^(add|remove)\s+(a|an|zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(.+?)"
        r"\s+so\s+that\s+it\s+contains\s+(?:exactly\s+)?"
        r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(.+?)"
        r"\s+instead\s+of\s+the\s+current\s+"
        r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
        flags=re.I,
    )
    m = ep_re.search(text)
    if m:
        op = m.group(1).lower()
        delta_k = _parse_int_token(m.group(2))
        target_n = _parse_int_token(m.group(4))
        current_m = _parse_int_token(m.group(6))
        subj = m.group(5).strip(" ,.;:")
        return {
            "op": op,
            "delta_k": delta_k,
            "target_n": target_n,
            "current_m": current_m,
            "subject": subj,
            "has_endpoints": True,
        }
    simple_re = re.compile(
        r"^(add|remove)\s+(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(.+?)"
        r"(?:\s+(?:to|from|in|on|into|near|beside)\s+the\s+(?:image|picture|scene|frame|background|foreground|left|right)\b"
        r"|\s+so\s+that\b|[,.;]|$)",
        flags=re.I,
    )
    m2 = simple_re.match(text)
    if m2:
        return {
            "op": m2.group(1).lower(),
            "delta_k": None,
            "target_n": None,
            "current_m": None,
            "subject": m2.group(2).strip(" ,.;:"),
            "has_endpoints": False,
        }
    return None


def extract_prompt_clause_counts(prompt: str) -> dict[str, int]:
    text = str(prompt or "").lower()
    if re.search(r"\b(without|except)\b", text):
        return {}
    cleaned = re.sub(r"^\s*(?:a|an)\s+(?:photo|picture|image|photograph|rendering|painting)\s+of\s+", "", text)
    clauses = re.split(
        r"\s+(?:and|with|,)\s+|\s+(?:on|in|to|at|above|below|under|beside|next\s+to|behind)\s+the\s+(?:left|right|top|bottom|above|below)(?:\s+of)?\s*",
        cleaned,
    )
    counts: dict[str, int] = {}
    num_pat = r"^(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+([a-z][a-z\s\-]{1,40}?)$"
    for clause in clauses:
        clause = re.sub(
            r"\s+(?:on|in|at)\s+the\s+(?:left|right|above|below|top|bottom).*$",
            "",
            clause.strip(" .,;:"),
        )
        m = re.match(num_pat, clause)
        if not m:
            continue
        cnt = _parse_int_token(m.group(1))
        noun = _norm_noun(m.group(2))
        if cnt is not None and cnt > 0 and noun:
            counts[noun] = counts.get(noun, 0) + cnt
    return counts


def check_instruction_consistency(instruction: str, prompt: str) -> tuple[bool, str | None]:
    """Deterministic arithmetic & prompt-count consistency gate (zero VLM bias)."""
    text = " ".join(str(instruction or "").split()).strip()
    if not text:
        return False, "empty_instruction"

    parsed = extract_count_subject_and_endpoints(text)
    if not parsed or not parsed["has_endpoints"]:
        return True, None

    op = parsed["op"]
    delta_k, target_n, current_m = parsed["delta_k"], parsed["target_n"], parsed["current_m"]
    if None in (delta_k, target_n, current_m):
        return True, None

    if op == "add":
        if target_n <= current_m or delta_k <= 0 or target_n - current_m != delta_k:
            return False, "invalid_add_endpoints"
    elif op == "remove":
        if target_n >= current_m or delta_k <= 0 or current_m - target_n != delta_k:
            return False, "invalid_remove_endpoints"

    obj_norm = _norm_noun(parsed["subject"])
    obj_head = obj_norm.split()[-1] if obj_norm else ""
    prompt_norm = _norm_noun(prompt)

    if op == "remove" and target_n == 0 and obj_head:
        if not re.search(r"\b(?:no|without|zero)\s+" + re.escape(obj_head), str(prompt or "").lower()):
            if obj_head in prompt_norm.split():
                return False, "removes_required_subject_to_zero"

    clause_counts = extract_prompt_clause_counts(prompt)
    if clause_counts and obj_norm:
        matched_prompt_count = clause_counts.get(obj_norm)
        if matched_prompt_count is None and obj_head:
            head_matches = [v for k, v in clause_counts.items() if k.split()[-1] == obj_head]
            if len(head_matches) == 1:
                matched_prompt_count = head_matches[0]
        if matched_prompt_count is not None:
            if current_m == matched_prompt_count:
                return False, "current_count_already_matches_prompt"
            if target_n != matched_prompt_count:
                return False, "target_count_contradicts_prompt"

    return True, None


def compile_concise_imperative(
    instruction: str,
    *,
    action: str | None = None,
    coarse_region_hint: str = "",
) -> str:
    """Compile a gated canonical audit instruction into a single-action concise imperative.

    Preserves the full endpoint arithmetic inside `ReflectionPolicy` for 100% deterministic
    gating, while emitting short single-action imperatives matching the corrector training
    distribution (eliminating verbose endpoint clauses, multi-sentence compound edits, and
    parenthetical hint clutter).
    """
    text = " ".join(str(instruction or "").split()).strip()
    if not text:
        return ""

    # Take only the first atomic sentence when the Observer packs multiple edits into one string.
    first_sent = re.split(r"(?<=[.!?])\s+(?=(?:Add|Remove|Change|Replace|Make|Move|Relocate|Swap)\b)", text, maxsplit=1)[0].strip()
    # Strip boilerplate suffix "; keep everything else unchanged."
    first_sent = re.sub(r"[;,.]?\s*keep\s+everything\s+else\s+unchanged\.?$", "", first_sent, flags=re.I).strip()

    ep = extract_count_subject_and_endpoints(first_sent)
    if ep and ep["has_endpoints"] and ep["delta_k"] is not None:
        op_word = "Add" if ep["op"] == "add" else "Remove"
        k = int(ep["delta_k"])
        # Extract the exact subject phrase after `add/remove <k>` (preserves color/material modifiers)
        m_subj = re.match(
            r"^(?:add|remove)\s+(?:a|an|zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(.+?)\s+so\s+that\b",
            first_sent,
            flags=re.I,
        )
        subj = (m_subj.group(1) if m_subj else ep["subject"]).strip(" ,.;:")
        # Preserve coarse placement hint for `add` if present in the sentence or hint field
        if ep["op"] == "add":
            m_place = re.search(
                r"placing\s+(?:the\s+)?(?:new\s+)?(?:one|ones|item|items|object|objects)\s+([^.;]+)",
                first_sent,
                flags=re.I,
            )
            place = (m_place.group(1).strip(" ,.;:") if m_place else str(coarse_region_hint or "").strip(" ,.;:"))
            if place:
                if not re.match(r"^(?:in|on|at|to|near|beside|above|below|under|next\s+to|behind)\b", place, flags=re.I):
                    place = f"in the {place}"
                return f"{op_word} {k} {subj} {place}."
        return f"{op_word} {k} {subj}."

    # Non-endpoint instruction: clean up any trailing filling/unchanged boilerplate
    cleaned = re.sub(r",?\s*filling\s+the\s+vacated\s+areas?\s+naturally\.?$", "", first_sent, flags=re.I).strip(" ,.;:")
    return f"{cleaned}." if cleaned else ""


@dataclass
class ReflectionPolicyConfig:
    focused_recheck_actions: tuple[str, ...] = ("add", "remove")
    confidence_threshold: float = 0.75
    contradiction_gate: bool = True
    max_new_tokens: int = 384
    recheck_max_new_tokens: int = 160
    temperature: float = 0.0


class ReflectionPolicy:
    """Generator-agnostic observer policy with structured JSON + bidirectional recheck + gating."""

    def __init__(self, config: ReflectionPolicyConfig | None = None):
        self.config = config or ReflectionPolicyConfig()

    def reflect(
        self,
        observer: ObserverBackend,
        preview_image: Image.Image,
        original_prompt: str,
    ) -> ReflectionDecision:
        raw_text = observer.query_image(
            preview_image,
            STRUCTURED_AUDIT_TEMPLATE.format(prompt=original_prompt),
            system_prompt=STRUCTURED_AUDIT_SYSTEM,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        parsed = _extract_json_object(raw_text)
        if parsed is None:
            return ReflectionDecision(
                outcome="gated_noop",
                action="none",
                confidence=0.0,
                gated_by="failed_json_parse",
                raw={"audit_raw": raw_text},
            )

        matched = bool(parsed.get("match", True))
        instruction = " ".join(str(parsed.get("instruction") or "").split()).strip()
        target_rewrite = " ".join(str(parsed.get("target_rewrite") or original_prompt).split()).strip()
        coarse_hint = " ".join(str(parsed.get("coarse_region_hint") or "").split()).strip()
        try:
            confidence = float(parsed.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        action = classify_action(instruction, parsed.get("action"))
        obs_summary = json.dumps(parsed.get("observed_objects", []), ensure_ascii=False)

        if matched or not instruction or action == "none":
            return ReflectionDecision(
                outcome="satisfied",
                action="none",
                instruction="",
                target_rewrite=target_rewrite,
                coarse_region_hint="",
                confidence=confidence,
                observed_summary=obs_summary,
                raw={"audit": parsed, "audit_raw": raw_text},
            )

        decision = ReflectionDecision(
            outcome="edit", action=action, instruction=instruction, target_rewrite=target_rewrite,
            coarse_region_hint=coarse_hint, confidence=confidence, observed_summary=obs_summary,
            raw={"audit": parsed, "audit_raw": raw_text},
        )

        def gated(reason: str, **trace) -> ReflectionDecision:
            return replace(decision, outcome="gated_noop", instruction="", gated_by=reason,
                           gated_instruction=instruction, **trace)

        if confidence < self.config.confidence_threshold:
            return gated("low_confidence")

        if self.config.contradiction_gate:
            ok, reason = check_instruction_consistency(instruction, original_prompt)
            if not ok:
                return gated(reason)

        # Gate 2: Prompt-blind focused visual recheck for BOTH `add` AND `remove`
        recheck_raw = None
        if action in self.config.focused_recheck_actions:
            ep = extract_count_subject_and_endpoints(instruction)
            if ep and ep["subject"]:
                recheck_raw = observer.query_image(
                    preview_image,
                    FOCUSED_RECHECK_TEMPLATE.format(subject=ep["subject"]),
                    system_prompt=FOCUSED_RECHECK_SYSTEM,
                    max_new_tokens=self.config.recheck_max_new_tokens,
                    temperature=self.config.temperature,
                )
                recheck_obj = _extract_json_object(recheck_raw)
                if recheck_obj is not None:
                    re_count = recheck_obj.get("count")
                    uncertain = bool(recheck_obj.get("uncertain", False))
                    reason = None
                    if uncertain:
                        reason = f"focused_{action}_uncertain"
                    elif isinstance(re_count, int) and ep["target_n"] is not None:
                        if action == "add" and re_count >= ep["target_n"]:
                            reason = "focused_add_recheck"
                        elif action == "remove" and re_count <= ep["target_n"]:
                            reason = "focused_remove_recheck"
                    if reason:
                        return gated(reason, recheck_response=recheck_raw,
                                     raw={"audit": parsed, "recheck": recheck_obj})

        decision.recheck_response = recheck_raw
        return decision
