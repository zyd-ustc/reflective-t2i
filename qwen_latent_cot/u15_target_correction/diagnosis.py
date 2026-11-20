"""Prompt-blind UND reflection with Breakthrough-A focused add re-inspection and consistency gating."""

import re
import torch

from .runtime import autocast

NUMBER_WORDS = {
    "zero": 0,
    "no": 0,
    "none": 0,
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@torch.no_grad()
def ask_text(rt, question, cfg, system_message=None, max_new_tokens=None):
    """Text-only answer, with no image anywhere in the call."""
    from sensenova_u1.models.neo_unify.conversation import get_conv_template

    m, tok = rt.model, rt.tokenizer
    query = m._build_t2i_query(
        question,
        system_message=system_message,
        append_text="<think>\n\n</think>\n\n",
    )
    encoded = tok(query, return_tensors="pt")
    input_ids = encoded["input_ids"].to(rt.device)
    positions = torch.arange(input_ids.shape[1], dtype=torch.long, device=rt.device)
    indexes = torch.stack([positions, torch.zeros_like(positions), torch.zeros_like(positions)])
    separator = get_conv_template(m.template).sep.strip()
    eos = tok.convert_tokens_to_ids(separator)
    with autocast(rt.device):
        output = m.language_model.generate(
            inputs_embeds=m.language_model.get_input_embeddings()(input_ids),
            indexes=indexes,
            attention_mask=encoded["attention_mask"].to(rt.device),
            max_new_tokens=int(max_new_tokens or 256),
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos,
        )
    text = tok.batch_decode(output, skip_special_tokens=True)[0].split(separator)[0].strip()
    return text.rsplit("</think>", 1)[-1].strip()


def _image_inputs(rt, image):
    """The image tag and understanding-side pixels for one protocol canvas."""
    m, tok = rt.model, rt.tokenizer
    m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    m.img_start_token_id = tok.convert_tokens_to_ids("<img>")
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = ((image.detach().float().clamp(-1, 1) + 1) / 2 - mean) / std
    gh, gw = rt.height // m.patch_size, rt.width // m.patch_size
    grid = torch.tensor([[gh, gw]], device=rt.device)
    pixels = (
        m.patchify(rgb, m.patch_size, channel_first=True)
        .reshape(gh * gw, -1)
        .to(m.vision_model.embeddings.patch_embedding.weight.dtype)
    )
    tag = "<img>" + "<IMG_CONTEXT>" * rt.tokens + "</img>"
    return grid, pixels, tag


@torch.no_grad()
def ask_with_image(rt, image, question, cfg, system_message=None, max_new_tokens=None):
    """Text answer conditioned on one image; same canvas and normalization as scoring."""
    from sensenova_u1.models.neo_unify.conversation import get_conv_template

    m, tok = rt.model, rt.tokenizer
    grid, pixels, img_tag = _image_inputs(rt, image)
    query = m._build_t2i_query(
        img_tag + "\n" + question,
        system_message=system_message,
        append_text="<think>\n\n</think>\n\n",
    )
    encoded = tok(query, return_tensors="pt")
    separator = get_conv_template(m.template).sep.strip()
    eos = tok.convert_tokens_to_ids(separator)
    with autocast(rt.device):
        vit = m.extract_feature(pixels, gen_model=False, grid_hw=grid)
        output = m.generate(
            pixel_values=pixels,
            input_ids=encoded["input_ids"].to(rt.device),
            grid_hw=grid,
            attention_mask=encoded["attention_mask"].to(rt.device),
            visual_features=vit,
            max_new_tokens=int(max_new_tokens or cfg["und_max_new_tokens"]),
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos,
        )
    text = tok.batch_decode(output, skip_special_tokens=True)[0].split(separator)[0].strip()
    return text.rsplit("</think>", 1)[-1].strip()


DESCRIBE_SYSTEM = (
    "You describe pictures. Report only what is visible. Never guess and never "
    "assume a subject. Count every instance one by one, aloud, before you state "
    "the total."
)

DESCRIBE_TEMPLATE = (
    "<figcaption>picture</figcaption>\n"
    "Describe this picture completely: name every object, how many of each, "
    "their colour and material, and where each group sits in the frame. "
    "Count each group one instance at a time. Finish with the counts you reached."
)

FOCUSED_SCAN_TEMPLATE = (
    "<figcaption>picture</figcaption>\n"
    "Scan the entire picture carefully for any '{subject}', including small, "
    "distant, foreground/background, or partially emerged instances. List each "
    "visible instance one by one with its location and colour, or state clearly "
    "that zero instances are visible. Finish with the exact count you found."
)

JUDGE_SYSTEM = (
    "You compare a description of a picture against a target prompt and then "
    "write one instruction for a retoucher. Follow these rules literally.\n"
    "RULE A. Judge the picture, not the description: does the picture satisfy "
    "every requirement in the prompt? Answer with the single word yes or no.\n"
    "RULE B. When the answer is no, write ONE instruction using the pattern the "
    "retoucher was trained on, in this order:\n"
    "  (1) OPERATION plus the number of instances: 'Add 1 ...', 'Remove 2 ...', "
    "'Move the ...', 'Change the ... to ...'. One operation per sentence; never "
    "mix an addition and a removal in one clause.\n"
    "  (2) ENDPOINTS: '... so that it contains exactly N <object> instead of the "
    "current M'.\n"
    "  (3) PLACEMENT HINT that leaves the choice to the retoucher: for an addition "
    "write 'placing the new ones in plausible empty positions'; for a removal write "
    "'removing the <descriptor> ones and filling the vacated areas naturally'; for a "
    "move write 'relocating them to <the region>'; for a recolour or replacement "
    "write 'matching the original scale, perspective and lighting'.\n"
    "  (4) 'keep everything else unchanged' as the closing clause.\n"
    "RULE C. Name which instances when the picture makes it obvious (for example "
    "'the three rightmost in the middle row'), but never replace the placement hint "
    "with a single exact spot for an addition: naming one spot makes the retoucher "
    "rebuild the whole framing. Prefer 'plausible empty positions'.\n"
    "RULE D. Never write only the target count. 'add 3 umbrellas so that 4 becomes 7' "
    "is incomplete; 'Add 3 blue umbrellas to the image so that it contains exactly 7 "
    "blue umbrellas instead of the current 4, placing the new ones in plausible empty "
    "positions; keep everything else unchanged.' is correct.\n"
    "RULE E. Do not invent objects that are not in the description. Do not ask for "
    "any change the prompt does not require."
)

JUDGE_TEMPLATE = (
    "Target prompt: {prompt}\n\n"
    "Description of the current picture:\n{description}\n\n"
    "Reply with exactly these two lines and nothing else.\n"
    "MATCH: <yes or no>\n"
    "INSTRUCTION: <one instruction following rules A-D, or NONE when MATCH is yes>"
)


def _parse_int_token(token):
    low = str(token or "").strip().lower()
    if low.isdigit():
        return int(low)
    return NUMBER_WORDS.get(low)


def _norm_noun(phrase):
    words = re.findall(r"[a-z]+", str(phrase or "").lower())
    stop = {
        "the", "a", "an", "of", "in", "on", "at", "from", "to",
        "image", "picture", "frame", "background", "foreground",
        "left", "right", "above", "below",
    }
    words = [w for w in words if w not in stop]
    if not words:
        return ""
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


def extract_add_subject(instruction):
    """Extract the target object noun phrase from an ``Add ...`` instruction, or None."""
    text = " ".join(str(instruction or "").split()).strip()
    m = re.match(
        r"^add\s+(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(.+?)"
        r"(?:\s+(?:to|in|on|into|near|beside)\s+the\s+(?:image|picture|scene|frame|background|foreground|left|right)\b"
        r"|\s+so\s+that\b|[,.;]|$)",
        text,
        flags=re.I,
    )
    if not m:
        return None
    subj = m.group(1).strip(" ,.;:")
    return subj if subj else None


def extract_prompt_clause_counts(prompt):
    """Extract explicit positive ``{normalized_noun: count}`` clauses from a T2I prompt."""
    text = str(prompt or "").lower()
    if re.search(r"\b(without|except)\b", text):
        return {}
    cleaned = re.sub(r"^\s*(?:a|an)\s+(?:photo|picture|image|photograph|rendering|painting)\s+of\s+", "", text)
    clauses = re.split(
        r"\s+(?:and|with|,)\s+|\s+(?:on|in|to|at|above|below|under|beside|next\s+to|behind)\s+the\s+(?:left|right|top|bottom|above|below)(?:\s+of)?\s*",
        cleaned,
    )
    counts = {}
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


def check_instruction_consistency(instruction, prompt):
    """Deterministic arithmetic & prompt-count consistency gate (zero VLM confirmation bias)."""
    text = " ".join(str(instruction or "").split()).strip()
    if not text:
        return False, "empty_instruction"

    ep_re = re.compile(
        r"^(add|remove)\s+(a|an|zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(.+?)"
        r"\s+so\s+that\s+it\s+contains\s+(?:exactly\s+)?"
        r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(.+?)"
        r"\s+instead\s+of\s+the\s+current\s+"
        r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
        flags=re.I,
    )
    m = ep_re.search(text)
    if not m:
        return True, None

    op = m.group(1).lower()
    delta_k = _parse_int_token(m.group(2))
    target_n = _parse_int_token(m.group(4))
    target_obj_raw = m.group(5)
    current_m = _parse_int_token(m.group(6))
    if None in (delta_k, target_n, current_m):
        return True, None

    if op == "add":
        if target_n <= current_m or delta_k <= 0 or target_n - current_m != delta_k:
            return False, "invalid_add_endpoints"
    elif op == "remove":
        if target_n >= current_m or delta_k <= 0 or current_m - target_n != delta_k:
            return False, "invalid_remove_endpoints"

    obj_norm = _norm_noun(target_obj_raw)
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


def parse_reflection(text, description=""):
    """Parse ``MATCH`` and ``INSTRUCTION`` lines emitted by the text judge."""
    fields = {}
    for key in ("match", "instruction"):
        marker = key.upper() + ":"
        position = text.upper().find(marker)
        if position < 0:
            continue
        rest = text[position + len(marker):]
        stop = len(rest)
        for other in ("MATCH:", "INSTRUCTION:"):
            where = rest.upper().find(other)
            if 0 <= where < stop:
                stop = where
        fields[key] = rest[:stop].strip()
    verdict = fields.get("match", "").strip().lower()
    if verdict not in ("yes", "no"):
        return dict(
            description=description,
            match=None,
            instruction="",
            preserve="",
            raw=text,
            outcome="failed_parse",
            gated_by=None,
            rechecked_add_subject=None,
        )
    matched = verdict == "yes"
    instruction = fields.get("instruction", "").strip()
    if matched:
        instruction = ""
    elif instruction.upper().startswith("NONE"):
        instruction = ""
    return dict(
        description=description,
        match=matched,
        instruction=instruction,
        preserve="",
        raw=text,
        outcome=("satisfied" if matched else ("edit" if instruction else "failed_unconverted")),
        gated_by=None,
        rechecked_add_subject=None,
    )


def review_reflection(rt, image, raw, cfg):
    """Prompt-blind preview description + text judge + Breakthrough-A false-trigger gates."""
    description = ask_with_image(
        rt,
        image,
        DESCRIBE_TEMPLATE,
        cfg,
        system_message=DESCRIBE_SYSTEM,
        max_new_tokens=int(cfg.get("describe_max_new_tokens", 320)),
    )
    judgment = ask_text(
        rt,
        JUDGE_TEMPLATE.format(prompt=raw, description=description),
        cfg,
        system_message=JUDGE_SYSTEM,
        max_new_tokens=int(cfg.get("judge_max_new_tokens", 320)),
    )
    parsed = parse_reflection(judgment, description)
    initial_instruction = parsed["instruction"]

    # Gate 1: Prompt-blind focused re-inspection on tentative `Add ...` triggers
    if parsed["outcome"] == "edit" and bool(cfg.get("recheck_add_on_preview", True)):
        subject = extract_add_subject(parsed["instruction"])
        if subject:
            focused_scan = ask_with_image(
                rt,
                image,
                FOCUSED_SCAN_TEMPLATE.format(subject=subject),
                cfg,
                system_message=DESCRIBE_SYSTEM,
                max_new_tokens=160,
            )
            combined_desc = (
                description.strip()
                + f"\n\nFocused re-scan for '{subject}':\n"
                + focused_scan.strip()
            )
            judgment2 = ask_text(
                rt,
                JUDGE_TEMPLATE.format(prompt=raw, description=combined_desc),
                cfg,
                system_message=JUDGE_SYSTEM,
                max_new_tokens=int(cfg.get("judge_max_new_tokens", 320)),
            )
            reparsed = parse_reflection(judgment2, combined_desc)
            if reparsed["outcome"] not in ("failed_parse", "failed_unconverted"):
                if reparsed["outcome"] == "satisfied":
                    reparsed["gated_by"] = "focused_add_recheck"
                    reparsed["gated_instruction"] = initial_instruction
                parsed = reparsed
                description = combined_desc
                judgment = judgment2
            parsed["rechecked_add_subject"] = subject
            parsed["focused_scan"] = focused_scan.strip()

    # Gate 2: Deterministic endpoint / prompt-count consistency gate
    if parsed["outcome"] == "edit" and bool(cfg.get("gate_contradictory_edits", True)):
        ok, reason = check_instruction_consistency(parsed["instruction"], raw)
        if not ok:
            parsed["gated_instruction"] = parsed["instruction"]
            parsed["instruction"] = ""
            parsed["match"] = True
            parsed["outcome"] = "satisfied"
            parsed["gated_by"] = reason

    parsed["description"] = description.strip()
    parsed["raw"] = "DESCRIPTION:\n" + parsed["description"] + "\n\nJUDGEMENT:\n" + judgment
    parsed["judge_raw"] = judgment
    return dict(
        questions=[],
        decomposition={"response": parsed["raw"]},
        preview_score={"questions": []},
        failed=[],
        rejected=[],
        edit_request=parsed["instruction"],
        preserve="",
        atomic_edit=None,
        reflection=parsed,
        diagnosis_outcome=parsed["outcome"],
    )
