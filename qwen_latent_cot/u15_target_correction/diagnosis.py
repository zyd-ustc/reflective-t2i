"""Three-step UND reflection over the actual preview image."""

import torch
from .runtime import autocast

@torch.no_grad()
def ask_text(rt, question, cfg, system_message=None, max_new_tokens=None):
    """Text-only answer, with no image anywhere in the call.

    The judge uses this so it cannot read the picture again. ``_build_t2i_query``
    with no ``<image>`` placeholder gives the model's own text template, and the
    language model generates from token embeddings, which is the path that
    produces clean labelled output here.
    """
    from sensenova_u1.models.neo_unify.conversation import get_conv_template

    m, tok = rt.model, rt.tokenizer
    query = m._build_t2i_query(question, system_message=system_message,
                               append_text="<think>\n\n</think>\n\n")
    encoded = tok(query, return_tensors="pt")
    input_ids = encoded["input_ids"].to(rt.device)
    # The language model rebuilds its own block-causal mask from these indexes,
    # so a text-only call still has to supply them.
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


# Step 1 looks at the picture alone. The prompt is deliberately absent: a
# describer that already knows the answer reads what it expects, which is the
# counting failure this whole loop exists to fix.
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

# Step 2 judges the description against the prompt; step 3 writes the edit.
# The wording comes from the training labels themselves. Measured on one phoenix
# scene, the album phrasing ("placing the new ones in plausible empty positions;
# keep everything else unchanged") added a second phoenix from both the real
# photo and the model's own trajectory, while a hand-written instruction that
# named an exact spot instead made the editor redraw the whole composition. The
# lesson is that the model was trained to CHOOSE the empty space and to leave
# everything else alone, so the instruction should say that, not place the
# object for it.
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

def parse_reflection(text, description=""):
    """Split the two labelled lines; the text is never interpreted.

    Only label positions are used, so a model that writes extra prose still
    parses as long as it emits the labels.
    """
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
        return dict(description=description, match=None, instruction="", preserve="",
                    raw=text, outcome="failed_parse")
    matched = verdict == "yes"
    instruction = fields.get("instruction", "").strip()
    if matched:
        instruction = ""
    elif instruction.upper().startswith("NONE"):
        instruction = ""
    return dict(description=description, match=matched, instruction=instruction,
                preserve="", raw=text, outcome=("satisfied" if matched else
                                                ("edit" if instruction else "failed_unconverted")))


def review_reflection(rt, image, raw, cfg):
    """Describe the picture alone, then judge it and write one instruction.

    Two calls on purpose. The describe call is never given the prompt, so the
    counts come from the pixels; the judge call never sees the image again, so
    it can only reason about the description it was handed.
    """
    description = ask_with_image(
        rt, image, DESCRIBE_TEMPLATE, cfg, system_message=DESCRIBE_SYSTEM,
        max_new_tokens=int(cfg.get('describe_max_new_tokens', 320)))
    # The judge never sees the picture: a blank canvas is still an image, and
    # the model reads it as "the frame really is empty", then judges every case
    # as unsatisfied. Text only.
    judgment = ask_text(
        rt, JUDGE_TEMPLATE.format(prompt=raw, description=description), cfg,
        system_message=JUDGE_SYSTEM,
        max_new_tokens=int(cfg.get('judge_max_new_tokens', 320)))
    parsed = parse_reflection(judgment, description)
    parsed['description'] = description.strip()
    parsed['raw'] = "DESCRIPTION:\n" + description + "\n\nJUDGEMENT:\n" + judgment
    parsed['judge_raw'] = judgment
    return dict(questions=[], decomposition={"response": parsed['raw']},
                preview_score={"questions": []}, failed=[], rejected=[],
                edit_request=parsed['instruction'], preserve=parsed['preserve'],
                atomic_edit=None, reflection=parsed, diagnosis_outcome=parsed['outcome'])
