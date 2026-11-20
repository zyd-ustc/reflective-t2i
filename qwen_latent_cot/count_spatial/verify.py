"""Step 5 and Step 6: did the image move exactly as planned, and what does it now say?

Verification is pairwise and fails closed: a relation the model did not record
cannot be confirmed, and an object it did not report is treated as an object
whose count is unknown, not as an unchanged one.

The training instruction is then written from the measured counts, not copied
from the plan. The plan proposed a sentence; the pictures decide the numbers
in it.
"""

from qwen_latent_cot.count_spatial.vlm import norm
from qwen_latent_cot.count_spatial.observe import boxes_of, counts_by_name, positions_of, relation_holds
from qwen_latent_cot.count_spatial.plan import SPATIAL, OPPOSITE_SIDE

# What may change beyond the target. A picture whose background was redrawn
# teaches a redraw, not an edit, so any third object changing is a failure.
MISMATCH_LIMIT = 0


def check(pair, plan, *, strict_preservation=True):
    """The verdict plus every item that produced it."""
    checks, reasons = [], []

    def note(ok, what):
        checks.append(dict(check=what, ok=bool(ok)))
        if not ok:
            reasons.append(what)

    target, wrong = pair.get("target"), pair.get("wrong")
    if not target or not wrong:
        note(False, "both images observed")
        return _result(plan, checks, reasons)
    note(norm(target.get("quality")) == "good", "target image quality")
    note(norm(wrong.get("quality")) != "poor", "wrong image quality")

    target_counts, wrong_counts = counts_by_name(target), counts_by_name(wrong)
    if plan["task"] == "count":
        wanted = plan["object"]
        target_count, wrong_count = target_counts.get(wanted), wrong_counts.get(wanted)
        if strict_preservation:
            note(target_count == plan["count_predicted"], "target count is the observed %s" % plan["count_predicted"])
            note(wrong_count == plan["count_source"], "wrong count is the planned %s" % plan["count_source"])
            note(target_count is not None and wrong_count is not None and target_count != wrong_count,
                 "counts actually differ")
        else:
            expected_delta = int(plan["count_predicted"]) - int(plan["count_source"])
            note(target_count is not None and wrong_count is not None, "target object observed in both images")
            note(
                target_count is not None
                and wrong_count is not None
                and wrong_count >= 1
                and (target_count - wrong_count) == expected_delta,
                "measured count delta matches +%s" % expected_delta,
            )
    else:
        target_count = wrong_count = None
        subject, reference = plan["subject"], plan["reference"]
        both_present = (
            target_counts.get(subject, 0) >= 1
            and wrong_counts.get(subject, 0) >= 1
            and target_counts.get(reference, 0) >= 1
            and wrong_counts.get(reference, 0) >= 1
        )
        note(both_present, "subject and reference present in both images")
        # The observation records one direction per pair, and which one is not
        # the model's to choose: a pair may be listed as "cup left of plate" or
        # "plate right of cup". Both readings of the recorded relation are
        # accepted; what must flip is the arrangement, read from positions.
        recorded = _holds(target, subject, plan["relation"], reference)
        flipped = _holds(wrong, subject, plan["relation_target"], reference)
        moved = (_opposite_side(target, wrong, subject, reference, plan["relation"])
                 and not _holds(wrong, subject, plan["relation"], reference))
        note(recorded, "target shows the recorded relation")
        # An unrecorded relation is not an unchanged one: the model may simply
        # not have listed the pair again. A side that plainly changed counts.
        note(flipped or moved, "wrong image shows the opposite relation")
        if strict_preservation:
            reference_before = set(positions_of(target, reference))
            reference_after = set(positions_of(wrong, reference))
            reference_moved = any(
                len(reference_before & sides) == len(reference_after & sides) == 1
                and (reference_before & sides) != (reference_after & sides)
                for sides in ({"left", "right"}, {"top", "bottom"})
            )
            note(not reference_moved, "reference object stayed in place")

    kept = set(target_counts) & set(wrong_counts)
    changed = sorted(name for name in kept - {plan.get("object")}
                     if target_counts[name] != wrong_counts[name])
    missing = sorted(((set(target_counts) | set(wrong_counts)) - kept) - {plan.get("object")})
    if strict_preservation:
        note(len(changed) <= MISMATCH_LIMIT, "unrelated object counts held")
        note(not missing, "unrelated objects observed in both images")
    return _result(plan, checks, reasons, dict(
        target=target, wrong=wrong, target_count=target_count, wrong_count=wrong_count,
        unrelated_count_changes=changed,
        dropped_objects=sorted((set(target_counts) | set(wrong_counts)) - kept)))


def _holds(observation, subject, relation, reference):
    """A recorded relation in either direction of the same pair."""
    opposite = {"left of": "right of", "right of": "left of", "above": "below", "below": "above"}
    return (relation_holds(observation, subject, relation, reference)
            or relation_holds(observation, reference, opposite.get(relation, ""), subject))


def _opposite_side(target, wrong, subject, reference, relation):
    """Whether the subject sits on one side of the reference in one picture only.

    Positions are what the model saw, so a side that changes between the two
    pictures is a move whether or not the relation list was exhaustive.
    """
    if relation in ("above", "below"):
        axis = {"top": 0, "top left": 0, "top right": 0, "center": 1,
                "bottom": 2, "bottom left": 2, "bottom right": 2}
    else:
        axis = {"left": 0, "top left": 0, "bottom left": 0, "center": 1,
                "right": 2, "top right": 2, "bottom right": 2}
    def relative(observation):
        subject_positions = {axis[p] for p in positions_of(observation, subject) if p in axis}
        reference_positions = {axis[p] for p in positions_of(observation, reference) if p in axis}
        if len(subject_positions) != 1 or len(reference_positions) != 1:
            return 0
        return next(iter(subject_positions)) - next(iter(reference_positions))
    return relative(target) * relative(wrong) < 0


def _result(plan, checks, reasons, extra=None):
    record = dict(passed=not reasons, reasons=reasons, checks=checks,
                  family=plan["edit_family"], plan=plan)
    record.update(extra or {})
    return record


def instruction(verification, *, preserve_background=True):
    """The training instruction, rebuilt around what verification measured."""
    plan = verification["plan"]
    if not verification["passed"]:
        return None
    if plan["edit_family"] == SPATIAL:
        # A move is described by its relation, not by a count sentence. The
        # check is on the family: the count families' semantic_task names
        # ("count_reduce"/"count_increase") never collide with it.
        return _spatial_instruction(verification, plan, preserve_background=preserve_background)
    source, goal = verification["wrong_count"], verification["target_count"]
    if source is None or goal is None or goal <= source:
        return None
    object_ = plan["object"]
    suffix = "; keep everything else unchanged." if preserve_background else "."
    return ("Add %d %s to the image so that it contains exactly %d %s instead of the current %d, "
            "placing the new ones in plausible empty positions%s"
            % (goal - source, object_, goal, object_, source, suffix))


def _spatial_instruction(verification, plan, *, preserve_background=True):
    """Where the subject sits in the corrected picture, read from that picture."""
    side = _observed_side(verification, plan)
    if side is None:
        return None
    direction = ("to the %s of" % side) if side in ("left", "right") else side
    suffix = "; keep the rest of the image unchanged." if preserve_background else "."
    return ("Move the %s %s the %s%s" % (plan["subject"], direction, plan["reference"], suffix))


def edit_semantics(verification):
    """Layer-1 verified structured semantics comparing post-generation source vs target."""
    plan = verification["plan"]
    if plan["edit_family"] == SPATIAL:
        side = _observed_side(verification, plan)
        return dict(
            edit_action="move",
            edit_object=plan["subject"],
            before_state={"relation": plan["relation_target"], "reference": plan["reference"]},
            after_state={"relation": plan["relation"], "reference": plan["reference"], "side": side},
            instruction_matches_change=bool(verification["passed"]),
        )
    return dict(
        edit_action="add",
        edit_object=plan["object"],
        before_state={"count": verification.get("wrong_count")},
        after_state={"count": verification.get("target_count")},
        instruction_matches_change=bool(verification["passed"]),
    )


def _pad_valid_boxes(boxes, pad):
    valid = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            continue
        area = (x2 - x1) * (y2 - y1)
        if not 0.004 <= area <= 0.90:
            continue
        valid.append([
            round(max(0.0, x1 - pad), 4),
            round(max(0.0, y1 - pad), 4),
            round(min(1.0, x2 + pad), 4),
            round(min(1.0, y2 + pad), 4),
        ])
    return valid


def coarse_edit_region(verification, *, pad=0.06, min_confidence=0.5):
    """Layer-2 coarse edit region built strictly from post-generation observations."""
    plan = verification["plan"]
    target_obs = verification.get("target") or {}
    wrong_obs = verification.get("wrong") or {}
    if plan["edit_family"] == SPATIAL:
        subj = plan["subject"]
        boxes_src, conf_src = boxes_of(wrong_obs, subj)
        boxes_tgt, conf_tgt = boxes_of(target_obs, subj)
        raw_boxes = boxes_src + boxes_tgt
        conf = min(conf_src, conf_tgt) if (boxes_src and boxes_tgt) else 0.0
        space = "union_source_target"
        has_both = bool(boxes_src and boxes_tgt)
    else:
        obj = plan["object"]
        raw_boxes, conf = boxes_of(target_obs, obj)
        space = "target"
        has_both = bool(raw_boxes)

    padded = _pad_valid_boxes(raw_boxes, float(pad))
    verified = bool(verification.get("passed") and has_both and padded and conf >= float(min_confidence))
    return dict(
        boxes=padded if verified else [],
        image_space=space,
        before_crop=True,
        confidence=round(float(conf), 4),
        verified=verified,
    )


def _observed_side(verification, plan):
    if _holds(verification.get("target") or {}, plan["subject"], plan["relation"], plan["reference"]):
        return {"left of": "left", "right of": "right", "above": "above", "below": "below"}.get(plan["relation"])
    positions = []
    for obj in (verification.get("target") or {}).get("objects") or []:
        if norm(obj.get("name")) == plan["subject"]:
            positions = [norm(p) for p in obj.get("positions") or []]
            break
    for position in positions:
        if position in OPPOSITE_SIDE:
            return position
    # Fall back to the relation's own word only when the target recorded it.
    if relation_holds(verification.get("target") or {}, plan["subject"], plan["relation"], plan["reference"]):
        return {"left of": "left", "right of": "right"}.get(plan["relation"])
    return None
