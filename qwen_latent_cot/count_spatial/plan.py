"""Step 3: turn an observation into a concrete breakage and a concrete repair.

The breakage names the instance ("the one at the right"), never a bare count:
an editor asked to remove "one plum" removes whichever it likes, and the wrong
image then differs from the instruction that is supposed to describe it.

The repair sentence is what the training sample carries. It is only a
prediction at this point; Step 6 keeps it only if verification counts the
change the predictor assumed.
"""

from qwen_latent_cot.count_spatial.vlm import norm
from qwen_latent_cot.count_spatial.observe import clear_relations, countable

COUNT_REDUCE = "count_reduce"
COUNT_INCREASE = "count_increase"
SPATIAL = "spatial"
FAMILIES = (COUNT_REDUCE, COUNT_INCREASE, SPATIAL)
# The plan's own mix: counts broken in both directions, fewer spatial pairs.
FAMILY_WEIGHTS = {COUNT_REDUCE: 0.4, COUNT_INCREASE: 0.4, SPATIAL: 0.2}

_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}

# The far side of a horizontal relation, per side the subject occupies now.
OPPOSITE_SIDE = {"left": "right", "right": "left", "top left": "bottom right",
                 "bottom right": "top left", "top right": "bottom left", "bottom left": "top right",
                 "above": "below", "below": "above"}
# A move must be large enough to be visible and describable in one clause.
MIN_MOVE_STEPS = 2


def word(number):
    return _NUMBERS.get(int(number), str(int(number)))


def family_counts(limit, weights=FAMILY_WEIGHTS):
    """How many plans each family may claim, summing to ``limit`` (largest remainder)."""
    exact = {family: limit * weight for family, weight in weights.items()}
    quotas = {family: int(value) for family, value in exact.items()}
    order = sorted(exact, key=lambda family: exact[family] - quotas[family], reverse=True)
    for family in order[:limit - sum(quotas.values())]:
        quotas[family] += 1
    return quotas


def options(observation):
    """Every breakage this observation supports, family by family."""
    found = {family: [] for family in FAMILIES}
    for cand in countable(observation):
        if cand["count"] >= 2:
            found[COUNT_REDUCE].append(cand)
        found[COUNT_INCREASE].append(cand)
    for rel in clear_relations(observation):
        found[SPATIAL].append(rel)
    return found


def choose(reference, observation, index, quotas=None, *, preserve_background=True):
    """One concrete plan for an observed image, or None when none is possible.

    ``quotas`` caps each family's remaining allowance; families are tried in
    this image's rotation order so the mix is filled rather than starved.
    """
    found = options(observation)
    order = [family for family in FAMILIES if found[family]]
    if not order:
        return None
    order = order[index % len(order):] + order[:index % len(order)]
    for family in order:
        if quotas is not None and quotas.get(family, 0) <= 0:
            continue
        items = found[family]
        item = items[(index // len(FAMILIES)) % len(items)]
        plan = _build(reference, observation, family, item, preserve_background=preserve_background)
        if plan is not None:
            return plan
    return None


def _build(reference, observation, family, item, *, preserve_background=True):
    if family == COUNT_REDUCE:
        count = int(item["count"])
        where = _instance(item["positions"])
        keep_tail = "; keep everything else unchanged." if preserve_background else "."
        return dict(
            edit_family=family, task="count", semantic_task=COUNT_REDUCE,
            count_direction="increase", object=item["object"],
            count_source=count - 1, count_predicted=count,
            breakage_instruction="Remove one of the %s%s. Keep everything else in the scene unchanged."
                                 % (item["object"], where),
            repair_instruction=(
                "Add 1 %s to the image so that it contains exactly %s %s instead of the current %s, "
                "placing it%s%s"
                % (item["object"], word(count), item["object"], word(count - 1), where, keep_tail)),
        )
    if family == COUNT_INCREASE:
        # The kept pair is (wrong with K, target with K+1): the third picture is
        # generated with one more instance, and the instruction names the
        # picture that holds MORE, because "Add N ... exactly M" can only
        # describe a target that has more than its source.
        count = int(item["count"])
        keep_tail = "; keep everything else unchanged." if preserve_background else "."
        return dict(
            edit_family=family, task="count", semantic_task=COUNT_INCREASE,
            count_direction="increase", object=item["object"],
            count_source=count, count_predicted=count + 1,
            breakage_instruction=(
                "Add one %s in a plausible empty position. Keep everything else in the scene unchanged."
                % item["object"]),
            repair_instruction=(
                "Add 1 %s to the image so that it contains exactly %s %s instead of the current %s, "
                "placing it at the %s%s"
                % (item["object"], word(count + 1), item["object"], word(count),
                   (item["positions"][-1] if item["positions"] else "original position"), keep_tail)),
        )
    subject, relation, subject_of = item["subject"], item["relation"], item["reference"]
    here, there = move_sides(subject, relation, subject_of, observation)
    if here is None:
        return None
    before = ("to the %s of" % here) if here in ("left", "right") else here
    after = ("to the %s side of" % there) if there in ("left", "right") else there
    keep_tail = "; keep everything else unchanged." if preserve_background else "."
    return dict(
        edit_family=SPATIAL, task="spatial", semantic_task=SPATIAL, count_direction=None,
        subject=subject, relation=relation, relation_target=swap(relation), reference=subject_of,
        subject_side=here, destination=there,
        breakage_instruction=(
            "Move only the %s %s the %s. The %s must stay exactly in its original position. "
            "Keep the background and everything else unchanged."
            % (subject, after, subject_of, subject_of)),
        repair_instruction="Move the %s back %s the %s%s"
                           % (subject, before, subject_of, keep_tail),
    )


def swap(relation):
    """The opposite relation, or None where there is no single opposite."""
    return {"left of": "right of", "right of": "left of", "above": "below", "below": "above",
            "in front of": "behind", "behind": "in front of"}.get(relation)


def _instance(positions):
    positions = [p for p in positions if p]
    if not positions:
        return ""
    return " the one at the %s" % positions[-1] if len(positions) > 1 else " the one at the %s" % positions[0]


def move_sides(subject, relation, reference, observation):
    """Current side and destination, both sideways enough to describe in words.

    Only uses sides whose opposite is known: "above"/"below" would need a
    vertical destination the editor could put in the wrong place, so a
    relation is only planned when its observed side is horizontal.
    """
    if relation not in ("left of", "right of", "above", "below"):
        return None, None
    side = {"left of": "left", "right of": "right", "above": "above", "below": "below"}[relation]
    if side not in OPPOSITE_SIDE:
        return None, None
    destination = OPPOSITE_SIDE[side]
    return side, destination


def relation_side(observation, subject, relation, reference):
    """Which side the observation puts the subject on, relative to the reference.

    The observation's own relation list is authoritative: it was recorded by
    looking at the image. A relation it lists for another pair is not used.
    """
    for rel in observation.get("spatial_relations") or []:
        if (norm(rel.get("subject")) == subject and norm(rel.get("reference")) == reference
                and norm(rel.get("relation")) == relation and rel.get("clear", True)):
            # The observed horizontal position of the subject decides the side,
            # because "left of" only says which way, not how far.
            return _side_of(observation, subject, relation)
    return None


def _side_of(observation, subject, relation):
    horizontal = {"left": "left", "right": "right",
                  "top left": "left", "bottom left": "left",
                  "top right": "right", "bottom right": "right"}
    for obj in observation.get("objects") or []:
        if norm(obj.get("name")) != subject:
            continue
        for position in obj.get("positions") or []:
            side = horizontal.get(norm(position))
            if side is not None:
                return side
    return None


def verify(plan, observation):
    """What the edited image must show for this plan to be a valid wrong source."""
    if plan["task"] == "count":
        return dict(kind="count", object=plan["object"], expected=plan["count_source"])
    return dict(kind="spatial", subject=plan["subject"], reference=plan["reference"],
                relation=plan["relation_target"])
