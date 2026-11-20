"""Step 2: structured scene observation for count and spatial pair planning."""

from qwen_latent_cot.count_spatial.vlm import norm

POSITIONS = (
    "left", "center", "right", "top", "bottom",
    "top left", "top right", "bottom left", "bottom right",
)
MAX_COUNTABLE = 8

OBSERVE_PROMPT = """You are inspecting one image so that an image editor can later make it wrong by changing a count or a spatial relation.

Report only what you can clearly see. Never report an object, a count, or a relation you cannot actually verify, and never infer objects from general knowledge of such scenes.

Return compact JSON only (single object, no markdown commentary):
{
  "objects": [{"name": "<singular noun, lower case>", "count": <int>, "positions": ["<POSITION>..."],
               "countable": <true if each instance is distinct and countable>}],
  "layout": "<one short sentence describing where the main objects sit>",
  "count_candidates": [{"object": "<name>", "count": <int>, "reason": "<why its count is clear and worth editing>"}],
  "spatial_relations": [{"subject": "<name>", "relation": "left of|right of|above|below|beside|in front of|behind|on|inside",
                         "reference": "<name>", "clear": <true if the relation is unambiguous>}],
  "quality": "<good|acceptable|poor>",
  "quality_notes": "<what is wrong with the image, if anything>"
}

POSITION must be one of: %s.
Focus on prominent foreground and mid-ground objects rather than tiny background clutter. A count_candidate must be a clearly separated object group with between 2 and 8 instances. Clear spatial_relations are pairs of distinct foreground/mid-ground objects whose left/right/above/below direction is obvious at a glance.
""" % ", ".join(POSITIONS)


def check_observation(value, require_candidates=True):
    """Reject a reply shape that cannot be turned into a plan."""
    keys = ("objects", "layout", "spatial_relations", "quality")
    if require_candidates:
        keys += ("count_candidates",)
    for key in keys:
        if key not in value:
            raise ValueError("observation lacks " + key)
    if not isinstance(value["objects"], list) or (require_candidates and not isinstance(value["count_candidates"], list)):
        raise ValueError("observation objects and count_candidates must be lists")
    if not isinstance(value["spatial_relations"], list):
        raise ValueError("observation spatial_relations must be a list")
    if norm(value["quality"]) not in ("good", "acceptable", "poor"):
        raise ValueError("observation quality is not one of good, acceptable, poor")
    for obj in value["objects"]:
        if not norm(obj.get("name")) or not isinstance(obj.get("count"), int):
            raise ValueError("an object lacks a name or an integer count")
    for rel in value["spatial_relations"]:
        if not norm(rel.get("subject")) or not norm(rel.get("reference")) or not norm(rel.get("relation")):
            raise ValueError("a relation lacks a subject, relation, or reference")


def suitable(observation, min_count=2, max_count=MAX_COUNTABLE):
    """Whether this image offers a clean, verifiable count (2..8) or spatial edit."""
    if norm(observation["quality"]) != "good":
        return False
    return bool(countable(observation, min_count=min_count, max_count=max_count) or clear_relations(observation))


def counts_by_name(observation):
    found = {}
    for obj in observation["objects"]:
        found[norm(obj["name"])] = max(found.get(norm(obj["name"]), 0), int(obj["count"]))
    return found


def countable(observation, min_count=2, max_count=MAX_COUNTABLE):
    """Count candidates kept only when countable, within [min_count, max_count], and consistent."""
    plain = counts_by_name(observation)
    kept = []
    for cand in observation["count_candidates"]:
        name = norm(cand.get("object"))
        count = plain.get(name, 0)
        seen = [obj for obj in observation["objects"] if norm(obj["name"]) == name]
        if seen and not all(obj.get("countable", True) for obj in seen):
            continue
        if min_count <= count <= max_count and (cand.get("count") in (None, count)):
            kept.append(dict(object=name, count=count, positions=_positions(observation, name)))
    return kept


def _positions(observation, name):
    for obj in observation["objects"]:
        if norm(obj["name"]) == name:
            return [norm(p) for p in obj.get("positions") or [] if norm(p) in POSITIONS]
    return []


def clear_relations(observation):
    """Unambiguous relations whose two ends are both counted in the image."""
    plain = counts_by_name(observation)
    kept = []
    for rel in observation["spatial_relations"]:
        subject, reference = norm(rel["subject"]), norm(rel["reference"])
        if subject == reference or not rel.get("clear", True):
            continue
        if subject in plain and reference in plain:
            kept.append(dict(subject=subject, relation=norm(rel["relation"]), reference=reference))
    return kept


def positions_of(observation, name):
    """Every recorded position for an object, in the order the model listed them."""
    for obj in observation.get("objects") or []:
        if norm(obj.get("name")) == name:
            return [norm(p) for p in obj.get("positions") or []]
    return []


def boxes_of(observation, name):
    """Observed coarse normalized boxes and confidence for ``name`` on this image."""
    for obj in observation.get("objects") or []:
        if norm(obj.get("name")) == name:
            boxes = obj.get("boxes") or []
            conf = float(obj.get("box_confidence", 1.0 if boxes else 0.0))
            return [list(map(float, b)) for b in boxes], conf
    return [], 0.0


def relation_holds(observation, subject, relation, reference):
    """Whether the observation records this exact relation as clear."""
    wanted = norm(relation)
    for rel in observation.get("spatial_relations") or []:
        if norm(rel.get("subject")) != subject or norm(rel.get("reference")) != reference:
            continue
        if norm(rel.get("relation")) == wanted and rel.get("clear", True):
            return True
    return False
