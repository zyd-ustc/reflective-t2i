"""Unit tests for scripts/curate_edit_pool_glm.py."""

from scripts.curate_edit_pool_glm import check_glm_verdict, classify_candidate_bucket


def test_classify_candidate_bucket_filters_pollution_and_maps_buckets():
    assert classify_candidate_bucket("Add 2 cups so that it contains 4 cups instead of the current 2.", "T8") == "count_add"
    assert classify_candidate_bucket("Add a blue backpack on the bench.", "T3") == "object_add"
    assert classify_candidate_bucket("Remove 2 black keyboards so that it contains 4.", "T8") == "count_remove"
    assert classify_candidate_bucket("Remove the umbrella on the left.", "T3") == "object_remove"
    assert classify_candidate_bucket("Move the dog to the left of the chair.", "T9") == "spatial_move"
    assert classify_candidate_bucket("Change the colour of the table to red.", "T5") == "attr_change"

    # Rejects makeup, weather, background replacement, style transfer, OCR text, broken nouns
    assert classify_candidate_bucket("Apply bold evening makeup to the person: deep red lipstick.", "T2") is None
    assert classify_candidate_bucket("Add a rainbow in the background.") is None
    assert classify_candidate_bucket("Change the season to winter.") is None
    assert classify_candidate_bucket("Restyle the whole image as ink wash painting.", "T6") is None
    assert classify_candidate_bucket("Remove 'PLAZA PLAZA' from the roof sign.") is None
    assert classify_candidate_bucket("Add 3 times to the image so that it contains exactly 4 times.", "T8") is None

    check_glm_verdict({"task_type": "count_add", "semantic_pass": True, "reason": "ok"})
