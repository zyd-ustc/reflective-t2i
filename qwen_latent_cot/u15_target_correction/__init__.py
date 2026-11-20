"""Strict isochronic manifold-transition LatentCoT editing."""

KIND = "u15_isochronic_state_flow_v1"
PROTOCOL_VERSION = "u15_isochronic_state_flow_v3"
GEOMETRY_PROTOCOL = "shared_field_of_view_v2"
CURRENT_GEOMETRY_PROTOCOL = GEOMETRY_PROTOCOL
GEOMETRY_PREPROCESSING = {
    "shared_field_of_view_v1": "wide scenes clamped to the square: height = width * ratio",
    "shared_field_of_view_v2": "wide scenes keep their shape: height = width / ratio",
}
LINEAR_PAIR = "linear_pair"
ROLE_L_EDIT = "L_edit"
EDITOR_INTERFACE = "isochronic_manifold_v1"
COMPAT_CHECKPOINT_INTERFACES = (EDITOR_INTERFACE, "semantic_denoising_v1")
CONDITION_PROTOCOL = "semantic_feedback_only_v1"
PARAMETER_LAYOUT = "semantic_gen_lora_all_layers"
