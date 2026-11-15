"""Stage 1 isochronic state-flow editing."""
KIND = "u15_isochronic_state_flow_v1"
PROTOCOL_VERSION = "u15_isochronic_state_flow_v3"
CONDITION_PROTOCOL = "isochronic_condition_v2"
# v1 squeezed every landscape scene into the full square (width * ratio).
# v2 sizes a wide image as width / ratio, so the shared field of view keeps the
# aspect ratio. The two preprocessings are not interchangeable.
GEOMETRY_PROTOCOL = "shared_field_of_view_v2"
# The only geometry a checkpoint may be loaded under without an explicit opt-in.
CURRENT_GEOMETRY_PROTOCOL = GEOMETRY_PROTOCOL
GEOMETRY_PREPROCESSING = {
    "shared_field_of_view_v1": "wide scenes clamped to the square: height = width * ratio",
    "shared_field_of_view_v2": "wide scenes keep their shape: height = width / ratio",
}
LINEAR_PAIR = "linear_pair"
ROLE_L_EDIT = "L_edit"
ROLE_L_REVERSE = "L_reverse"
