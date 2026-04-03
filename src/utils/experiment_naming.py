from src.utils.projector_utils import normalize_projector_variant


def projector_tag(args_or_flag):
    if hasattr(args_or_flag, "use_pretrained_proj"):
        value = getattr(args_or_flag, "use_pretrained_proj")
        variant = normalize_projector_variant(getattr(args_or_flag, "pretrained_proj_variant", ""))
    else:
        value = args_or_flag
        variant = ""

    if not bool(int(value)):
        return "randproj"
    if variant:
        return f"preproj-{variant}"
    return "preproj"
