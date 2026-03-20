def projector_tag(args_or_flag):
    if hasattr(args_or_flag, "use_pretrained_proj"):
        value = getattr(args_or_flag, "use_pretrained_proj")
    else:
        value = args_or_flag
    return "preproj" if bool(int(value)) else "randproj"
