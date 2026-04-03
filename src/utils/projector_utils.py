from pathlib import Path
from typing import Optional

from src.utils.model_utils import normalize_model_name


_PROJECTOR_VARIANT_ALIASES = {
    "": "",
    "default": "",
    "base": "",
    "full": "",
    "cl_rmg": "",
    "clrmg": "",
    "cl_only": "clonly",
    "clonly": "clonly",
    "contrastive_only": "clonly",
    "contrastive": "clonly",
    "rmg_only": "rmgonly",
    "rmgonly": "rmgonly",
    "max_margin": "maxmargin",
    "maxmargin": "maxmargin",
    "margin": "maxmargin",
}


def normalize_projector_variant(variant: Optional[str]) -> str:
    if variant is None:
        return ""
    key = str(variant).strip().lower().replace("-", "_")
    return _PROJECTOR_VARIANT_ALIASES.get(key, key)


def projector_variant_suffix(variant: Optional[str]) -> str:
    normalized = normalize_projector_variant(variant)
    return f"_{normalized}" if normalized else ""


def projector_variant_label(variant: Optional[str]) -> str:
    normalized = normalize_projector_variant(variant)
    return normalized if normalized else "default"


def projector_checkpoint_name(projector_name: str, model_name: str, emb_dim: int, variant: Optional[str] = "") -> str:
    normalized_model = normalize_model_name(model_name)
    return f"{projector_name}_{normalized_model}_{emb_dim}{projector_variant_suffix(variant)}.pth"


def resolve_projector_weight_path(
    model_name: str,
    emb_dim: int,
    projector_name: str = "mlp+norm",
    variant: Optional[str] = "",
    override_path: Optional[str] = "",
) -> Path:
    if override_path:
        candidate = Path(override_path).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Specified pretrained projector path does not exist: {candidate}")

    normalized_model = normalize_model_name(model_name)
    projector_dir = Path(__file__).resolve().parents[2] / "saved" / "projector_weights"
    normalized_variant = normalize_projector_variant(variant)

    if normalized_variant:
        candidates = [
            projector_dir / f"{projector_name}_{normalized_model}_{emb_dim}_{normalized_variant}.pth",
        ]
    else:
        candidates = [
            projector_dir / f"{projector_name}_{normalized_model}_{emb_dim}.pth",
            projector_dir / f"{projector_name}{emb_dim}.pth",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_str = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Pretrained projector weights not found. Tried: {candidate_str}")
