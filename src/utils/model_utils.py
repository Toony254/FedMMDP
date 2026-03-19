from typing import Optional

EMBEDDING_MODELS = ("clip", "align", "siglip")
MODEL_CHOICES = EMBEDDING_MODELS + ("resnet",)


def normalize_model_name(model_name: Optional[str]) -> str:
    if model_name is None:
        return "clip"
    return str(model_name).strip().lower()


def is_embedding_model(model_name: Optional[str]) -> bool:
    return normalize_model_name(model_name) in EMBEDDING_MODELS


def validate_model_name(model_name: Optional[str]) -> str:
    normalized = normalize_model_name(model_name)
    if normalized not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model: {model_name}. Expected one of {MODEL_CHOICES}")
    return normalized
