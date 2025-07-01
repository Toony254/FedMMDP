__all__ = ['get_model']

from src.networks.models.pcme import PCME


def get_model(config, mlp_local):
    return PCME(config, mlp_local)
