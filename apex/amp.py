from contextlib import contextmanager


def initialize(models=None, optimizers=None, **kwargs):
    return models, optimizers


@contextmanager
def scale_loss(loss, optimizer):
    yield loss
