from .encoder import Encoder
from .siglip2 import Siglip2Encoder

__all__ = ["Siglip2Encoder"]


def get_encoder(model: str, **kwargs) -> Encoder:
    if model == "siglip2":
        return Siglip2Encoder(**kwargs)
    else:
        raise ValueError(f"Model {model} not found")
