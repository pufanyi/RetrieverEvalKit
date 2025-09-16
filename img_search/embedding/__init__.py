from .encoder import Encoder
from .jina_v4 import JinaV4Encoder
from .siglip2 import Siglip2Encoder

__all__ = ["Siglip2Encoder", "JinaV4Encoder"]


def get_encoder(model: str, **kwargs) -> Encoder:
    if model == "siglip2":
        return Siglip2Encoder(**kwargs)
    elif model == "jina_v4":
        return JinaV4Encoder(**kwargs)
    else:
        raise ValueError(f"Model {model} not found")
