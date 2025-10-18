from pathlib import Path

import pytest
from omegaconf import OmegaConf

from img_search.embedding import get_encoder

Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def siglip2_encoder():
    cfg = OmegaConf.create(
        {
            "model": "siglip2",
            "model_name": "google/siglip2-base-patch32-256",
            "data_parallel": False,
        }
    )
    encoder = get_encoder(cfg)
    encoder.build()
    return encoder


@pytest.fixture(scope="module")
def sample_image() -> Image.Image:
    img_path = Path(__file__).parents[1] / "imgs" / "image.png"
    return Image.open(img_path)


def _vision_hidden_size(encoder) -> int:
    model = encoder.model
    config = (
        model.module.config
        if isinstance(model, torch.nn.DataParallel)
        else model.config
    )
    return config.vision_config.hidden_size


def test_siglip2_encode_image(siglip2_encoder, sample_image: Image.Image):
    """Verify that encoding an image returns an embedding of correct shape and type."""
    embedding = siglip2_encoder.encode(image=sample_image)

    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (_vision_hidden_size(siglip2_encoder),)
    assert embedding.dtype == torch.float32


def test_siglip2_encode_text(siglip2_encoder):
    """Verify that encoding text returns an embedding of correct shape and type."""
    text_embedding = siglip2_encoder.encode(
        text="A big white dog with a small yellow dog"
    )

    assert isinstance(text_embedding, torch.Tensor)
    assert text_embedding.shape == (_vision_hidden_size(siglip2_encoder),)
    assert text_embedding.dtype == torch.float32
