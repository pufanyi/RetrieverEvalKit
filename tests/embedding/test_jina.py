
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from img_search.embedding import get_encoder

Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def jina_encoder():
    cfg = OmegaConf.create({
        "model": "jina_v4",
        "model_name": "jinaai/jina-embeddings-v4",
        "batch_size": 64
    })
    encoder = get_encoder(cfg)
    encoder.build()
    return encoder


@pytest.fixture(scope="module")
def sample_image() -> Image.Image:
    img_path = Path(__file__).parents[1] / "imgs" / "image.png"
    return Image.open(img_path)


def test_jina_encode_image(jina_encoder, sample_image: Image.Image):
    """Verify that encoding an image returns an embedding with the correct shape and type."""
    embedding = jina_encoder.encode(image=sample_image)

    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (2048,)
    assert embedding.dtype == torch.float32


def test_jina_encode_text(jina_encoder):
    """Verify that encoding text returns an embedding with the correct shape and type."""
    text_embedding = jina_encoder.encode(
        text="A big white dog with a small yellow dog", prompt_name="query"
    )

    assert isinstance(text_embedding, torch.Tensor)
    assert text_embedding.shape == (2048,)
    assert text_embedding.dtype == torch.float32


def test_jina_similarity(jina_encoder, sample_image: Image.Image):
    """Test similarity scores between text and image embeddings."""
    image_embedding = jina_encoder.encode(image=sample_image)
    
    text_embedding_positive = jina_encoder.encode(
        text="A big white dog with a small yellow dog", prompt_name="query"
    )
    text_embedding_negative = jina_encoder.encode(
        text="A picture of a cat", prompt_name="query"
    )

    similarity_positive = jina_encoder.model.similarity(text_embedding_positive, image_embedding)
    similarity_negative = jina_encoder.model.similarity(text_embedding_negative, image_embedding)

    assert isinstance(similarity_positive, torch.Tensor)
    assert similarity_positive.item() > 0.5
    assert similarity_negative.item() < 0.5
    assert similarity_positive.item() > similarity_negative.item()
