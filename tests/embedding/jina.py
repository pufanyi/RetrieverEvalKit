from pathlib import Path

from PIL import Image

from img_search.embedding import get_encoder

if __name__ == "__main__":
    img_path = Path(__file__).parents[1] / "imgs" / "image.png"
    img = Image.open(img_path)

    encoder = get_encoder("jina_v4")
    encoder.build()
    embedding = encoder.encode(image=img)
    print(embedding)

    text_embedding = encoder.encode(
        text="A big white dog with a small yellow dog", prompt_name="query"
    )
    print(encoder.model.similarity(text_embedding, embedding))

    text_embedding = encoder.encode(
        text="A big yellow dog with a small white dog", prompt_name="query"
    )
    print(encoder.model.similarity(text_embedding, embedding))

    # Test image encoding
    img_embedding = encoder.encode(text="Dogs", prompt_name="query")
    print(encoder.model.similarity(img_embedding, embedding))
