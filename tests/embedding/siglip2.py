from pathlib import Path

from PIL import Image

from img_serach.embedding import get_encoder

if __name__ == "__main__":
    img_path = Path(__file__).parents[1] / "imgs" / "image.png"
    img = Image.open(img_path)

    encoder = get_encoder("siglip2")
    encoder.build()
    embedding = encoder.encode(img)
    print(embedding)
