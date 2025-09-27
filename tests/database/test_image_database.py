import io
import sys
from pathlib import Path

import pytest

SITE_PACKAGES = (
    Path(__file__).resolve().parents[2]
    / ".venv"
    / f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if SITE_PACKAGES.exists():
    site_path = str(SITE_PACKAGES)
    if site_path not in sys.path:
        sys.path.append(site_path)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
root_path = str(PROJECT_ROOT)
if root_path not in sys.path:
    sys.path.append(root_path)

pytest.importorskip("PIL")
from PIL import Image

from img_search.database.images import ImageDatabase


@pytest.fixture()
def temp_db(tmp_path: Path) -> ImageDatabase:
    db_path = tmp_path / "images.lmdb"
    with ImageDatabase(db_path) as db:
        yield db
    # context manager ensures closure


def test_put_and_get_roundtrip(temp_db: ImageDatabase) -> None:
    payload = b"hello-world"
    temp_db.put("sample", payload)

    assert "sample" in temp_db
    assert len(temp_db) == 1
    assert temp_db.get("sample") == payload


def test_put_image_from_pil(temp_db: ImageDatabase) -> None:
    image = Image.new("RGB", (8, 4), color=(10, 20, 30))

    temp_db.put_image("pil", image, image_format="PNG")

    loaded = temp_db.get_image("pil", mode="RGB")
    assert loaded is not None
    assert loaded.size == (8, 4)
    assert loaded.getpixel((0, 0)) == (10, 20, 30)


def test_put_image_from_bytes(temp_db: ImageDatabase) -> None:
    image = Image.new("L", (2, 2), color=128)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    temp_db.put_image("bytes", buffer.getvalue(), overwrite=True)

    loaded = temp_db.get_image("bytes")
    assert loaded is not None
    assert loaded.mode == "L"


def test_delete_and_missing_key(temp_db: ImageDatabase) -> None:
    temp_db.put("to-delete", b"payload")
    temp_db.delete("to-delete")

    assert "to-delete" not in temp_db
    with pytest.raises(KeyError):
        temp_db.delete("to-delete")


def test_readonly_rejects_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "readonly.lmdb"
    db = ImageDatabase(db_path)
    db.put("x", b"1")
    db.close()

    readonly = ImageDatabase(db_path, readonly=True)
    with pytest.raises(RuntimeError):
        readonly.put("x", b"2")

    readonly.close()
