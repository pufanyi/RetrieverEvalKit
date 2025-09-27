from __future__ import annotations

import io
import os
from collections.abc import Iterator
from pathlib import Path

import lmdb
from PIL import Image


class ImageDatabase:
    """Minimal LMDB-backed image cache."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        map_size: int = 1 << 30,
        readonly: bool = False,
    ):
        self.path = Path(path)
        self.map_size = map_size
        self.readonly = readonly
        self._env: lmdb.Environment | None = None

    def _ensure_env(self) -> lmdb.Environment:
        if self._env is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._env = lmdb.open(
                str(self.path),
                create=not self.readonly,
                readonly=self.readonly,
                map_size=self.map_size,
                subdir=True,
                lock=not self.readonly,
                readahead=self.readonly,
            )
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self) -> ImageDatabase:
        self._ensure_env()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _encode_key(self, key: str | bytes) -> bytes:
        if isinstance(key, bytes):
            return key
        return key.encode("utf-8")

    def _grow_map(self, extra: int) -> None:
        env = self._ensure_env()
        info = env.info()
        target = max(info["map_size"] * 2, info["map_size"] + extra + (64 << 20))
        env.set_mapsize(target)

    def put(self, key: str | bytes, value: bytes, *, overwrite: bool = True) -> None:
        if self.readonly:
            raise RuntimeError("Cannot write to a read-only database")
        encoded_key = self._encode_key(key)
        env = self._ensure_env()
        try:
            with env.begin(write=True) as txn:
                if not txn.put(encoded_key, value, overwrite=overwrite):
                    raise KeyError(f"Key {key!r} already exists")
        except lmdb.MapFullError:
            self._grow_map(len(value))
            self.put(key, value, overwrite=overwrite)

    def put_image(
        self,
        key: str | bytes,
        image: Image.Image | bytes | str | os.PathLike[str],
        *,
        image_format: str = "PNG",
        overwrite: bool = True,
    ) -> None:
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format=image_format)
            data = buffer.getvalue()
        elif isinstance(image, str | os.PathLike):
            data = Path(image).read_bytes()
        elif isinstance(image, bytes):
            data = image
        else:
            raise TypeError("Unsupported image type")
        self.put(key, data, overwrite=overwrite)

    def get(self, key: str | bytes) -> bytes | None:
        env = self._ensure_env()
        with env.begin(write=False) as txn:
            value = txn.get(self._encode_key(key))
            if value is None:
                return None
            return bytes(value)

    def get_image(
        self, key: str | bytes, *, mode: str | None = None
    ) -> Image.Image | None:
        data = self.get(key)
        if data is None:
            return None
        image = Image.open(io.BytesIO(data))
        if mode is not None:
            image = image.convert(mode)
        return image

    def delete(self, key: str | bytes) -> None:
        if self.readonly:
            raise RuntimeError("Cannot write to a read-only database")
        env = self._ensure_env()
        with env.begin(write=True) as txn:
            if not txn.delete(self._encode_key(key)):
                raise KeyError(f"Key {key!r} not found")

    def __contains__(self, key: str | bytes) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        env = self._ensure_env()
        return env.stat()["entries"]

    def keys(self) -> Iterator[str]:
        env = self._ensure_env()
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            try:
                for key, _ in cursor:
                    yield key.decode("utf-8")
            finally:
                cursor.close()

    def items(self) -> Iterator[tuple[str, bytes]]:
        env = self._ensure_env()
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            try:
                for key, value in cursor:
                    yield key.decode("utf-8"), bytes(value)
            finally:
                cursor.close()
