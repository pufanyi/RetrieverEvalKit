from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .dataset import TextDataset


@dataclass(frozen=True)
class CaptionRecord:
    index: int
    caption_id: str
    caption: str
    image: str
    image_id: str


class CaptionsJsonlDataset(TextDataset):
    def __init__(self, path: str | Path, name: str = "Flickr30kCaptions"):
        super().__init__(name)
        self.path = Path(path)
        self._records: list[CaptionRecord] | None = None

    def build(self):
        if self._records is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Caption file not found: {self.path}")

        records: list[CaptionRecord] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                image_name = payload.get("image", "")
                image_stem = Path(image_name).stem or "unknown"
                captions = payload.get("caption", [])
                if isinstance(captions, str):
                    captions = [captions]

                for local_idx, caption_text in enumerate(captions):
                    caption_id = f"{image_stem}#{local_idx}"
                    records.append(
                        CaptionRecord(
                            index=len(records),
                            caption_id=caption_id,
                            caption=str(caption_text),
                            image=image_name,
                            image_id=image_stem,
                        )
                    )

        self._records = records

    @property
    def records(self) -> list[CaptionRecord]:
        if self._records is None:
            self.build()
        assert self._records is not None
        return self._records

    def length(self) -> int:
        return len(self.records)

    def get_texts(self, batch_size: int = 1) -> Iterator[list[tuple[int, str]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        batch: list[tuple[int, str]] = []
        for record in self.records:
            batch.append((record.index, record.caption))
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def get_record(self, index: int) -> CaptionRecord:
        return self.records[index]
