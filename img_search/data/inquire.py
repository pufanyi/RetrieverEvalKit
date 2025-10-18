from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from datasets import load_dataset
from PIL import Image

from .dataset import ImageDataset


class InquireDataset(ImageDataset):
    """Iterates over INQUIRE images published via Hugging Face.

    The upstream dataset (`evendrow/INQUIRE-Rerank`) exposes multiple metadata
    columns, including a stable file name for every image.  Downstream pipelines
    expect a deterministic identifier for each sample so we default to the
    ``inat24_file_name`` column and fall back to the dataset index when the
    column is unavailable (e.g. during tests with synthetic rows).
    """

    def __init__(
        self,
        path: str = "evendrow/INQUIRE-Rerank",
        *,
        split: str = "test",
        id_column: str = "inat24_file_name",
        image_column: str = "image",
        fallback_to_index: bool = True,
    ):
        super().__init__("INQUIRE")
        self.dataset_path = path
        self.split = split
        self.id_column = id_column
        self.image_column = image_column
        self.fallback_to_index = fallback_to_index
        self._dataset: Any | None = None

    def build(self):
        self._dataset = load_dataset(self.dataset_path, split=self.split)

    @property
    def dataset(self):
        if not self._dataset:
            self.build()
        return self._dataset

    def length(self) -> int:
        return len(self.dataset)

    def _resolve_identifier(self, index: int, sample: Mapping[str, Any]) -> str:
        """Resolve a stable identifier for a dataset row."""
        if self.id_column and self.id_column in sample:
            return str(sample[self.id_column])
        if "inat24_file_name" in sample:
            return str(sample["inat24_file_name"])
        if "inat24_image_id" in sample:
            return str(sample["inat24_image_id"])
        if self.fallback_to_index:
            return str(index)
        available = ", ".join(sample.keys())
        raise KeyError(
            f"Identifier column '{self.id_column}' not found in INQUIRE sample. "
            f"Available columns: {available}"
        )

    def get_images(
        self, batch_size: int = 1
    ) -> Iterator[list[tuple[str, Image.Image | str]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        current_list = []
        for index, sample in enumerate(self.dataset):
            identifier = self._resolve_identifier(index, sample)
            try:
                image = sample[self.image_column]
            except KeyError as exc:
                available = ", ".join(sample.keys())
                raise KeyError(
                    f"Image column '{self.image_column}' not found in INQUIRE sample. "
                    f"Available columns: {available}"
                ) from exc
            current_list.append((identifier, image))
            if len(current_list) == batch_size:
                yield current_list
                current_list = []
        if current_list:
            yield current_list
