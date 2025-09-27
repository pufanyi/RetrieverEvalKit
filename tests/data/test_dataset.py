
from collections.abc import Iterator

import pytest
from PIL import Image

from img_search.data.dataset import ImageDataset


class MockImageDataset(ImageDataset):
    def __init__(self, name: str, items: list[str | Image.Image]):
        super().__init__(name)
        self.items = items
        self.built = False

    def build(self):
        self.built = True

    def length(self) -> int:
        return len(self.items)

    def get_images(self, batch_size: int = 1) -> Iterator[list[str | Image.Image]]:
        for i in range(0, len(self.items), batch_size):
            yield self.items[i : i + batch_size]


def test_image_dataset_iterator():
    """Verify that iterating over a dataset yields single items."""
    items = ["image1", "image2", "image3"]
    dataset = MockImageDataset(name="mock", items=items)

    # The __iter__ method should yield one item at a time
    result = list(dataset)

    assert result == items


def test_image_dataset_len():
    """Test that len(dataset) calls the length() method."""
    items = ["image1", "image2"]
    dataset = MockImageDataset(name="mock", items=items)

    assert len(dataset) == 2

def test_iter_raises_error_on_incorrect_batch_size():
    """Test that the iterator raises a ValueError if get_images returns a batch > 1."""
    class BadDataset(MockImageDataset):
        def get_images(self, batch_size: int = 1) -> Iterator[list[str | Image.Image]]:
            # This implementation is wrong because it ignores batch_size=1 from __iter__
            yield self.items # Yields the whole list as one batch

    dataset = BadDataset(name="bad", items=["a", "b"])
    with pytest.raises(ValueError, match="Batch size is not 1, but 2"):
        list(dataset) # Trigger the iterator
