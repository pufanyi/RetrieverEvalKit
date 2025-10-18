from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from img_search.data.inquire import InquireDataset


@pytest.fixture
def mock_load_dataset():
    """Fixture to patch the load_dataset function."""
    with patch("img_search.data.inquire.load_dataset") as mock:
        # Create a dummy dataset
        dummy_data = [
            {
                "image": Image.new("RGB", (10, 10), color="red"),
                "inat24_file_name": "gallery/red.jpg",
                "inat24_image_id": 1,
            },
            {
                "image": Image.new("RGB", (10, 10), color="green"),
                "inat24_file_name": "gallery/green.jpg",
                "inat24_image_id": 2,
            },
            {
                "image": Image.new("RGB", (10, 10), color="blue"),
                "inat24_file_name": "gallery/blue.jpg",
                "inat24_image_id": 3,
            },
        ]
        mock.return_value = dummy_data
        yield mock


def test_inquire_dataset_build(mock_load_dataset: MagicMock):
    """Test that build calls load_dataset with the correct path and split."""
    dataset = InquireDataset(path="test/path", split="validation")
    dataset.build()

    mock_load_dataset.assert_called_once_with("test/path", split="validation")
    assert dataset._dataset is not None


def test_inquire_dataset_lazy_build(mock_load_dataset: MagicMock):
    """Test that the dataset is loaded automatically on first access."""
    dataset = InquireDataset()
    assert dataset._dataset is None  # Not loaded yet

    # Accessing the .dataset property should trigger build()
    _ = dataset.dataset
    mock_load_dataset.assert_called_once()
    assert dataset._dataset is not None


def test_inquire_dataset_length(mock_load_dataset: MagicMock):
    """Test that length returns the correct number of items."""
    dataset = InquireDataset()
    assert len(dataset) == 3  # Based on the dummy_data in the fixture


def test_inquire_get_images_batching(mock_load_dataset: MagicMock):
    """Test the get_images method with different batch sizes."""
    dataset = InquireDataset()

    # Test with batch_size = 2
    batches = list(dataset.get_images(batch_size=2))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1  # Remainder
    ids_batch_one = [item[0] for item in batches[0]]
    assert ids_batch_one == ["gallery/red.jpg", "gallery/green.jpg"]

    # Test with batch_size = 3 (exact match)
    batches = list(dataset.get_images(batch_size=3))
    assert len(batches) == 1
    assert len(batches[0]) == 3

    # Verify image types
    first_image = batches[0][0][1]
    assert isinstance(first_image, Image.Image)


def test_inquire_identifier_fallback(mock_load_dataset: MagicMock):
    """Absent identifier columns should fall back to the row index."""
    fallback_rows = [
        {"image": Image.new("RGB", (8, 8), color="cyan")},
        {"image": Image.new("RGB", (8, 8), color="magenta")},
    ]
    mock_load_dataset.return_value = fallback_rows
    dataset = InquireDataset(id_column="missing_column")
    batches = list(dataset.get_images(batch_size=1))
    ids = [item[0] for batch in batches for item in batch]
    assert ids == ["0", "1"]


def test_inquire_invalid_batch_size(mock_load_dataset: MagicMock):
    """Batch size must be positive."""
    dataset = InquireDataset()
    with pytest.raises(ValueError):
        list(dataset.get_images(batch_size=0))
