
from unittest.mock import MagicMock, call

import pytest
from omegaconf import OmegaConf

from img_search.pipeline import embed


@pytest.fixture
def mock_get_encoder(monkeypatch):
    mock = MagicMock(name="get_encoder")
    monkeypatch.setattr(embed, "get_encoder", mock)
    return mock


@pytest.fixture
def mock_get_dataset(monkeypatch):
    mock = MagicMock(name="get_dataset")
    monkeypatch.setattr(embed, "get_dataset", mock)
    return mock


def test_get_models_and_datasets(mock_get_encoder, mock_get_dataset):
    """Verify that models and datasets are created from the config."""
    cfg = OmegaConf.create({
        "models": [{"name": "model1"}, {"name": "model2"}],
        "datasets": [{"name": "dataset1"}, {"name": "dataset2"}],
    })

    models, datasets = embed.get_models_and_datasets(cfg)

    assert mock_get_encoder.call_count == 2
    mock_get_encoder.assert_has_calls([
        call(cfg.models[0]),
        call(cfg.models[1]),
    ])
    assert models == [mock_get_encoder.return_value, mock_get_encoder.return_value]

    assert mock_get_dataset.call_count == 2
    mock_get_dataset.assert_has_calls([
        call(cfg.datasets[0]),
        call(cfg.datasets[1]),
    ])
    assert datasets == [mock_get_dataset.return_value, mock_get_dataset.return_value]


def test_embed_all_yields_correct_data():
    """Test that the embed_all generator yields the correct items."""
    # Mock models
    mock_embedding1 = MagicMock()
    mock_embedding1.shape = (1, 128)
    model1 = MagicMock()
    model1.name = "model1"
    model1.encode.return_value = mock_embedding1
    
    mock_embedding2 = MagicMock()
    mock_embedding2.shape = (1, 128)
    model2 = MagicMock()
    model2.name = "model2"
    model2.encode.return_value = mock_embedding2
    models = [model1, model2]

    # Mock datasets
    dataset1 = MagicMock()
    dataset1.name = "dataset1"
    dataset1.get_images.return_value = [["image_a"], ["image_b"]]
    dataset1.length.return_value = 2
    dataset2 = MagicMock()
    dataset2.name = "dataset2"
    dataset2.get_images.return_value = [["image_c"]]
    dataset2.length.return_value = 1
    datasets = [dataset1, dataset2]

    tasks_config = OmegaConf.create({"batch_size": 1})

    generator = embed.embed_all(models, datasets, tasks_config=tasks_config)
    results = list(generator)

    # Check build calls
    model1.build.assert_called_once()
    model2.build.assert_called_once()
    assert dataset1.build.call_count == 2
    assert dataset2.build.call_count == 2

    # Check that get_images was called correctly
    assert dataset1.get_images.call_count == 2
    assert dataset2.get_images.call_count == 2

    # Check encode calls
    assert model1.encode.call_count == 3 # image_a, image_b, image_c
    assert model2.encode.call_count == 3 # image_a, image_b, image_c

    # Check yielded results
    expected_results = [
        ("model1", "dataset1", mock_embedding1, (1, 128)),
        ("model1", "dataset1", mock_embedding1, (1, 128)),
        ("model1", "dataset2", mock_embedding1, (1, 128)),
        ("model2", "dataset1", mock_embedding2, (1, 128)),
        ("model2", "dataset1", mock_embedding2, (1, 128)),
        ("model2", "dataset2", mock_embedding2, (1, 128)),
    ]

    assert len(results) == len(expected_results)
    for i, (m, d, e, s) in enumerate(results):
        expected_m, expected_d, expected_e, expected_s = expected_results[i]
        assert m == expected_m
        assert d == expected_d
        assert e == expected_e
        assert s == expected_s
