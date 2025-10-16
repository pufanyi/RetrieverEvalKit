import pytest

from img_search.search import evaluate
from img_search.search.evaluate import _expand_method_configs


def test_expand_method_configs_includes_gpu_when_available():
    methods = [{"method": "flat"}]
    configs = _expand_method_configs(methods, None, gpu_available=True)

    assert configs == [
        {"backend": "faiss", "method": "flat", "use_gpu": False},
        {"backend": "faiss", "method": "flat", "use_gpu": True},
    ]


def test_expand_method_configs_skips_gpu_when_unavailable():
    methods = [{"method": "flat"}, {"method": "ivf_flat"}]
    configs = _expand_method_configs(methods, None, gpu_available=False)

    assert configs == [
        {"backend": "faiss", "method": "flat", "use_gpu": False},
        {"backend": "faiss", "method": "ivf_flat", "use_gpu": False},
    ]


def test_expand_method_configs_forces_cpu_when_gpu_requested_but_missing():
    methods = [{"method": "flat"}]
    configs = _expand_method_configs(methods, True, gpu_available=False)

    assert configs == [{"backend": "faiss", "method": "flat", "use_gpu": False}]


def test_expand_method_configs_preserves_non_faiss_backend() -> None:
    methods = [{"backend": "scann", "method": "scann"}]
    configs = _expand_method_configs(methods, None, gpu_available=True)

    assert configs == [{"backend": "scann", "method": "scann", "use_gpu": False}]


def test_filter_unavailable_backends_drops_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    methods = [
        {"backend": "faiss", "method": "flat"},
        {"backend": "scann", "method": "scann"},
        {"backend": "hnswlib", "method": "hnsw"},
    ]

    monkeypatch.setattr(evaluate, "scann_available", lambda: False)
    monkeypatch.setattr(evaluate, "hnswlib_available", lambda: False)

    filtered = evaluate._filter_unavailable_backends(methods)

    assert filtered == [{"backend": "faiss", "method": "flat"}]
