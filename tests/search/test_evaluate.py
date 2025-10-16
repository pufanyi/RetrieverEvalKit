from img_search.search.evaluate import _expand_method_configs


def test_expand_method_configs_includes_gpu_when_available():
    methods = [{"method": "flat"}]
    configs = _expand_method_configs(methods, None, gpu_available=True)

    assert configs == [
        {"method": "flat", "use_gpu": False},
        {"method": "flat", "use_gpu": True},
    ]


def test_expand_method_configs_skips_gpu_when_unavailable():
    methods = [{"method": "flat"}, {"method": "ivf_flat"}]
    configs = _expand_method_configs(methods, None, gpu_available=False)

    assert configs == [
        {"method": "flat", "use_gpu": False},
        {"method": "ivf_flat", "use_gpu": False},
    ]


def test_expand_method_configs_forces_cpu_when_gpu_requested_but_missing():
    methods = [{"method": "flat"}]
    configs = _expand_method_configs(methods, True, gpu_available=False)

    assert configs == [{"method": "flat", "use_gpu": False}]
