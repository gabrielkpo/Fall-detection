import numpy as np

from fall_detection.promotion import paired_bootstrap_f1


def _case(seed=0, n_groups=30, per_group=4):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    y = np.repeat(rng.integers(0, 2, n_groups), per_group).astype(float)
    return y, groups


def test_identical_models_are_not_promoted():
    y, groups = _case()
    p = np.where(y == 1, 0.9, 0.1)
    out = paired_bootstrap_f1(y, p, p, groups, n_boot=200)
    assert out["ci95"][0] <= 0 <= out["ci95"][1]


def test_clearly_better_candidate_is_promoted():
    y, groups = _case()
    rng = np.random.default_rng(3)
    champion = np.where(rng.random(len(y)) < 0.5, y, 1 - y)  # coin-flip quality
    candidate = y.copy()  # perfect
    out = paired_bootstrap_f1(y, champion, candidate, groups, n_boot=300)
    assert out["ci95"][0] > 0


def _rules_input(delta_lo=0.05, recall_drop=0.0, diff=1e-7, latency=0.1, params=30_000):
    champion = {"recall": 0.8}
    candidate = {"recall": 0.8 - recall_drop}
    boot = {"ci95": [delta_lo, delta_lo + 0.1]}
    checks = {"max_abs_diff": diff, "latency_p95_ms": latency, "parameters": params}
    return champion, candidate, boot, checks


def test_decide_promotes_when_every_rule_passes():
    from fall_detection.promotion import decide

    promote, rules = decide(*_rules_input())
    assert promote and all(r["passed"] for r in rules)


def test_decide_blocks_each_failure():
    from fall_detection.promotion import decide

    for kwargs in (
        {"delta_lo": -0.01},  # gain not proven
        {"recall_drop": 0.05},  # misses more falls
        {"diff": 1e-2},  # ONNX not faithful
        {"latency": 50.0},  # too slow for the browser
        {"params": 10_000_000},  # too big
    ):
        promote, _ = decide(*_rules_input(**kwargs))
        assert not promote, kwargs


def test_decide_fails_closed_without_onnx_checks():
    from fall_detection.promotion import decide

    champion, candidate, boot, _ = _rules_input()
    promote, _ = decide(champion, candidate, boot, {"parameters": 30_000})
    assert not promote
