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
