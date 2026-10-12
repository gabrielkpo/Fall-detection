import numpy as np

from fall_detection import monitoring


def _windows(n=80, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 30, 132)).astype(np.float32)


def test_window_features_shape():
    f = monitoring.window_features(_windows())
    assert f.shape == (80, len(monitoring.FEATURES))


def test_ks_and_psi_zero_for_identical_samples():
    a = np.random.default_rng(1).normal(size=500)
    assert monitoring.ks_statistic(a, a) == 0.0
    assert monitoring.psi(a, a) < 1e-6
    assert monitoring.ks_pvalue(0.0, 500, 500) == 1.0


def test_detects_a_mean_shift():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 400), rng.normal(1.5, 1, 400)
    row = monitoring.compare(a[:, None], b[:, None], ["x"])[0]
    assert row["drift"] and row["psi"] > monitoring.PSI_ALERT


def test_self_test_passes_on_real_reference():
    ref = monitoring.load_reference("models/monitoring_reference.json")
    assert monitoring.self_test(ref["features"], ref["scores"])["passed"]


def test_log_roundtrip(tmp_path):
    path = tmp_path / "log.jsonl"
    monitoring.log_windows(path, _windows(5), np.linspace(0, 1, 5))
    feats, scores = monitoring.read_log(path)
    assert feats.shape == (5, len(monitoring.FEATURES)) and len(scores) == 5
