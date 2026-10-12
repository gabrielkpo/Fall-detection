import numpy as np

from fall_detection.evaluation import confusion, reconstruct_groups, wilson_interval


def _video(n_windows: int, start: float) -> np.ndarray:
    """Slowly drifting pose, n_windows * 30 frames, as (n_windows, 30, 132)."""
    frames = (start + 0.001 * np.arange(n_windows * 30))[:, None] + np.zeros((1, 132))
    return frames.reshape(n_windows, 30, 132).astype(np.float32)


def test_reconstruct_groups_splits_on_pose_jump_and_label_change():
    X = np.concatenate([_video(3, 0.1), _video(2, 0.9), _video(2, 0.5)])
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    groups = reconstruct_groups(X, y)
    assert groups.tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_confusion_metrics():
    m = confusion(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_wilson_interval_contains_estimate():
    lo, hi = wilson_interval(10, 10)
    assert lo < 1.0 and hi == 1.0
