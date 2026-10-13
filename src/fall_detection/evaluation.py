"""Out-of-video evaluation of the fall classifier.

The dataset (``X.npy``) is a stack of non-overlapping 30-frame windows cut from many videos, in
folder order. A plain random split puts windows of the *same* video in both train and test, which
inflates the scores. Here windows are first grouped by video, then split by group.

``X.npy`` stores no video id, so groups are rebuilt from temporal continuity: the last frame of
window i and the first frame of window i+1 are consecutive frames of one video, unless the camera
view changes (a new video starts) or the label changes.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

from fall_detection.architectures import DEFAULT_ARCH, build_model

N_LANDMARKS = 33


def _xy(frame: np.ndarray) -> np.ndarray:
    """(x, y) of the 33 landmarks of one frame stored as 132 floats (x, y, z, visibility)."""
    return frame.reshape(N_LANDMARKS, 4)[:, :2]


def reconstruct_groups(X: np.ndarray, y: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Assign a pseudo video id to every window.

    A new group starts when the label changes, or when the pose jumps between the end of window i
    and the start of window i+1 by more than the ``quantile`` of ordinary frame-to-frame steps.
    """
    steps = np.array(
        [np.abs(_xy(w[k]) - _xy(w[k + 1])).mean() for w in X for k in range(w.shape[0] - 1)]
    )
    threshold = np.quantile(steps, quantile)
    groups = np.zeros(len(X), dtype=int)
    for i in range(1, len(X)):
        jump = np.abs(_xy(X[i - 1][-1]) - _xy(X[i][0])).mean()
        new_video = jump > threshold or y[i] != y[i - 1]
        groups[i] = groups[i - 1] + int(new_video)
    return groups


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion (reliable for small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = tp + fp + fn + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": n,
        "accuracy": (tp + tn) / n,
        "accuracy_ci95": wilson_interval(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "recall_ci95": wilson_interval(tp, tp + fn),
        "f1": f1,
    }


def train_one(
    Xtr,
    ytr,
    Xva,
    yva,
    seed: int,
    max_epochs: int = 150,
    patience: int = 50,
    channels: int = 132,
    arch: str = DEFAULT_ARCH,
):
    """Same recipe as scripts/train.py: Adam 1e-3, BCE, batch 32, early stopping on val loss."""
    torch.manual_seed(seed)
    model = build_model(arch, input_channels=channels, seq_length=Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    xt = torch.tensor(Xtr).permute(0, 2, 1)
    yt = torch.tensor(ytr).unsqueeze(1)
    xv = torch.tensor(Xva).permute(0, 2, 1)
    yv = torch.tensor(yva).unsqueeze(1)
    best, best_state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv).item()
        if val < best:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(X).permute(0, 2, 1)).squeeze(1).numpy()


def latency_ms(model: nn.Module, X: np.ndarray, repeats: int = 200) -> dict:
    """Model-only latency for one window (batch of 1, CPU). Pose extraction is not included."""
    x = torch.tensor(X[:1]).permute(0, 2, 1)
    times = []
    with torch.no_grad():
        for _ in range(20):  # warm-up
            model(x)
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def cross_validate(X, y, groups, n_splits: int = 5, seed: int = 42) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        # hold out a group-wise slice of the training part for early stopping
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        fit_idx, val_idx = next(inner.split(X[tr], y[tr], groups[tr]))
        model = train_one(
            X[tr][fit_idx], y[tr][fit_idx], X[tr][val_idx], y[tr][val_idx], seed + fold
        )
        oof[te] = predict(model, X[te])
    return {"oof_probability": oof, "metrics": confusion(y, (oof > 0.5).astype(int))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="docs/evaluation.json")
    args = parser.parse_args()

    X = np.load(f"{args.data}/X.npy").astype(np.float32)
    y = np.load(f"{args.data}/y.npy").astype(np.float32)
    groups = reconstruct_groups(X, y)
    result = cross_validate(X, y, groups, n_splits=args.folds)
    report = {
        "windows": len(y),
        "falls": int(y.sum()),
        "no_falls": int((1 - y).sum()),
        "pseudo_videos": int(groups.max() + 1),
        "folds": args.folds,
        "out_of_video_metrics": result["metrics"],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
