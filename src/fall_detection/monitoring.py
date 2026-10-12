"""Input and score drift checks, with a self-test so the monitor itself is monitored.

Features are cheap, interpretable per-window summaries of the pose landmarks. Drift is measured
against a committed reference (`monitoring/reference.json`, derived statistics only, no video).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

FEATURES = [
    "mean_visibility",
    "hip_height",
    "shoulder_height",
    "torso_length",
    "motion_energy",
    "mean_depth",
]
PSI_ALERT = 0.2  # common rule of thumb: > 0.2 is a significant shift
KS_ALPHA = 0.01


def window_features(X: np.ndarray) -> np.ndarray:
    """(n, 30, 132) windows -> (n, len(FEATURES)) summaries."""
    n, frames, _ = X.shape
    L = X.reshape(n, frames, 33, 4)
    hip = L[:, :, [23, 24], 1].mean(2)  # (n, frames), image y grows downwards
    shoulder = L[:, :, [11, 12], 1].mean(2)
    return np.column_stack(
        [
            L[..., 3].mean((1, 2)),
            hip.mean(1),
            shoulder.mean(1),
            np.abs(hip - shoulder).mean(1),
            np.abs(np.diff(L[..., :2], axis=1)).mean((1, 2, 3)),
            L[..., 2].mean((1, 2)),
        ]
    )


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def ks_pvalue(d: float, n: int, m: int) -> float:
    """Asymptotic two-sample Kolmogorov-Smirnov p-value."""
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-9:
        return 1.0
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(min(1.0, max(0.0, p)))


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins taken from the reference."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: compare the two means only
        return float(abs(reference.mean() - current.mean()) > 1e-6)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, edges)[0] / len(reference)
    cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p, cur_p = np.clip(ref_p, 1e-4, None), np.clip(cur_p, 1e-4, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def compare(reference: np.ndarray, current: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for j, name in enumerate(names):
        d = ks_statistic(reference[:, j], current[:, j])
        p = ks_pvalue(d, len(reference), len(current))
        s = psi(reference[:, j], current[:, j])
        rows.append(
            {
                "feature": name,
                "ks": round(d, 4),
                "ks_p": round(p, 6),
                "psi": round(s, 4),
                "drift": bool(p < KS_ALPHA and s > PSI_ALERT),
            }
        )
    return rows


def drift_report(
    ref_features: np.ndarray,
    cur_features: np.ndarray,
    ref_scores: np.ndarray,
    cur_scores: np.ndarray,
) -> dict:
    rows = compare(
        np.column_stack([ref_features, ref_scores]),
        np.column_stack([cur_features, cur_scores]),
        FEATURES + ["fall_score"],
    )
    return {
        "n_reference": len(ref_features),
        "n_current": len(cur_features),
        "positive_rate_reference": float((ref_scores > 0.5).mean()),
        "positive_rate_current": float((cur_scores > 0.5).mean()),
        "features": rows,
        "drift": any(r["drift"] for r in rows),
    }


def self_test(ref_features: np.ndarray, ref_scores: np.ndarray, seed: int = 0) -> dict:
    """The detector must stay quiet on a resample and fire on an injected shift."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref_features), len(ref_features))
    quiet = drift_report(ref_features, ref_features[idx], ref_scores, ref_scores[idx])

    shifted = ref_features.copy()
    shifted[:, FEATURES.index("hip_height")] += 0.15  # camera moved / person closer
    shifted[:, FEATURES.index("motion_energy")] *= 2.0
    loud = drift_report(ref_features, shifted, ref_scores, ref_scores)
    return {
        "no_false_alarm_on_resample": not quiet["drift"],
        "detects_injected_shift": loud["drift"],
        "passed": (not quiet["drift"]) and loud["drift"],
    }


def load_reference(path: str | os.PathLike) -> dict:
    ref = json.loads(Path(path).read_text())
    ref["features"] = np.asarray(ref["features"], dtype=np.float64)
    ref["scores"] = np.asarray(ref["scores"], dtype=np.float64)
    return ref


def log_windows(path: str | os.PathLike, windows: np.ndarray, scores: np.ndarray) -> None:
    """Append anonymous per-window summaries (never landmarks or video) as JSON lines."""
    feats = window_features(windows)
    with open(path, "a") as f:
        f.writelines(
            json.dumps({"features": [round(float(v), 5) for v in row], "score": round(float(s), 5)})
            + "\n"
            for row, s in zip(feats, scores, strict=True)
        )


def read_log(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return (
        np.asarray([r["features"] for r in rows], dtype=np.float64),
        np.asarray([r["score"] for r in rows], dtype=np.float64),
    )
