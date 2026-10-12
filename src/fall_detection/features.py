"""Camera-invariant features computed from MediaPipe landmarks.

Raw landmark coordinates depend on where the person stands in the frame and how far from the
camera. These features are expressed relative to the body (torso length, hip centre), so a fall
looks the same at 2 m or 4 m and at the left or the right of the image.
"""

from __future__ import annotations

import numpy as np

# Frames were resized to 640x480 before pose extraction: one unit of y is 3/4 of a unit of x.
ASPECT = 640 / 480

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

FRAME_FEATURES = [
    "torso_sin",  # torso angle from vertical (sin)
    "torso_cos",  # (cos): 1 = upright, 0 = horizontal, negative = head below hips
    "head_hip",  # (nose_y - hip_y) / torso length
    "knee_hip",  # (knee_y - hip_y) / torso length
    "ankle_hip",  # (ankle_y - hip_y) / torso length
    "bbox_aspect",  # visible bounding box width / height
    "bbox_height",  # bounding box height / torso length
    "hip_height",  # hip centre y in the image (0 top, 1 bottom)
]
DYNAMIC_FEATURES = ["hip_dx", "hip_dy", "bbox_dh"]  # per-frame change, in torso lengths
ALL_FEATURES = FRAME_FEATURES + DYNAMIC_FEATURES


def _mid(L: np.ndarray, a: int, b: int) -> np.ndarray:
    return (L[..., a, :] + L[..., b, :]) / 2


def window_features(X: np.ndarray, include_absolute_height: bool = True) -> np.ndarray:
    """(n, T, 132) raw windows -> (n, T, F) invariant features."""
    n, T, _ = X.shape
    L = X.reshape(n, T, 33, 4).astype(np.float64).copy()
    L[..., 0] *= ASPECT  # square pixel units
    xy = L[..., :2]
    vis = L[..., 3]

    shoulder, hip = _mid(xy, L_SHOULDER, R_SHOULDER), _mid(xy, L_HIP, R_HIP)
    torso = shoulder - hip  # image y grows downward: upright torso has negative y
    length = np.linalg.norm(torso, axis=-1).clip(1e-3)
    sin = torso[..., 0] / length
    cos = -torso[..., 1] / length

    head_hip = (xy[..., NOSE, 1] - hip[..., 1]) / length
    knee_hip = (_mid(xy, L_KNEE, R_KNEE)[..., 1] - hip[..., 1]) / length
    ankle_hip = (_mid(xy, L_ANKLE, R_ANKLE)[..., 1] - hip[..., 1]) / length

    visible = (vis > 0.5)[..., None]
    big = 1e6
    lo = np.where(visible, xy, big).min(axis=2)
    hi = np.where(visible, xy, -big).max(axis=2)
    width = (hi[..., 0] - lo[..., 0]).clip(1e-3)
    height = (hi[..., 1] - lo[..., 1]).clip(1e-3)

    frame = np.stack(
        [sin, cos, head_hip, knee_hip, ankle_hip, width / height, height / length, hip[..., 1]],
        axis=-1,
    )
    scale = length.mean(axis=1, keepdims=True)  # one body scale per window
    d_hip = np.diff(hip, axis=1, prepend=hip[:, :1]) / scale[..., None]
    d_height = np.diff(height, axis=1, prepend=height[:, :1]) / scale
    out = np.concatenate([frame, d_hip, d_height[..., None]], axis=-1)
    if not include_absolute_height:
        out = np.delete(out, ALL_FEATURES.index("hip_height"), axis=-1)
    return np.nan_to_num(out).astype(np.float32)


def centered_scaled(X: np.ndarray) -> np.ndarray:
    """Raw (x, y) of the 33 landmarks, centred on the hip centre and divided by torso length."""
    n, T, _ = X.shape
    L = X.reshape(n, T, 33, 4).astype(np.float64)
    xy = L[..., :2].copy()
    xy[..., 0] *= ASPECT
    hip = _mid(xy, L_HIP, R_HIP)
    length = np.linalg.norm(_mid(xy, L_SHOULDER, R_SHOULDER) - hip, axis=-1).clip(1e-3)
    rel = (xy - hip[:, :, None, :]) / length[..., None, None]
    return rel.reshape(n, T, 66).astype(np.float32)
