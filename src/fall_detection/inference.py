"""Model loading and windowed inference, shared by the API, the demo and the monitoring job."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fall_detection.architectures import DEFAULT_ARCH, build_model

WINDOW = 30  # frames per window, must match training (scripts/extract_pose_dataset.py)
N_FEATURES = 132  # 33 landmarks x (x, y, z, visibility)
DEFAULT_MODEL_PATH = "models/champion.pth"
DEFAULT_META_PATH = "models/champion.json"


@dataclass(frozen=True)
class Prediction:
    fall: bool
    score: float  # highest window probability
    window_scores: list[float]


def load_model(path: str | os.PathLike | None = None, arch: str | None = None) -> torch.nn.Module:
    """Load weights; the architecture comes from `arch`, else from the champion metadata."""
    path = path or os.environ.get("FALL_MODEL_PATH", DEFAULT_MODEL_PATH)
    arch = arch or load_metadata().get("arch", DEFAULT_ARCH)
    model = build_model(arch, input_channels=N_FEATURES, seq_length=WINDOW)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def load_metadata(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("FALL_META_PATH", DEFAULT_META_PATH))
    return json.loads(path.read_text()) if path.exists() else {}


def sliding_windows(frames: np.ndarray, size: int = WINDOW, stride: int = 15) -> np.ndarray:
    """Cut (n_frames, 132) into overlapping (n_windows, size, 132) windows."""
    if frames.ndim != 2 or frames.shape[1] != N_FEATURES:
        raise ValueError(f"expected shape (n_frames, {N_FEATURES}), got {frames.shape}")
    if len(frames) < size:
        raise ValueError(f"need at least {size} frames, got {len(frames)}")
    starts = range(0, len(frames) - size + 1, stride)
    return np.stack([frames[s : s + size] for s in starts]).astype(np.float32)


def score_windows(model: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    """Fall probability of each (n, 30, 132) window."""
    with torch.no_grad():
        x = torch.from_numpy(windows).permute(0, 2, 1)  # (n, 132, 30), as in training
        return model(x).squeeze(1).numpy()


def predict(
    model: torch.nn.Module, frames: np.ndarray, threshold: float = 0.5, stride: int = 15
) -> Prediction:
    """A clip is a fall if any of its windows exceeds the threshold."""
    scores = score_windows(model, sliding_windows(frames, stride=stride))
    return Prediction(
        fall=bool(scores.max() > threshold),
        score=float(scores.max()),
        window_scores=[round(float(s), 4) for s in scores],
    )
