"""Build the original training set from videos stored as frame folders.

Layout (labels come from the top-level folder names):

    fall_dataset/<sequence>/frame_0001.png ...   label 1
    adl_dataset/<sequence>/frame_0001.png ...    label 0 (activities of daily living)

Each frame is resized to 640x480, MediaPipe Pose extracts 33 landmarks x (x, y, z, visibility)
= 132 values, frames without a detected person are skipped, and every 30 consecutive
detections become one non-overlapping window. Writes X.npy (n, 30, 132) and y.npy (n,).

No video id is saved, which is why evaluation rebuilds groups heuristically; new data should
go through scripts/ingest_clips.py instead, which keeps it.

    python scripts/extract_pose_dataset.py --root . --out data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from fall_detection.inference import N_FEATURES, WINDOW
from fall_detection.pose import IMG_SIZE

LABELS = {"fall_dataset": 1, "adl_dataset": 0}


def landmarks(pose, frame: np.ndarray) -> list[float] | None:
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None
    return [v for lm in result.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="folder holding the datasets")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    X: list[list[list[float]]] = []
    y: list[int] = []
    # static_image_mode: each frame is detected on its own, as in the original extraction
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        for folder, label in LABELS.items():
            sequences = sorted(p for p in (args.root / folder).iterdir() if p.is_dir())
            for seq in tqdm(sequences, desc=folder):
                window: list[list[float]] = []
                for frame_path in sorted(seq.iterdir()):
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        continue  # not an image
                    row = landmarks(pose, cv2.resize(frame, IMG_SIZE))
                    if row is not None:
                        window.append(row)
                    if len(window) == WINDOW:
                        X.append(window)
                        y.append(label)
                        window = []

    args.out.mkdir(parents=True, exist_ok=True)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, WINDOW, N_FEATURES)
    np.save(args.out / "X.npy", X_arr)
    np.save(args.out / "y.npy", np.asarray(y, dtype=np.int64))
    print(f"{len(X_arr)} windows, shape {X_arr.shape} → {args.out}")


if __name__ == "__main__":
    main()
