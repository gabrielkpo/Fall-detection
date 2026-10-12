"""Video -> MediaPipe landmarks. Heavy dependencies are imported lazily."""

from __future__ import annotations

import numpy as np

IMG_SIZE = (640, 480)  # same resize as the training data extraction


def extract_landmarks(video_path: str, max_frames: int = 150) -> np.ndarray:
    """Return an (n_frames, 132) array. Frames without a detected person are skipped,
    exactly as in the training data extraction."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    rows: list[list[float]] = []
    read = 0
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        while read < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            frame = cv2.resize(frame, IMG_SIZE)
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                rows.append(
                    [
                        v
                        for lm in result.pose_landmarks.landmark
                        for v in (lm.x, lm.y, lm.z, lm.visibility)
                    ]
                )
    cap.release()
    return np.asarray(rows, dtype=np.float32).reshape(-1, 132)
