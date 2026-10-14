"""Real-time fall detection on the default webcam, with the promoted champion model.

Run from the repository root after `pip install -e ".[train]"`:
    python scripts/run_webcam.py            # press q to quit

Nothing is saved to disk. A fall is reported after `--consecutive` windows above the
threshold, with a terminal bell. Research prototype, not a safety system.
"""

from __future__ import annotations

import argparse
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from fall_detection import inference
from fall_detection.pose import IMG_SIZE


def main() -> None:
    meta = inference.load_metadata()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--threshold", type=float, default=meta.get("threshold", 0.5))
    parser.add_argument("--consecutive", type=int, default=3, help="windows above threshold")
    args = parser.parse_args()

    model = inference.load_model()
    buffer: deque[list[float]] = deque(maxlen=inference.WINDOW)
    above = 0
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")

    with mp.solutions.pose.Pose(min_detection_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, IMG_SIZE)  # same resize as the training data
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                buffer.append(
                    [
                        v
                        for lm in result.pose_landmarks.landmark
                        for v in (lm.x, lm.y, lm.z, lm.visibility)
                    ]
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )

            if len(buffer) == inference.WINDOW:
                window = np.asarray(buffer, dtype=np.float32)[None]
                score = float(inference.score_windows(model, window)[0])
                above = above + 1 if score > args.threshold else 0
                fall = above >= args.consecutive
                if above == args.consecutive:
                    print(f"\aFall detected (score {score:.2f})", flush=True)
                label, color = ("FALL", (0, 0, 255)) if fall else ("ok", (0, 200, 0))
                cv2.putText(
                    frame, f"{label} {score:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
                )

            cv2.imshow("Fall detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
