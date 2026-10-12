"""Turn labelled video clips into training windows, keeping the video id of every window.

Folder layout (labels come from the folder names):

    clips/
      fall/    fall_01.mp4 ...
      adl/     stand_01.mp4, sit_01.mp4, bend_01.mp4 ...   (activities of daily living)

    python scripts/ingest_clips.py clips --out data/own   # writes X.npy, y.npy, groups.npy, index.json

Unlike the original dataset, a `groups.npy` is saved, so evaluation can split by video exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fall_detection.inference import WINDOW, sliding_windows
from fall_detection.pose import extract_landmarks

LABELS = {"fall": 1, "adl": 0}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", help="folder with fall/ and adl/ subfolders")
    parser.add_argument("--out", default="data/own")
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=900)
    args = parser.parse_args()

    X, y, groups, index = [], [], [], []
    videos = sorted(
        p for label in LABELS for p in (Path(args.clips) / label).glob("*")
        if p.suffix.lower() in VIDEO_SUFFIXES
    )
    for gid, path in enumerate(videos):
        label = LABELS[path.parent.name]
        frames = extract_landmarks(str(path), max_frames=args.max_frames)
        if len(frames) < WINDOW:
            print(f"skip {path.name}: only {len(frames)} frames with a person")
            continue
        windows = sliding_windows(frames, stride=args.stride)
        X.append(windows)
        y += [label] * len(windows)
        groups += [gid] * len(windows)
        index.append({"group": gid, "file": path.name, "label": label, "windows": len(windows)})
        print(f"{path.parent.name}/{path.name}: {len(windows)} windows")

    if not X:
        raise SystemExit("no usable clip found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", np.concatenate(X).astype(np.float32))
    np.save(out / "y.npy", np.asarray(y, dtype=np.float32))
    np.save(out / "groups.npy", np.asarray(groups, dtype=np.int64))
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {sum(len(w) for w in X)} windows from {len(index)} clips to {out}")


if __name__ == "__main__":
    main()
