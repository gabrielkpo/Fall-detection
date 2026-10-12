# Evaluation

Reproduce: `python -m fall_detection.evaluation --data data/processed` (about 1 minute on CPU).
Raw output: [`evaluation.json`](evaluation.json).

## Data

- 294 windows of 30 frames (33 MediaPipe landmarks x 4 values), 100 falls and 194 non-falls.
- Windows are non-overlapping chunks cut from videos stored as frame folders.
- `X.npy` has **no video id**. Windows were grouped back into 83 pseudo-videos from temporal
  continuity (see `reconstruct_groups`). This is a heuristic, not ground truth.

## Protocol

Stratified, group-wise 5-fold cross-validation: all windows of a pseudo-video are in the same
fold, so the model is always scored on videos it has never seen. Training recipe is the one of
`scripts/train.py` (Adam 1e-3, BCE, batch 32, early stopping on a group-wise validation slice).
Threshold 0.5. Every window is predicted exactly once (out-of-fold).

## Results (out-of-video, n = 294)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 82.3 % | 77.5 - 86.2 % |
| Precision | 77.9 % | |
| Recall (falls caught) | 67.0 % | 57.3 - 75.4 % |
| F1 | 0.72 | |

Confusion matrix: TP 67, FN 33, FP 19, TN 175.

Sensitivity to the choices above (same data):

| Variant | Pseudo-videos | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Default (jump quantile 0.99, seed 42) | 83 | 0.823 | 0.779 | 0.670 | 0.720 |
| Quantile 0.95 | 92 | 0.833 | 0.807 | 0.670 | 0.732 |
| Quantile 0.999 | 74 | 0.789 | 0.676 | 0.730 | 0.702 |
| Seed 1 | 83 | 0.806 | 0.717 | 0.710 | 0.714 |
| Seed 7 | 83 | 0.837 | 0.842 | 0.640 | 0.727 |

Across these variants: accuracy 79 - 84 %, recall 64 - 73 %.

## Latency

Model only, one window, CPU, batch of 1: median 0.12 ms, p95 0.14 ms (31,489 parameters).
Pose extraction (MediaPipe, one frame at a time, CPU) is **not** included and is the real
bottleneck: about 0.2 s per frame measured in the Docker image on an Apple Silicon laptop
(240 frames in about 50 s). The demo therefore processes at most 150 frames per video.

## What this does not say

- The earlier README claim of ">90 % accuracy" came from a random split of windows, where windows
  of the same video are on both sides. On the original random split the shipped weights score
  30/30 on a 30-window test set, which is too small to mean anything and likely leaks.
- 294 windows from a few dozen videos recorded by one person: results may not transfer to other
  people, rooms or camera angles.
- One in three falls is missed (recall 67 %). This is **not** suitable for safety-critical use.
- Pseudo-video groups can be wrong (two videos merged, one video split). The sensitivity table
  shows the conclusion holds for the thresholds tried, not that it is exact.
- The model has no "not a person" class: random landmark values are scored as a fall with
  probability 1.0 (checked on the served API). Input must come from a pose detector.
- The training frame rate is not recorded; the demo feeds consecutive frames of the uploaded video.

## In-browser demo parity

The static demo uses MediaPipe's JavaScript Pose Landmarker and the ONNX export of the model.
Checked on the same 8 s clip (window scores, stride 15):

| Window | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Python (`mp.solutions.pose`) | 0.0007 | 0.0126 | 0.2523 | 0.8100 | 0.8683 | 0.7982 |
| Browser (Tasks API + ONNX) | 0.0007 | 0.0106 | 0.2287 | 0.8192 | 0.8903 | 0.8290 |

ONNX vs PyTorch: max absolute difference 0 on 64 random windows. The browser path must resize
frames to 640x480 first (as the training extraction did): without it, the z coordinates differ by
a factor of about 2 and every window scores at the floor value. Differences up to about 0.03 remain
between the two MediaPipe implementations; this has been checked on one clip only.
