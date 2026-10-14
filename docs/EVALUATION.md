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
fold, so the model is always scored on videos it has never seen. Training recipe is the original
one (Adam 1e-3, BCE, batch 32), with early stopping on a group-wise validation slice instead of
the random split of the first training script, which leaked windows of a video across splits.
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

## Frozen test split and promotion

`models/frozen_split.json` holds one fold of videos (18 pseudo-videos, 61 windows, 20 falls),
never used for training or model selection, tied to the dataset by a SHA-256 fingerprint.
The first champion (0.1.0) had been trained on a random split that contained these windows, so it
was retrained without them (0.2.0). On the frozen test, 61 windows only:

| | Value | 95% CI |
|---|---|---|
| Accuracy | 85.2 % | 74.3 - 92.0 % |
| Recall | 80.0 % | 58.4 - 91.9 % |
| Precision | 76.2 % | |
| F1 | 0.78 | |

The intervals are wide and overlap the cross-validation figures above: do not read the higher
point estimates as an improvement. A candidate is promoted only if the lower bound of the 95 %
paired bootstrap interval (resampling videos) of the F1 gain is above zero.

Raw data is not in the repository (its origin and licence are not documented). The promotion
workflow therefore needs a private `DATA_URL` secret and is skipped until it exists.

## Feature representations (negative result)

Tried to make the input independent of camera distance and position (the demo showed false alarms
standing up and missed falls on a real webcam). Same group-wise 5-fold CV, 3 seeds, 294 windows:

| Input | Accuracy | Recall | F1 | Delta F1 vs raw (95 % CI) |
|---|---|---|---|---|
| Raw landmarks, 132 channels (current) | 0.830 | 0.673 | 0.729 | - |
| Centred on the hips, scaled by torso length, x/y only (66) | 0.830 | 0.670 | 0.728 | -0.06 (-0.15 to +0.02) |
| Derived body features: torso angle, relative heights, bounding-box aspect, velocities (11) | 0.763 | 0.473 | 0.575 | -0.17 (-0.29 to -0.05) |
| Derived without absolute hip height (10) | 0.779 | 0.493 | 0.602 | -0.14 (-0.25 to -0.03) |

Nothing beats the raw input; the hand-made features are clearly worse. Cross-validation on a
single-person, single-camera dataset cannot reward camera invariance, so this result does **not**
say invariance is useless on a new camera: it says the current data cannot show it. The next
measurable step is more varied data (other angles, distances, rooms, people, and hard negatives
such as sitting down or picking something up), recorded with `scripts/ingest_clips.py`, which also
keeps the video id of each window so splits are exact.

Code: `src/fall_detection/features.py`, `scripts/feature_experiment.py`
(`docs/feature_experiment.json` has the raw numbers). The features are not used in production.

## Architecture tournament (first run)

`python -m fall_detection.tournament`, 2 seeds, group-wise 5-fold CV on the 233 windows outside
the frozen test split (the test split is not looked at). All four architectures are about the
same size (19 k to 31 k parameters), export faithfully to ONNX and run in under 0.2 ms per window.

| Architecture | F1 | Recall | Accuracy | Delta F1 vs CNN (95 % CI) |
|---|---|---|---|---|
| CNN 1D (champion) | 0.564 | 0.481 | 0.745 | - |
| GRU | 0.662 | 0.625 | 0.777 | +0.093 (+0.011 to +0.181) |
| TCN | 0.619 | 0.531 | 0.775 | +0.023 (-0.031 to +0.076) |
| Transformer | 0.742 | 0.713 | 0.828 | +0.169 (+0.076 to +0.268) |

The transformer won and was then judged once on the frozen test split (61 windows):

| | Accuracy | Precision | Recall | F1 | Missed falls | False alarms |
|---|---|---|---|---|---|---|
| CNN 1D (champion 0.2.0) | 0.852 | 0.762 | 0.800 | 0.780 | 4 | 5 |
| Transformer (candidate) | 0.852 | 0.923 | 0.600 | 0.727 | 8 | 1 |

**Not promoted.** The F1 gain is not proven (95 % CI -0.21 to +0.14) and recall drops by 0.20:
the transformer trades missed falls for fewer false alarms, which the rules forbid for a fall
detector. The cross-validation lead did not survive the held-out test: with 294 windows from one
person, both numbers are noisy, and the frozen split is exactly what keeps such a lead from being
shipped on luck. To rerun once more data exists: `gh workflow run promote.yml -f arch=auto`.
