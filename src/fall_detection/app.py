"""Gradio demo mounted on the API (entry point of the Docker image / Hugging Face Space)."""

from __future__ import annotations

import gradio as gr

from fall_detection import inference
from fall_detection.api import THRESHOLD
from fall_detection.api import app as api
from fall_detection.pose import extract_landmarks

NOTICE = (
    "Research prototype, **not a safety system**. Out-of-video evaluation: accuracy 82 %, "
    "recall 67 % (about 1 fall in 3 is missed). The video is processed in memory and is not "
    "stored; only anonymous statistics may be logged."
)

_model = None


def analyze(video_path: str):
    global _model
    if video_path is None:
        raise gr.Error("Upload a short video (a few seconds, one person in frame).")
    _model = _model or inference.load_model()
    frames = extract_landmarks(video_path)
    try:
        pred = inference.predict(_model, frames, threshold=THRESHOLD)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    label = {"Fall detected": pred.score, "No fall": 1 - pred.score}
    return label, {"frames_used": len(frames), "window_scores": pred.window_scores}


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Video(label="Short video (<= 25 MB)"),
    outputs=[gr.Label(label="Result"), gr.JSON(label="Details")],
    title="Fall detection (MediaPipe pose + 1D CNN)",
    description=NOTICE,
)

app = gr.mount_gradio_app(api, demo, path="/")
