"""FastAPI service: the model runs server-side, nothing sensitive reaches the client."""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from fall_detection import inference, monitoring

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
THRESHOLD = float(os.environ.get("FALL_THRESHOLD", "0.5"))


class LandmarkRequest(BaseModel):
    landmarks: list[list[float]] = Field(
        description="n_frames x 132 values (33 MediaPipe landmarks x x,y,z,visibility), n >= 30"
    )


class PredictionResponse(BaseModel):
    fall: bool
    score: float
    window_scores: list[float]
    threshold: float
    frames_used: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = inference.load_model()
    app.state.meta = inference.load_metadata()
    yield


app = FastAPI(
    title="Fall detection",
    description="Research prototype. Not a safety system: about 1 fall in 3 is missed.",
    lifespan=lifespan,
)


def _respond(frames: np.ndarray) -> PredictionResponse:
    try:
        pred = inference.predict(app.state.model, frames, threshold=THRESHOLD)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    log_path = os.environ.get("FALL_LOG_PATH")  # opt-in, anonymous feature summaries only
    if log_path:
        windows = inference.sliding_windows(frames)
        monitoring.log_windows(log_path, windows, np.asarray(pred.window_scores))
    return PredictionResponse(
        fall=pred.fall,
        score=pred.score,
        window_scores=pred.window_scores,
        threshold=THRESHOLD,
        frames_used=len(frames),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict:
    return app.state.meta


@app.post("/predict", response_model=PredictionResponse)
def predict_landmarks(body: LandmarkRequest) -> PredictionResponse:
    frames = np.asarray(body.landmarks, dtype=np.float32)
    if frames.ndim != 2:
        raise HTTPException(status_code=422, detail="landmarks must be a 2-D array")
    return _respond(frames)


@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video(file: UploadFile) -> PredictionResponse:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="video larger than 25 MB")
    from fall_detection.pose import extract_landmarks

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        frames = extract_landmarks(tmp.name)
    return _respond(frames)
