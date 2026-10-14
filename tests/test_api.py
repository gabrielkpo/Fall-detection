import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from fall_detection import inference
from fall_detection.model import FallCNN1D


@pytest.fixture()
def client(tmp_path, monkeypatch):
    weights = tmp_path / "w.pth"
    torch.save(FallCNN1D(132, 30).state_dict(), weights)
    meta = tmp_path / "m.json"
    meta.write_text('{"version": "test"}')
    monkeypatch.setenv("FALL_MODEL_PATH", str(weights))
    monkeypatch.setenv("FALL_META_PATH", str(meta))
    from fall_detection.api import app

    with TestClient(app) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model").json() == {"version": "test"}


def test_predict_returns_scores_for_each_window(client):
    frames = np.random.rand(75, 132).tolist()  # 75 frames, stride 15 -> 4 windows
    r = client.post("/predict", json={"landmarks": frames})
    assert r.status_code == 200
    body = r.json()
    assert len(body["window_scores"]) == 4
    assert 0.0 <= body["score"] <= 1.0
    assert body["frames_used"] == 75


def test_predict_rejects_too_few_frames(client):
    r = client.post("/predict", json={"landmarks": np.random.rand(10, 132).tolist()})
    assert r.status_code == 422


def test_predict_rejects_wrong_feature_count(client):
    r = client.post("/predict", json={"landmarks": np.random.rand(40, 10).tolist()})
    assert r.status_code == 422


def test_predict_rejects_too_many_frames(client):
    r = client.post("/predict", json={"landmarks": np.zeros((901, 132)).tolist()})
    assert r.status_code == 422


def test_sliding_windows_shape():
    w = inference.sliding_windows(np.zeros((60, 132), dtype=np.float32), stride=10)
    assert w.shape == (4, 30, 132)
