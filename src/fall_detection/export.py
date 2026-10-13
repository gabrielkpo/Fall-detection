"""ONNX export with a parity check and a CPU latency measurement (onnx + onnxruntime needed)."""

from __future__ import annotations

import time

import numpy as np
import torch

from fall_detection.inference import N_FEATURES, WINDOW


def export_onnx(model: torch.nn.Module, out: str) -> dict:
    """Export, then compare ONNX and PyTorch outputs and time one window on the CPU."""
    import onnxruntime as ort

    model = model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, N_FEATURES, WINDOW),
        out,
        input_names=["window"],
        output_names=["fall_probability"],
        dynamic_axes={"window": {0: "batch"}, "fall_probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((64, N_FEATURES, WINDOW), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    got = session.run(None, {"window": x})[0]

    one = x[:1]
    for _ in range(20):  # warm-up
        session.run(None, {"window": one})
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        session.run(None, {"window": one})
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "max_abs_diff": float(np.abs(expected - got).max()),
        "latency_p95_ms": float(np.percentile(times, 95)),
    }
