"""Export models/champion.pth to ONNX for the in-browser demo and check parity with PyTorch.

    python scripts/export_onnx.py            # writes site/model.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from fall_detection.inference import N_FEATURES, WINDOW, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/champion.pth")
    parser.add_argument("--out", default="site/model.onnx")
    args = parser.parse_args()

    model = load_model(args.weights)
    dummy = torch.randn(1, N_FEATURES, WINDOW)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["window"],
        output_names=["fall_probability"],
        dynamic_axes={"window": {0: "batch"}, "fall_probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(args.out)
    x = np.random.default_rng(0).random((64, N_FEATURES, WINDOW), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    got = session.run(None, {"window": x})[0]
    max_diff = float(np.abs(expected - got).max())
    print(f"wrote {args.out}; max |torch - onnx| = {max_diff:.2e}")
    assert max_diff < 1e-5, "ONNX output differs from PyTorch"


if __name__ == "__main__":
    main()
