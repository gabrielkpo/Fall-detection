"""Export the champion to ONNX for the in-browser demo, with a parity check against PyTorch.

    python scripts/export_onnx.py            # writes site/model.onnx
"""

from __future__ import annotations

import argparse

from fall_detection.export import export_onnx
from fall_detection.inference import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/champion.pth")
    parser.add_argument("--arch", default=None, help="defaults to the one in models/champion.json")
    parser.add_argument("--out", default="site/model.onnx")
    args = parser.parse_args()

    check = export_onnx(load_model(args.weights, arch=args.arch), args.out)
    print(f"wrote {args.out}; {check}")
    assert check["max_abs_diff"] < 1e-4, "ONNX output differs from PyTorch"


if __name__ == "__main__":
    main()
