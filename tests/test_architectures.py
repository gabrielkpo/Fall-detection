import json

import pytest
import torch

from fall_detection import inference
from fall_detection.architectures import ARCHITECTURES, build_model, n_parameters


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_same_contract_for_every_architecture(arch):
    model = build_model(arch).eval()
    with torch.no_grad():
        out = model(torch.randn(4, 132, 30))
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()
    assert n_parameters(model) < 200_000


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_onnx_export_is_faithful(arch, tmp_path):
    pytest.importorskip("onnxruntime")
    from fall_detection.export import export_onnx

    check = export_onnx(build_model(arch), str(tmp_path / "m.onnx"))
    assert check["max_abs_diff"] < 1e-4


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError):
        build_model("resnet152")


def test_load_model_follows_the_champion_metadata(tmp_path, monkeypatch):
    weights, meta = tmp_path / "w.pth", tmp_path / "m.json"
    torch.save(build_model("tcn").state_dict(), weights)
    meta.write_text(json.dumps({"arch": "tcn"}))
    monkeypatch.setenv("FALL_META_PATH", str(meta))
    model = inference.load_model(weights)
    assert type(model).__name__ == "TCN"
