import torch

from fall_detection.model import FallCNN1D


def test_output_shape_and_range():
    model = FallCNN1D(input_channels=132, seq_length=30).eval()
    x = torch.randn(4, 132, 30)  # (batch, features, frames)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_accepts_other_sequence_lengths():
    # AdaptiveMaxPool1d makes the model independent of the number of frames.
    model = FallCNN1D(input_channels=132, seq_length=30).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 132, 45)).shape == (1, 1)
