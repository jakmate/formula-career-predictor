import torch
import torch.nn as nn
from app.core.pytorch_model import RacingPredictor


def test_model_initialization():
    model = RacingPredictor(input_dim=10)
    assert isinstance(model, nn.Module)


def test_forward_pass():
    model = RacingPredictor(input_dim=5)
    x = torch.randn(32, 5)
    output = model(x)
    assert output.shape == (32,)
