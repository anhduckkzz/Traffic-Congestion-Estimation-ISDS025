import torch

from src.models.segmentation.deeplab_seresnet import build_segmentation_model


def test_deeplab_seresnet_forward():
    model = build_segmentation_model(num_classes=2)
    dummy = torch.randn(2, 3, 256, 256)
    outputs = model(dummy)
    assert "out" in outputs
    out = outputs["out"]
    assert out.shape == (2, 2, 256, 256)
    out.mean().backward()
