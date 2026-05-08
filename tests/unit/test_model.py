import torch

def test_model_output_shape(model):
    input = torch.randn(2, 3, 256, 256)
    output = model(input)

    assert output.shape == (2, 2, 256, 256)
