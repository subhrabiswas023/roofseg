import torch


def test_model_output_shape(mock_model):
    input = torch.randn(2, 3, 256, 256)
    output = mock_model(input)

    assert output.shape == (2, 2, 256, 256)
