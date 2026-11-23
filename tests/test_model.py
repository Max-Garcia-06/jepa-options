import torch

from volj.model import ContextEncoder, vicreg_terms


def test_encoder_shape():
    assert ContextEncoder(10)(torch.randn(4, 60, 10)).shape == (4, 32)


def test_vicreg_gradient_flows():
    z = torch.randn(64, 32, requires_grad=True)
    zs = z * 0.01
    v, c = vicreg_terms(zs)
    (v + c).backward()
    assert z.grad.abs().sum() > 0
