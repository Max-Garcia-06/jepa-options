"""JEPA components: context encoder, predictor, VICReg regularizers."""

import torch
from torch import nn

from volj.config import D, HIDDEN


class ContextEncoder(nn.Module):
    """GRU -> LayerNorm -> Linear. Maps (B, T, F) windows to (B, d) embeddings."""

    def __init__(self, n_features: int, hidden: int = HIDDEN, d: int = D):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.proj(self.norm(h[-1]))


class Predictor(nn.Module):
    def __init__(self, d: int = D, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def vicreg_terms(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance hinge and covariance penalty on a (batch, d) embedding."""
    n, d = z.shape
    zc = z - z.mean(dim=0)
    var = zc.pow(2).sum(dim=0) / (n - 1)
    var_loss = torch.relu(1.0 - torch.sqrt(var + 1e-4)).mean()
    cov = (zc.T @ zc) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / d
    return var_loss, cov_loss
