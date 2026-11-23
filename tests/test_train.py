import numpy as np

from volj.model import ContextEncoder
from volj.train import build_windows, pretrain
import torch


def _ar1(n=400, f=10, phi=0.9, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, f))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal(f)
    return (x - x.mean(0)) / x.std(0)


def _out(enc, X):
    with torch.no_grad():
        return enc(torch.tensor(X[:60][None], dtype=torch.float32)).numpy()


def test_build_windows():
    X = np.arange(100)[:, None].astype(float)
    ctx, tgt = build_windows(X, np.array([59, 60, 90]), L=60, h=21)
    assert ctx.shape == (2, 60, 1) and tgt.shape == (2, 21, 1)
    assert ctx[0, -1, 0] == 59
    assert tgt[0, 0, 0] == 60
    assert tgt[0, -1, 0] == 80


def test_pretrain_deterministic():
    X = _ar1(300)
    pos = np.arange(59, 270)
    a = _out(pretrain(X, pos, seed=0, epochs=1), X)
    b = _out(pretrain(X, pos, seed=0, epochs=1), X)
    c = _out(pretrain(X, pos, seed=1, epochs=1), X)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_pretrain_returns_context_only():
    X = _ar1(300)
    enc = pretrain(X, np.arange(59, 270), seed=0, epochs=1)
    assert type(enc) is ContextEncoder
    assert not hasattr(enc, "target")
    assert not enc.training


def test_loss_decreases():
    X = _ar1(400)
    enc = pretrain(X, np.arange(59, 379), seed=0, epochs=5)
    assert len(enc.loss_history) == 5
    assert enc.loss_history[-1] < enc.loss_history[0]


def test_train_pos_window_bound():
    X = _ar1(300)
    ctx, _ = build_windows(X, np.arange(59, 300))
    assert len(ctx) == 300 - 21 - 59
