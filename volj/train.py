"""JEPA pretraining. The EMA target encoder lives and dies in this module."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from volj.config import BATCH, EMA, EPOCHS, H, L, LR, WD, W_COV, W_SIM, W_VAR
from volj.model import ContextEncoder, Predictor, vicreg_terms


def build_windows(
    X: np.ndarray, idx: np.ndarray, L: int = L, h: int = H
) -> tuple[np.ndarray, np.ndarray]:
    """Context X[t-L+1..t] and future X[t+1..t+h] for each t in idx.

    Positions lacking a full context or future window are skipped.
    """
    ctx, tgt = [], []
    for t in np.asarray(idx, dtype=int):
        if t - L + 1 < 0 or t + h > len(X) - 1:
            continue
        ctx.append(X[t - L + 1 : t + 1])
        tgt.append(X[t + 1 : t + h + 1])
    n_f = X.shape[1]
    if not ctx:
        return np.empty((0, L, n_f)), np.empty((0, h, n_f))
    return np.stack(ctx), np.stack(tgt)


def pretrain(
    X: np.ndarray, train_pos: np.ndarray, seed: int, epochs: int = EPOCHS
) -> ContextEncoder:
    """Pretrain with EMA-target JEPA + VICReg; return the context encoder only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_pos = np.asarray(train_pos, dtype=int)
    # Target windows must end at or before the last train position.
    usable = train_pos[train_pos + H <= train_pos.max()]
    ctx_np, tgt_np = build_windows(X, usable)
    if len(ctx_np) < 2:
        raise ValueError("too few training windows to pretrain")
    ctx = torch.tensor(ctx_np, dtype=torch.float32)
    tgt = torch.tensor(tgt_np, dtype=torch.float32)

    n_feat = X.shape[1]
    encoder = ContextEncoder(n_feat)
    predictor = Predictor()
    _tgt_encoder = copy.deepcopy(encoder)
    for p in _tgt_encoder.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=LR, weight_decay=WD
    )
    gen = torch.Generator().manual_seed(seed)
    history: list[float] = []
    n = len(ctx)
    for _ in range(epochs):
        encoder.train()
        predictor.train()
        _tgt_encoder.eval()
        perm = torch.randperm(n, generator=gen)
        total, count = 0.0, 0
        starts = list(range(0, n, BATCH))
        for k, i in enumerate(starts):
            # a lone trailing sample is merged into the previous batch (var/cov need >=2)
            if k == len(starts) - 1 and n - i == 1 and k > 0:
                break
            end = n if (k == len(starts) - 2 and n - starts[-1] == 1) else i + BATCH
            b = perm[i:end]
            z = encoder(ctx[b])
            with torch.no_grad():
                z_tgt = _tgt_encoder(tgt[b])
            sim = F.mse_loss(predictor(z), z_tgt)
            var, cov = vicreg_terms(z)
            loss = W_SIM * sim + W_VAR * var + W_COV * cov
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pt, pc in zip(_tgt_encoder.parameters(), encoder.parameters()):
                    pt.mul_(EMA).add_(pc.detach(), alpha=1.0 - EMA)
            total += loss.item() * len(b)
            count += len(b)
        history.append(total / count)

    encoder.eval()
    encoder.loss_history = history
    return encoder
