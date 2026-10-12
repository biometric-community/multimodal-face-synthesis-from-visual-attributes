"""Attribute L2 and simple feature-stats FID proxy (no Inception required)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def attr_l2(pred_attrs: torch.Tensor, gt_attrs: torch.Tensor) -> float:
    return float(F.mse_loss(pred_attrs, gt_attrs).detach())


@torch.no_grad()
def fid_proxy_stats(real: torch.Tensor, fake: torch.Tensor) -> float:
    """
    Frechet distance between spatial-mean feature vectors (RGB channels stats).
    Not paper Inception FID — documented Dn; still useful relative metric.
    """
    def feats(x):
        # x: N×3×H×W in [-1,1]
        x = (x + 1) * 0.5
        mu = x.mean(dim=(2, 3))  # N×3
        return mu.cpu().numpy()

    r = feats(real)
    f = feats(fake)
    mu_r, mu_f = r.mean(0), f.mean(0)
    # diagonal cov approximation for stability
    var_r = r.var(0) + 1e-6
    var_f = f.var(0) + 1e-6
    diff = mu_r - mu_f
    return float(diff @ diff + np.sum(var_r + var_f - 2 * np.sqrt(var_r * var_f)))
