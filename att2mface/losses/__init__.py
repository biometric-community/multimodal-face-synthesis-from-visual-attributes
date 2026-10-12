"""GAN + reconstruction + attribute losses for Att2MFace."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Att2MFaceLoss(nn.Module):
    def __init__(self, lambda_rec: float = 10.0, lambda_cls: float = 1.0):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls
        self.bce = nn.BCEWithLogitsLoss()

    def d_loss(self, real_logit, fake_logit, real_cls, attrs):
        loss_real = self.bce(real_logit, torch.ones_like(real_logit))
        loss_fake = self.bce(fake_logit, torch.zeros_like(fake_logit))
        loss_cls = F.binary_cross_entropy(real_cls, attrs)
        total = loss_real + loss_fake + self.lambda_cls * loss_cls
        return total, {
            "d_real": float(loss_real.detach()),
            "d_fake": float(loss_fake.detach()),
            "d_cls": float(loss_cls.detach()),
        }

    def g_loss(self, fake_logit, fake_cls, attrs, fake_imgs: dict, real_imgs: dict):
        loss_adv = self.bce(fake_logit, torch.ones_like(fake_logit))
        loss_cls = F.binary_cross_entropy(fake_cls, attrs)
        loss_rec = 0.0
        for k in fake_imgs:
            if k in real_imgs:
                loss_rec = loss_rec + F.l1_loss(fake_imgs[k], real_imgs[k])
        total = loss_adv + self.lambda_cls * loss_cls + self.lambda_rec * loss_rec
        return total, {
            "g_adv": float(loss_adv.detach()),
            "g_cls": float(loss_cls.detach()),
            "g_rec": float(loss_rec.detach()) if torch.is_tensor(loss_rec) else float(loss_rec),
        }
