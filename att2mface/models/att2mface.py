"""Att2MFace generator with multimodal stretch-out heads (paper Fig. 2–3, simplified)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


class EqualLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, lr_mul: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        nn.init.kaiming_normal_(self.conv.weight, a=0.2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_norm(self.act(self.conv(x)))


class StretchOut(nn.Module):
    """Per-modality head from shared features → RGB (paper stretch-out)."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 3, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Att2MFace(nn.Module):
    """
    Attribute + noise → shared progressive feature map → stretch-out modalities.
    Practical fixed-resolution build (default 64²); progressive schedule documented as Dn.
    """

    def __init__(
        self,
        attr_dim: int = 12,
        z_dim: int = 128,
        image_size: int = 64,
        base_ch: int = 512,
        modalities: Tuple[str, ...] = ("visible", "sketch"),
    ):
        super().__init__()
        self.attr_dim = attr_dim
        self.z_dim = z_dim
        self.image_size = image_size
        self.modalities = modalities
        self.map = nn.Sequential(
            EqualLinear(attr_dim + z_dim, base_ch),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(base_ch, base_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # grow from 4x4
        n_up = int(math.log2(image_size) - 2)
        chs = [base_ch]
        for i in range(n_up):
            chs.append(max(base_ch // (2 ** (i + 1)), 64))
        self.const = nn.Parameter(torch.randn(1, base_ch, 4, 4))
        blocks = []
        for i in range(n_up):
            blocks.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), ConvBlock(chs[i], chs[i + 1])))
        self.blocks = nn.ModuleList(blocks)
        self.to_mod = nn.ModuleDict({m: StretchOut(chs[-1]) for m in modalities})
        self.out_ch = chs[-1]

    def forward(self, attr: torch.Tensor, z: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        b = attr.size(0)
        if z is None:
            z = torch.randn(b, self.z_dim, device=attr.device)
        h = self.map(torch.cat([attr, z], dim=1))
        x = self.const.repeat(b, 1, 1, 1)
        x = x + h.view(b, -1, 1, 1)
        x = pixel_norm(x)
        for blk in self.blocks:
            x = blk(x)
        return {m: self.to_mod[m](x) for m in self.modalities}


class Discriminator(nn.Module):
    """Shared stretch-in backbone + src head + attr classifier (paper stretch-in)."""

    def __init__(self, attr_dim: int = 12, image_size: int = 64, base_ch: int = 64):
        super().__init__()
        layers: List[nn.Module] = []
        ch = 3
        cur = base_ch
        size = image_size
        while size > 4:
            layers += [
                nn.Conv2d(ch, cur, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = cur
            cur = min(cur * 2, 512)
            size //= 2
        self.encode = nn.Sequential(*layers)
        self.conv_final = nn.Conv2d(ch, 512, 4, 1, 0)
        self.src = nn.Linear(512, 1)
        self.cls = nn.Linear(512, attr_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        h = F.leaky_relu(self.conv_final(h), 0.2).flatten(1)
        return self.src(h), torch.sigmoid(self.cls(h))


def build_models(cfg: dict, attr_dim: int) -> Tuple[Att2MFace, Discriminator]:
    m = cfg["model"]
    G = Att2MFace(
        attr_dim=attr_dim,
        z_dim=int(m.get("z_dim", 128)),
        image_size=int(m.get("image_size", 64)),
        base_ch=int(m.get("base_ch", 512)),
        modalities=tuple(m.get("modalities", ["visible", "sketch"])),
    )
    D = Discriminator(
        attr_dim=attr_dim,
        image_size=int(m.get("image_size", 64)),
        base_ch=int(m.get("d_base_ch", 64)),
    )
    return G, D
