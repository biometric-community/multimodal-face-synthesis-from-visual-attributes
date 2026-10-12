"""Generate visible/sketch grids from attributes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from att2mface.data import build_dataloader
from att2mface.models import build_models


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(
        args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    loader = build_dataloader(cfg, "test")
    attr_dim = loader.dataset.attr_dim
    G, _ = build_models(cfg, attr_dim)
    ckpt = Path(args.checkpoint or Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    G.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["G"])
    G.to(device).eval()
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for batch in loader:
            attrs = batch["attr"].to(device)
            fake = G(attrs)
            for i in range(attrs.size(0)):
                grid = torch.stack(
                    [
                        batch["visible"][i],
                        fake["visible"][i].cpu(),
                        batch["sketch"][i],
                        fake["sketch"][i].cpu(),
                    ],
                    dim=0,
                )
                # denorm for save
                grid = (grid * 0.5 + 0.5).clamp(0, 1)
                name = Path(batch["name"][i]).stem
                vutils.save_image(grid, out_dir / f"{name}_vis_gt_fake_sk_gt_fake.png", nrow=4)
                saved += 1
                if saved >= args.num:
                    print(f"saved {saved} → {out_dir}")
                    return
    print(f"saved {saved} → {out_dir}")


if __name__ == "__main__":
    main()
