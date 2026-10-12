"""Evaluate Att2MFace: attr L2 via D classifier + FID proxy on visible."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from att2mface.data import build_dataloader
from att2mface.metrics import attr_l2, fid_proxy_stats
from att2mface.models import build_models


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(
        args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    loader = build_dataloader(cfg, "test")
    attr_dim = loader.dataset.attr_dim
    G, D = build_models(cfg, attr_dim)
    ckpt = Path(args.checkpoint or Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    G.load_state_dict(state["G"])
    D.load_state_dict(state["D"])
    G, D = G.to(device).eval(), D.to(device).eval()

    attr_errs, reals, fakes = [], [], []
    n = 0
    max_n = int(cfg["eval"].get("max_images", 512))
    with torch.no_grad():
        for batch in loader:
            attrs = batch["attr"].to(device)
            real_v = batch["visible"].to(device)
            fake = G(attrs)
            _, pred_cls = D(fake["visible"])
            attr_errs.append(attr_l2(pred_cls, attrs))
            reals.append(real_v.cpu())
            fakes.append(fake["visible"].cpu())
            n += real_v.size(0)
            if n >= max_n:
                break
    real_t = torch.cat(reals, 0)[:max_n]
    fake_t = torch.cat(fakes, 0)[:max_n]
    results = {
        "attr_l2": float(sum(attr_errs) / max(len(attr_errs), 1)),
        "fid_proxy": fid_proxy_stats(real_t, fake_t),
        "n": int(min(n, max_n)),
        "checkpoint": str(ckpt),
        "image_size": cfg["model"].get("image_size"),
        "note": "fid_proxy is channel-mean Frechet distance (not Inception FID); see DEVIATIONS.md",
    }
    out = Path(cfg["paths"]["log_dir"]) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
