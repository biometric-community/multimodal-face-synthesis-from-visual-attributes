"""Train Att2MFace on CelebA (visible + sketch)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from att2mface.data import build_dataloader
from att2mface.losses import Att2MFaceLoss
from att2mface.models import build_models


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device(
        args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(int(cfg["train"].get("seed", 42)))

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(cfg, "train")
    attr_dim = train_loader.dataset.attr_dim
    G, D = build_models(cfg, attr_dim)
    G, D = G.to(device), D.to(device)
    n_g = sum(p.numel() for p in G.parameters()) / 1e6
    print(f"Att2MFace G={n_g:.2f}M attr_dim={attr_dim} n_train={len(train_loader.dataset)} device={device}", flush=True)

    opt_g = torch.optim.Adam(G.parameters(), lr=float(cfg["train"]["lr"]), betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=float(cfg["train"]["lr"]), betas=(0.0, 0.99))
    criterion = Att2MFaceLoss(
        lambda_rec=float(cfg["train"].get("lambda_rec", 10.0)),
        lambda_cls=float(cfg["train"].get("lambda_cls", 1.0)),
    )

    epochs = int(cfg["train"]["epochs"])
    max_steps = cfg["train"].get("max_steps")
    log_every = int(cfg["train"].get("log_every", 50))
    history = {"epochs": [], "g_loss": [], "d_loss": []}
    global_step = 0
    t0 = time.time()
    modalities = list(cfg["model"].get("modalities", ["visible", "sketch"]))

    for epoch in range(epochs):
        G.train()
        D.train()
        run_g, run_d, n_b = 0.0, 0.0, 0
        for batch in train_loader:
            attrs = batch["attr"].to(device)
            real = {m: batch[m].to(device) for m in modalities if m in batch}
            # --- D ---
            with torch.no_grad():
                fake = G(attrs)
            # train D on visible primarily + sketch
            d_losses = []
            for m in modalities:
                real_logit, real_cls = D(real[m])
                fake_logit, _ = D(fake[m].detach())
                loss_d, parts_d = criterion.d_loss(real_logit, fake_logit, real_cls, attrs)
                d_losses.append(loss_d)
            loss_d = sum(d_losses) / len(d_losses)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # --- G ---
            fake = G(attrs)
            # adversarial on visible
            fake_logit, fake_cls = D(fake["visible"])
            loss_g, parts_g = criterion.g_loss(fake_logit, fake_cls, attrs, fake, real)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            run_g += float(loss_g.detach())
            run_d += float(loss_d.detach())
            n_b += 1
            global_step += 1
            if global_step % log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"G={float(loss_g.detach()):.3f} D={float(loss_d.detach()):.3f} "
                    f"rec={parts_g['g_rec']:.3f} cls={parts_g['g_cls']:.3f}",
                    flush=True,
                )
            if max_steps is not None and global_step >= int(max_steps):
                break

        history["epochs"].append(epoch)
        history["g_loss"].append(run_g / max(n_b, 1))
        history["d_loss"].append(run_d / max(n_b, 1))
        blob = {"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch, "attr_dim": attr_dim, "cfg": cfg}
        torch.save(blob, ckpt_dir / "latest.pt")
        torch.save(blob, ckpt_dir / "best.pt")
        print(f"epoch={epoch} G={history['g_loss'][-1]:.3f} D={history['d_loss'][-1]:.3f}", flush=True)
        if max_steps is not None and global_step >= int(max_steps):
            break

    summary = {
        "epoch": history["epochs"][-1] if history["epochs"] else -1,
        "g_loss": history["g_loss"][-1] if history["g_loss"] else None,
        "d_loss": history["d_loss"][-1] if history["d_loss"] else None,
        "protocol": cfg["train"].get("protocol"),
        "seconds": time.time() - t0,
        "n_params_g_m": n_g,
        "attr_dim": attr_dim,
        "image_size": cfg["model"].get("image_size"),
        "dataset": "celeba",
    }
    (log_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
