"""REPORT.md + figures from real logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from att2mface.plot_style import FIGSIZE, LABEL_SIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, TITLE_SIZE, apply_rcparams, apply_style


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    hist = json.loads((log_dir / "train_history.json").read_text()) if (log_dir / "train_history.json").is_file() else None
    summary = json.loads((log_dir / "train_summary.json").read_text()) if (log_dir / "train_summary.json").is_file() else {}
    eval_js = json.loads((log_dir / "eval_results.json").read_text()) if (log_dir / "eval_results.json").is_file() else {}

    if hist:
        apply_rcparams()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(hist["epochs"], hist["g_loss"], color=MULTI_SERIES_COLORS[0], lw=LINEWIDTH_MAIN, label="G")
        ax.plot(hist["epochs"], hist["d_loss"], color=MULTI_SERIES_COLORS[1], lw=LINEWIDTH_MAIN, label="D")
        ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
        ax.set_ylabel("Loss", fontsize=LABEL_SIZE)
        ax.set_title("Att2MFace training losses", fontsize=TITLE_SIZE)
        ax.legend()
        apply_style(ax)
        fig.savefig(fig_dir / "train_loss.svg", bbox_inches="tight")
        plt.close(fig)

    def fmt(v, nd=4):
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return "n/a"

    Path("REPORT.md").write_text(
        f"""# Att2MFace — Results

Paper: Di & Patel, TBIOM 2021 (attribute → multimodal face synthesis).

## Training

| Metric | Value |
|--------|------:|
| Protocol | {summary.get('protocol', 'n/a')} |
| Epoch | {summary.get('epoch', 'n/a')} |
| G loss | {fmt(summary.get('g_loss'))} |
| D loss | {fmt(summary.get('d_loss'))} |
| Image size | {summary.get('image_size', 'n/a')} |
| Attr dim | {summary.get('attr_dim', 'n/a')} |

![loss](outputs/figures/train_loss.svg)

## Test (CelebA)

| Metric | Value |
|--------|------:|
| Attr L2 | {fmt(eval_js.get('attr_l2'))} |
| FID proxy | {fmt(eval_js.get('fid_proxy'), 3)} |
| N | {eval_js.get('n', 'n/a')} |

Real CelebA faces only; sketch modality from pencil-sketch of RGB (see DEVIATIONS.md).
"""
    )
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
