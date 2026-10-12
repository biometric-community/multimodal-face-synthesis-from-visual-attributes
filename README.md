# Multimodal Face Synthesis from Visual Attributes (Att2MFace)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

PyTorch rebuild of Di & Patel (TBIOM 2021): attribute-conditioned multimodal face synthesis (visible + sketch) on real CelebA.

**Repository:** https://github.com/biometric-community/multimodal-face-synthesis-from-visual-attributes

```bash
bash scripts/setup_env.sh
bash scripts/train.sh
bash scripts/train_full.sh
bash scripts/eval.sh && bash scripts/predict.sh && bash scripts/report.sh
```

See [REPORT.md](REPORT.md), [DEVIATIONS.md](DEVIATIONS.md), [FIDELITY_AUDIT.md](FIDELITY_AUDIT.md).
