# Deviations — Att2MFace

| ID | Paper | This rebuild | Rationale |
|----|-------|--------------|-----------|
| D1 | ARL / CelebA-HQ / CASIA NIR-VIS multimodal | CelebA visible + pencil-sketch | Local public data; ARL/CASIA stubs empty |
| D2 | Progressive 4→256 with long iters | Fixed 64², 20 epochs | Single-GPU practicality |
| D3 | Inception FID / LPIPS | Channel-mean FID proxy + Attr L2 | Lightweight metrics, documented |
| D4 | Pixel-wise equalization ProgressiveGAN | PixelNorm + EqualLinear | Structural proxy |
