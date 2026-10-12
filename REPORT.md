# Att2MFace — Results

Paper: Di & Patel, TBIOM 2021 (attribute → multimodal face synthesis).

## Training

| Metric | Value |
|--------|------:|
| Protocol | smoke |
| Epoch | 0 |
| G loss | 8.7492 |
| D loss | 3.4174 |
| Image size | 64 |
| Attr dim | 12 |

![loss](outputs/figures/train_loss.svg)

## Test (CelebA)

| Metric | Value |
|--------|------:|
| Attr L2 | 0.1856 |
| FID proxy | 0.061 |
| N | 64 |

Real CelebA faces only; sketch modality from pencil-sketch of RGB (see DEVIATIONS.md).
