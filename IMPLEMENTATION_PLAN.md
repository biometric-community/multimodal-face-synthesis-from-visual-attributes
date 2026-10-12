# Implementation Plan — Att2MFace

**Paper:** Di & Patel, TBIOM 2021  
**Package:** `att2mface`

## Method
Attribute + noise → shared generator → multimodal stretch-out heads (visible, sketch).
Discriminator stretch-in + attribute classifier.

## Data
CelebA (real). Sketch modality = pencil-sketch of RGB (Dn vs CelebA-HQ/ARL/CASIA).
