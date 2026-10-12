"""CelebA attribute → visible + sketch multimodal pairs (real faces only)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Compact attribute subset used in paper-style CelebA experiments
DEFAULT_ATTRS = [
    "Male",
    "Young",
    "Smiling",
    "Eyeglasses",
    "Bald",
    "Blond_Hair",
    "Black_Hair",
    "Brown_Hair",
    "Mustache",
    "No_Beard",
    "Heavy_Makeup",
    "Wearing_Lipstick",
]


def resolve_celeba_root(cfg: dict) -> Path:
    root = Path(cfg["data"].get("celeba_root", "../../datasets/celeba/extracted/celeba"))
    if not root.is_absolute():
        proj = Path(__file__).resolve().parents[2]
        cand = (proj / root).resolve()
        if cand.exists():
            root = cand
        else:
            mono = Path(__file__).resolve()
            for p in mono.parents:
                alt = p / "projects" / "datasets" / "celeba" / "extracted" / "celeba"
                if alt.exists():
                    root = alt
                    break
    if not (root / "img_align_celeba").is_dir():
        raise FileNotFoundError(f"CelebA img_align_celeba missing under {root}")
    return root


def pencil_sketch(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 → sketch float HxWx3 in [0,1] (proxy for paper sketch modality)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    sketch = np.clip(sketch, 0, 255).astype(np.float32) / 255.0
    return np.stack([sketch, sketch, sketch], axis=-1)


class CelebAAttrMultimodal(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        image_size: int = 64,
        attr_names: Optional[List[str]] = None,
        max_n: Optional[int] = None,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.attr_names = attr_names or list(DEFAULT_ATTRS)
        img_dir = self.root / "img_align_celeba"
        attr_file = self.root / "list_attr_celeba.txt"
        part_file = self.root / "list_eval_partition.txt"

        with open(attr_file) as f:
            lines = f.readlines()
        header = lines[1].strip().split()
        name_to_idx = {n: i for i, n in enumerate(header)}
        self.attr_idx = [name_to_idx[n] for n in self.attr_names]

        sid = {"train": 0, "val": 1, "test": 2}[split]
        part = {}
        if part_file.is_file():
            with open(part_file) as f:
                for line in f:
                    fn, p = line.strip().split()
                    part[fn] = int(p)

        samples: List[Tuple[str, np.ndarray]] = []
        for line in lines[2:]:
            toks = line.strip().split()
            fn = toks[0]
            if part and part.get(fn, 0) != sid:
                continue
            if not part and sid != 0:
                continue
            vals = np.array([int(toks[1:][i]) for i in self.attr_idx], dtype=np.float32)
            vals = (vals + 1.0) * 0.5  # {-1,1} → {0,1}
            samples.append((fn, vals))

        if max_n is not None:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(samples), size=min(max_n, len(samples)), replace=False)
            samples = [samples[i] for i in sorted(idx.tolist())]

        self.samples = samples
        self.img_dir = img_dir
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    @property
    def attr_dim(self) -> int:
        return len(self.attr_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fn, attrs = self.samples[idx]
        rgb = np.asarray(Image.open(self.img_dir / fn).convert("RGB"))
        sketch_np = pencil_sketch(rgb)
        # resize sketch with PIL for consistency
        sketch_img = Image.fromarray((sketch_np * 255).astype(np.uint8))
        visible = self.to_tensor(Image.fromarray(rgb))
        sketch = self.to_tensor(sketch_img)
        return {
            "visible": visible,
            "sketch": sketch,
            "attr": torch.from_numpy(attrs),
            "name": fn,
        }


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    root = resolve_celeba_root(cfg)
    max_key = {"train": "max_train", "val": "max_val", "test": "max_test"}[split]
    ds = CelebAAttrMultimodal(
        root,
        split=split if split != "test" else "test",
        image_size=int(cfg["model"].get("image_size", 64)),
        attr_names=cfg["data"].get("attrs"),
        max_n=cfg["data"].get(max_key),
        seed=int(cfg["train"].get("seed", 42)),
    )
    bs = int(cfg["train"]["batch_size"] if split == "train" else cfg.get("eval", {}).get("batch_size", 32))
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split == "train"),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=(split == "train"),
    )
