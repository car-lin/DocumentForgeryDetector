from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"real": 0, "edited": 1, "ai_generated": 2}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    # Stores the aligned file paths for one dataset sample:
    # the RGB image, its three forensic maps, and the class label.
    rgb: Path
    ela: Path
    srm: Path
    fft: Path
    label: int


class ForensicDataset(Dataset):
    def __init__(self, rgb_root: Path, ela_root: Path, srm_root: Path, fft_root: Path, image_size: int = 224):
        # Save dataset roots for all four modalities and build
        # the list of valid samples during initialization.
        self.rgb_root = Path(rgb_root)
        self.ela_root = Path(ela_root)
        self.srm_root = Path(srm_root)
        self.fft_root = Path(fft_root)
        self.image_size = image_size
        self.samples = self._index_samples()

    def _index_samples(self) -> List[Sample]:
        # Scans the RGB dataset class folders and keeps only those images
        # for which matching ELA, SRM, and FFT files also exist.
        # This ensures every returned sample is complete across all modalities.
        out: List[Sample] = []
        for class_name, label in LABEL_MAP.items():
            class_dir = self.rgb_root / class_name
            if not class_dir.exists():
                continue
            for p in class_dir.rglob("*"):
                if p.suffix.lower() not in IMG_EXTS:
                    continue

                rel = p.relative_to(self.rgb_root)
                ela = self.ela_root / rel.with_suffix(".png")
                srm = self.srm_root / rel.with_suffix(".png")
                fft = self.fft_root / rel.with_suffix(".png")

                if not (ela.exists() and srm.exists() and fft.exists()):
                    continue

                out.append(
                    Sample(
                        rgb=p,
                        ela=ela,
                        srm=srm,
                        fft=fft,
                        label=label,
                    )
                )
        return out

    def _load_rgb(self, path: Path) -> torch.Tensor:
        # Loads an RGB image, converts OpenCV's BGR format to RGB,
        # resizes it to the model input size, normalizes pixel values
        # to [0, 1], and converts it into a PyTorch tensor in CHW format.
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read RGB image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def _load_gray(self, path: Path) -> torch.Tensor:
        # Loads a single-channel forensic map (ELA / SRM / FFT),
        # resizes it, normalizes it to [0, 1], and adds a channel
        # dimension so it becomes shape [1, H, W].
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read forensic map: {path}")
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)

    def __len__(self) -> int:
        # Returns the total number of valid indexed samples.
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Loads one sample by index:
        # - RGB image
        # - ELA, SRM, FFT forensic maps
        # Then combines the three forensic maps into a 3-channel tensor.
        # It returns both:
        # - label3: 3-class label (real / edited / ai_generated)
        # - label2: binary label for stage 1 (real vs forged)
        sample = self.samples[idx]
        rgb = self._load_rgb(sample.rgb)
        ela = self._load_gray(sample.ela)
        srm = self._load_gray(sample.srm)
        fft = self._load_gray(sample.fft)
        forensic = torch.cat([ela, srm, fft], dim=0)
        label = torch.tensor(sample.label, dtype=torch.long)
        is_forged = torch.tensor(0 if sample.label == 0 else 1, dtype=torch.long)
        return {"rgb": rgb, "forensic": forensic, "label3": label, "label2": is_forged}