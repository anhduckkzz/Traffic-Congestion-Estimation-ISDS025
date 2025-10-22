from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path


def load_split_file(file_path: str | Path) -> List[Path]:
    split_path = Path(file_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    return [Path(entry) for entry in entries]


def collect_directory_samples(
    root: str | Path,
    image_dir: str,
    mask_dir: str,
    split_subdir: str,
    image_suffix: str,
    mask_suffix: str,
) -> List[Tuple[Path, Path]]:
    root = Path(root)
    image_root = root / image_dir / split_subdir
    mask_root = root / mask_dir / split_subdir

    if not image_root.exists():
        raise FileNotFoundError(f"Image split directory not found: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask split directory not found: {mask_root}")

    pattern = f"*{image_suffix}" if image_suffix else "*"
    samples: List[Tuple[Path, Path]] = []
    for image_path in sorted(image_root.glob(pattern)):
        stem = image_path.stem
        mask_path = mask_root / f"{stem}{mask_suffix}"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        samples.append((image_path, mask_path))

    if not samples:
        raise RuntimeError(f"No samples found under {image_root}")
    return samples


class SegmentationDataset(Dataset):
    """Road marking segmentation dataset with optional palette conversion."""

    def __init__(
        self,
        root: str | Path,
        samples: Sequence[Path | Tuple[Path, Path]],
        image_dir: str = "images",
        mask_dir: str = "masks",
        image_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        augmentations: Optional[Mapping[str, object]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        palette_csv: Optional[str | Path] = None,
        include_class_ids: Optional[Sequence[int]] = None,
        background_class_id: int = 0,
        binary_mask: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.augmentations = dict(augmentations or {})
        self.binary_mask = binary_mask
        self.include_class_ids = set(include_class_ids or []) or None
        self.background_class_id = int(background_class_id)

        self.color_jitter = self._build_color_jitter()
        self.palette_lookup = self._load_palette(palette_csv) if palette_csv else None

        self.samples: List[SegmentationSample] = []
        for entry in samples:
            if isinstance(entry, tuple):
                image_path, mask_path = entry
            else:
                image_path = (self.image_dir / entry).with_suffix(image_suffix)
                mask_path = (self.mask_dir / entry).with_suffix(mask_suffix)
            image_path = Path(image_path)
            mask_path = Path(mask_path)
            if not image_path.exists() or not mask_path.exists():
                raise FileNotFoundError(f"Missing pair: {image_path}, {mask_path}")
            self.samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path))

    def _build_color_jitter(self) -> Optional[ColorJitter]:
        cfg = self.augmentations.get("color_jitter")
        if not cfg:
            return None
        if isinstance(cfg, Mapping):
            return ColorJitter(**cfg)
        if isinstance(cfg, (int, float)):
            value = float(cfg)
            return ColorJitter(brightness=value, contrast=value, saturation=value, hue=value * 0.1)
        if isinstance(cfg, bool):
            return ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) if cfg else None
        raise TypeError("color_jitter augmentation must be a mapping, bool, or numeric value")

    @staticmethod
    def _load_palette(csv_path: str | Path) -> Dict[int, Tuple[int, int, int]]:
        palette: Dict[int, Tuple[int, int, int]] = {}
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Palette CSV not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                try:
                    class_id = int(row[0])
                    r, g, b = (int(row[-3]), int(row[-2]), int(row[-1]))
                    palette[class_id] = (r, g, b)
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid palette row: {row}") from exc
        return palette

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask_image = Image.open(sample.mask_path)

        image_tensor = F.to_tensor(image)
        mask_array = self._prepare_mask(mask_image)

        if self.augmentations:
            image_tensor, mask_array = self._apply_augmentations(image_tensor, mask_array)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return {
            "image": image_tensor,
            "mask": mask_array,
            "meta": {
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
            },
        }

    def _prepare_mask(self, mask_image: Image.Image) -> torch.Tensor:
        if self.palette_lookup:
            mask_array = self._mask_from_palette(mask_image)
        else:
            if mask_image.mode != "L":
                mask_image = mask_image.convert("L")
            mask_array = np.array(mask_image, dtype=np.int32)

        if self.include_class_ids is not None:
            positive = np.isin(mask_array, list(self.include_class_ids))
            if self.binary_mask:
                mask_array = positive.astype(np.float32)
            else:
                mask_array = np.where(positive, mask_array, self.background_class_id).astype(np.int32)
        else:
            mask_array = mask_array.astype(np.float32 if self.binary_mask else np.int32)
            if self.binary_mask and mask_array.max() > 1:
                mask_array = (mask_array > 0).astype(np.float32)

        tensor = torch.from_numpy(mask_array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.float() if self.binary_mask else tensor

    def _mask_from_palette(self, mask_image: Image.Image) -> np.ndarray:
        array = np.array(mask_image.convert("RGB"), dtype=np.uint8)
        h, w, _ = array.shape
        flat = array.reshape(-1, 3)
        keys = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2].astype(np.uint32)
        unique_keys, inverse = np.unique(keys, return_inverse=True)

        palette_map: Dict[int, int] = {}
        for class_id, color in self.palette_lookup.items():
            color_key = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
            palette_map[color_key] = class_id

        class_ids = np.array(
            [palette_map.get(int(key), self.background_class_id) for key in unique_keys],
            dtype=np.int32,
        )
        mask_array = class_ids[inverse].reshape(h, w)
        return mask_array

    def _apply_augmentations(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augmentations.get("horizontal_flip", False) and torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        if self.augmentations.get("gaussian_noise", False):
            std = float(self.augmentations.get("noise_std", 0.01))
            noise = torch.randn_like(image) * std
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image, mask


def segmentation_collate(batch: Iterable[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    meta = [item["meta"] for item in batch]
    return {"image": images, "mask": masks, "meta": meta}
