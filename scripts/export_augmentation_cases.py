#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pick one random tiled image and export all augmentation combinations\n"
            "for PPT documentation.\n"
            "Combinations: flip(on/off) x color(noop/rgbshift/hsv/channelshuffle)."
        )
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
        help="Tiled COCO dataset root.",
    )
    p.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="train",
        help="Which split to sample from.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "augment_cases",
        help="Directory to save exported images.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection.",
    )
    p.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Optional explicit image path. If set, random selection is skipped.",
    )
    p.add_argument(
        "--save-original",
        action="store_true",
        help="Also save original image copy.",
    )
    return p.parse_args()


def list_split_images(split_dir: Path) -> List[Path]:
    images = []
    for p in sorted(split_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def save_rgb_image(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path, quality=95)


def build_color_transforms() -> Dict[str, A.BasicTransform]:
    return {
        "noop": A.NoOp(p=1.0),
        "rgbshift": A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20,
            p=1.0,
        ),
        "hsv": A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=1.0,
        ),
        "channelshuffle": A.ChannelShuffle(p=1.0),
    }


def apply_case(image: np.ndarray, use_flip: bool, color_tf: A.BasicTransform) -> np.ndarray:
    transforms = []
    if use_flip:
        transforms.append(A.HorizontalFlip(p=1.0))
    transforms.append(color_tf)
    comp = A.Compose(transforms)
    out = comp(image=image)
    return out["image"]


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path is not None:
        selected = args.image_path.resolve()
        if not selected.exists():
            raise FileNotFoundError(f"--image-path not found: {selected}")
    else:
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")
        images = list_split_images(split_dir)
        if not images:
            raise RuntimeError(f"No image files found in: {split_dir}")
        rnd = random.Random(args.seed)
        selected = rnd.choice(images).resolve()

    base = load_image_rgb(selected)
    stem = selected.stem

    if args.save_original:
        save_rgb_image(output_dir / f"{stem}__original.jpg", base)

    color_tfs = build_color_transforms()
    exported = []
    for use_flip in (False, True):
        for color_name, color_tf in color_tfs.items():
            aug = apply_case(base, use_flip=use_flip, color_tf=color_tf)
            out_name = f"{stem}__flip-{'on' if use_flip else 'off'}__color-{color_name}.jpg"
            out_path = output_dir / out_name
            save_rgb_image(out_path, aug)
            exported.append(str(out_path))

    meta = {
        "selected_image": str(selected),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "seed": args.seed,
        "cases": {
            "flip": ["off", "on"],
            "color": list(color_tfs.keys()),
            "total": len(exported),
        },
        "exported_files": exported,
    }
    meta_path = output_dir / "augmentation_cases_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Selected image: {selected}")
    print(f"Exported {len(exported)} augmented images to: {output_dir}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
