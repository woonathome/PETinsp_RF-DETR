#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class SourceSample:
    index: int
    image_path: Path
    label_path: Path
    image_stem: str
    dominant_class: int
    class_hist: Dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare RF-DETR COCO dataset from YOLO labels: "
            "2046->2048 resize, 8x8 tiling, keep defect tiles, stratified split."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Dataset-v3.v1i.yolov5pytorch"),
    )
    parser.add_argument("--images-subdir", type=Path, default=Path("train/images"))
    parser.add_argument("--labels-subdir", type=Path, default=Path("train/labels"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
    )
    parser.add_argument("--resize-size", type=int, default=2048)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-box-area", type=float, default=1.0)
    parser.add_argument(
        "--split-strategy",
        choices=["dominant_class", "random"],
        default="dominant_class",
        help="dominant_class keeps train/valid/test ratio similar per dominant class.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--keep-empty-tiles",
        action="store_true",
        help="Keep tiles without boxes (default: defect tiles only).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory before generating.",
    )
    parser.add_argument(
        "--pockmark-class-id",
        type=int,
        default=None,
        help="YOLO class id for pockmark. If omitted, auto-detected from data.yaml names.",
    )
    parser.add_argument(
        "--pockmark-unstable-class-id",
        type=int,
        default=8,
        help="Class id used for low-contrast pockmark boxes (default: 8).",
    )
    parser.add_argument(
        "--pockmark-top-percent",
        type=float,
        default=0.10,
        help="Top contrast ratio to keep as pockmark (default: 0.10 = top 10%%).",
    )
    parser.add_argument(
        "--pockmark-border-px",
        type=int,
        default=2,
        help="Outer border thickness in pixels for contrast score (default: 2).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resize_size != args.grid_size * args.tile_size:
        raise ValueError(
            "--resize-size must be equal to grid_size * tile_size "
            f"(got {args.resize_size} vs {args.grid_size * args.tile_size})."
        )
    if not (0.0 <= args.val_ratio < 1.0 and 0.0 <= args.test_ratio < 1.0):
        raise ValueError("--val-ratio and --test-ratio must be in [0, 1).")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1.0.")
    if args.min_box_area < 0:
        raise ValueError("--min-box-area must be >= 0.")
    if not (0.0 < args.pockmark_top_percent <= 1.0):
        raise ValueError("--pockmark-top-percent must be in (0, 1].")
    if args.pockmark_border_px < 1:
        raise ValueError("--pockmark-border-px must be >= 1.")
    if args.pockmark_unstable_class_id < 0:
        raise ValueError("--pockmark-unstable-class-id must be >= 0.")


def prepare_output_dirs(output_root: Path, overwrite: bool) -> Dict[str, Path]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --overwrite to regenerate."
            )
        shutil.rmtree(output_root)

    split_dirs = {}
    for split in SPLITS:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = split_dir
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    return split_dirs


def parse_yolo_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        rows.append((cls, cx, cy, w, h))
    return rows


def find_dominant_class(rows: Sequence[Tuple[int, float, float, float, float]]) -> Tuple[int, Dict[int, int]]:
    if not rows:
        return -1, {}
    c = Counter(int(r[0]) for r in rows)
    dominant = sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return dominant, dict(c)


def load_names_from_data_yaml(source_root: Path) -> Dict[int, str]:
    yaml_path = source_root / "data.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    names = payload.get("names", [])
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def resolve_pockmark_class_id(
    names_from_yaml: Dict[int, str],
    discovered_ids: Sequence[int],
    requested_id: int | None,
) -> int:
    if requested_id is not None:
        return requested_id

    for cls_id, name in names_from_yaml.items():
        if str(name).strip().lower() == "pockmark":
            return int(cls_id)

    if 5 in discovered_ids:
        print("Warning: pockmark class name not found in data.yaml; falling back to class id 5.")
        return 5

    raise ValueError(
        "Unable to auto-detect pockmark class id. "
        "Please set --pockmark-class-id explicitly."
    )


def collect_samples(
    images_dir: Path,
    labels_dir: Path,
    max_images: int | None = None,
) -> Tuple[List[SourceSample], List[str]]:
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        image_paths = image_paths[: max_images]

    samples: List[SourceSample] = []
    missing_labels: List[str] = []
    for i, image_path in enumerate(image_paths):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue
        rows = parse_yolo_file(label_path)
        dominant_class, class_hist = find_dominant_class(rows)
        samples.append(
            SourceSample(
                index=i,
                image_path=image_path,
                label_path=label_path,
                image_stem=image_path.stem,
                dominant_class=dominant_class,
                class_hist=class_hist,
            )
        )
    return samples, missing_labels


def allocate_counts(n: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * val_ratio))
    if n >= 3 and test_ratio > 0 and n_test == 0:
        n_test = 1
    if n >= 3 and val_ratio > 0 and n_valid == 0:
        n_valid = 1
    if n_test + n_valid >= n:
        overflow = n_test + n_valid - n + 1
        n_valid = max(0, n_valid - overflow)
    n_train = n - n_valid - n_test
    return n_train, n_valid, n_test


def split_samples(
    samples: Sequence[SourceSample],
    split_strategy: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[int, str], Dict[str, Dict[str, int]]]:
    split_map: Dict[int, str] = {}
    strata_stats: Dict[str, Dict[str, int]] = {}

    if split_strategy == "random":
        shuffled = list(samples)
        random.Random(seed).shuffle(shuffled)
        n_train, n_valid, _ = allocate_counts(len(shuffled), val_ratio, test_ratio)
        for s in shuffled[:n_train]:
            split_map[s.index] = "train"
        for s in shuffled[n_train : n_train + n_valid]:
            split_map[s.index] = "valid"
        for s in shuffled[n_train + n_valid :]:
            split_map[s.index] = "test"
        strata_stats["all"] = {
            "total": len(shuffled),
            "train": n_train,
            "valid": n_valid,
            "test": len(shuffled) - n_train - n_valid,
        }
        return split_map, strata_stats

    by_stratum: Dict[int, List[SourceSample]] = defaultdict(list)
    for sample in samples:
        by_stratum[sample.dominant_class].append(sample)

    for stratum in sorted(by_stratum.keys()):
        group = by_stratum[stratum]
        rnd = random.Random(seed + (stratum + 101) * 97)
        rnd.shuffle(group)
        n_train, n_valid, _ = allocate_counts(len(group), val_ratio, test_ratio)
        for s in group[:n_train]:
            split_map[s.index] = "train"
        for s in group[n_train : n_train + n_valid]:
            split_map[s.index] = "valid"
        for s in group[n_train + n_valid :]:
            split_map[s.index] = "test"
        strata_stats[str(stratum)] = {
            "total": len(group),
            "train": n_train,
            "valid": n_valid,
            "test": len(group) - n_train - n_valid,
        }
    return split_map, strata_stats


def yolo_to_xyxy_resized(
    yolo_rows: Sequence[Tuple[int, float, float, float, float]],
    resized_w: int,
    resized_h: int,
) -> List[Tuple[int, int, float, float, float, float]]:
    out: List[Tuple[int, int, float, float, float, float]] = []
    for row_idx, (cls, cx, cy, bw, bh) in enumerate(yolo_rows):
        x1 = (cx - bw / 2.0) * resized_w
        y1 = (cy - bh / 2.0) * resized_h
        x2 = (cx + bw / 2.0) * resized_w
        y2 = (cy + bh / 2.0) * resized_h

        x1 = max(0.0, min(float(resized_w), x1))
        y1 = max(0.0, min(float(resized_h), y1))
        x2 = max(0.0, min(float(resized_w), x2))
        y2 = max(0.0, min(float(resized_h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((row_idx, cls, x1, y1, x2, y2))
    return out


def compute_box_contrast(
    gray_image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    border_px: int,
) -> float:
    height, width = gray_image.shape
    ix1 = max(0, min(width, int(math.floor(x1))))
    iy1 = max(0, min(height, int(math.floor(y1))))
    ix2 = max(0, min(width, int(math.ceil(x2))))
    iy2 = max(0, min(height, int(math.ceil(y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inner = gray_image[iy1:iy2, ix1:ix2]
    if inner.size == 0:
        return 0.0
    inner_mean = float(inner.mean())

    ox1 = max(0, ix1 - border_px)
    oy1 = max(0, iy1 - border_px)
    ox2 = min(width, ix2 + border_px)
    oy2 = min(height, iy2 + border_px)
    outer = gray_image[oy1:oy2, ox1:ox2]
    if outer.size == 0:
        return 0.0

    mask = np.ones(outer.shape, dtype=bool)
    in_x1 = ix1 - ox1
    in_y1 = iy1 - oy1
    in_x2 = in_x1 + (ix2 - ix1)
    in_y2 = in_y1 + (iy2 - iy1)
    mask[in_y1:in_y2, in_x1:in_x2] = False

    ring = outer[mask]
    if ring.size == 0:
        return 0.0
    ring_mean = float(ring.mean())
    return abs(inner_mean - ring_mean)


def collect_pockmark_keep_keys(
    samples: Sequence[SourceSample],
    resize_size: int,
    pockmark_class_id: int,
    top_percent: float,
    border_px: int,
) -> Tuple[Set[Tuple[int, int]], Dict[str, float | int | str]]:
    scored: List[Tuple[Tuple[int, int], float]] = []

    for sample in tqdm(samples, desc="Scoring pockmark contrast"):
        yolo_rows = parse_yolo_file(sample.label_path)
        if not yolo_rows:
            continue
        boxes = yolo_to_xyxy_resized(yolo_rows, resize_size, resize_size)
        pockmark_boxes = [b for b in boxes if b[1] == pockmark_class_id]
        if not pockmark_boxes:
            continue

        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
        if image.size != (resize_size, resize_size):
            image = image.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
        gray = np.asarray(image, dtype=np.float32).mean(axis=2)

        for box_row_idx, _, x1, y1, x2, y2 in pockmark_boxes:
            score = compute_box_contrast(gray, x1, y1, x2, y2, border_px=border_px)
            scored.append(((sample.index, box_row_idx), score))

    total = len(scored)
    if total == 0:
        return set(), {
            "status": "no_pockmark_boxes_found",
            "total_pockmark_boxes": 0,
            "keep_count": 0,
            "unstable_count": 0,
            "top_percent": top_percent,
            "contrast_threshold": 0.0,
            "border_px": border_px,
        }

    scored.sort(key=lambda x: x[1], reverse=True)
    keep_count = max(1, int(math.ceil(total * top_percent)))
    keep_items = scored[:keep_count]
    keep_keys = {k for k, _ in keep_items}
    threshold = float(keep_items[-1][1]) if keep_items else 0.0

    return keep_keys, {
        "status": "ok",
        "total_pockmark_boxes": total,
        "keep_count": keep_count,
        "unstable_count": total - keep_count,
        "top_percent": top_percent,
        "contrast_threshold": threshold,
        "border_px": border_px,
        "max_score": float(scored[0][1]),
        "min_score": float(scored[-1][1]),
        "mean_score": float(sum(v for _, v in scored) / total),
    }


def intersect_with_tile(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tx1: int,
    ty1: int,
    tx2: int,
    ty2: int,
) -> Tuple[float, float, float, float] | None:
    ix1 = max(x1, tx1)
    iy1 = max(y1, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1 - tx1, iy1 - ty1, ix2 - ix1, iy2 - iy1


def main() -> None:
    args = parse_args()
    validate_args(args)

    source_root = args.source_root.resolve()
    images_dir = (source_root / args.images_subdir).resolve()
    labels_dir = (source_root / args.labels_subdir).resolve()
    output_root = args.output_root.resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    split_dirs = prepare_output_dirs(output_root, overwrite=args.overwrite)
    samples, missing_labels = collect_samples(images_dir, labels_dir, max_images=args.max_images)
    if not samples:
        raise RuntimeError("No image/label pairs found in source dataset.")

    names_from_yaml = load_names_from_data_yaml(source_root)
    discovered_ids = sorted({c for s in samples for c in s.class_hist.keys()})
    if names_from_yaml:
        yolo_ids = sorted(names_from_yaml.keys())
    else:
        yolo_ids = discovered_ids

    pockmark_class_id = resolve_pockmark_class_id(
        names_from_yaml=names_from_yaml,
        discovered_ids=discovered_ids,
        requested_id=args.pockmark_class_id,
    )
    if pockmark_class_id not in yolo_ids:
        yolo_ids = sorted(set(yolo_ids + [pockmark_class_id]))
    if args.pockmark_unstable_class_id in discovered_ids:
        raise ValueError(
            "--pockmark-unstable-class-id already exists in source labels. "
            "Please choose a new class id."
        )
    yolo_ids = sorted(set(yolo_ids + [args.pockmark_unstable_class_id]))

    print(
        f"Pockmark relabel enabled: pockmark={pockmark_class_id}, "
        f"unstable={args.pockmark_unstable_class_id}, "
        f"top_percent={args.pockmark_top_percent:.2f}, border_px={args.pockmark_border_px}"
    )
    pockmark_keep_keys, pockmark_filter_stats = collect_pockmark_keep_keys(
        samples=samples,
        resize_size=args.resize_size,
        pockmark_class_id=pockmark_class_id,
        top_percent=args.pockmark_top_percent,
        border_px=args.pockmark_border_px,
    )
    print(
        "Pockmark contrast summary: "
        f"total={pockmark_filter_stats['total_pockmark_boxes']}, "
        f"keep={pockmark_filter_stats['keep_count']}, "
        f"unstable={pockmark_filter_stats['unstable_count']}, "
        f"threshold={pockmark_filter_stats['contrast_threshold']:.4f}"
    )

    yolo_to_coco = {yid: yid + 1 for yid in yolo_ids}
    class_name_map = {yid: names_from_yaml.get(yid, f"class_{yid}") for yid in yolo_ids}
    class_name_map[args.pockmark_unstable_class_id] = "pockmark_unstable"
    categories = [
        {"id": yolo_to_coco[yid], "name": class_name_map[yid], "supercategory": "defect"}
        for yid in yolo_ids
    ]

    split_map, strata_stats = split_samples(
        samples=samples,
        split_strategy=args.split_strategy,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    coco_by_split = {
        split: {
            "info": {
                "description": "RF-DETR tiled dataset (Model2)",
                "date_created": datetime.now(timezone.utc).isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        for split in SPLITS
    }
    image_id_counter = {split: 1 for split in SPLITS}
    ann_id_counter = {split: 1 for split in SPLITS}
    split_stats = {
        split: {
            "source_images": 0,
            "saved_tiles": 0,
            "saved_annotations": 0,
            "skipped_empty_tiles": 0,
            "pockmark_kept": 0,
            "pockmark_unstable": 0,
        }
        for split in SPLITS
    }
    manifest_rows: List[Dict[str, str | int]] = []

    for sample in tqdm(samples, desc="Processing source images"):
        split = split_map[sample.index]
        split_stats[split]["source_images"] += 1

        yolo_rows = parse_yolo_file(sample.label_path)
        if not yolo_rows and not args.keep_empty_tiles:
            continue

        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
        if image.size != (args.resize_size, args.resize_size):
            image = image.resize((args.resize_size, args.resize_size), Image.Resampling.BILINEAR)

        boxes = yolo_to_xyxy_resized(yolo_rows, args.resize_size, args.resize_size)

        for row in range(args.grid_size):
            for col in range(args.grid_size):
                tx1 = col * args.tile_size
                ty1 = row * args.tile_size
                tx2 = tx1 + args.tile_size
                ty2 = ty1 + args.tile_size

                tile_boxes = []
                for row_idx, ycls, x1, y1, x2, y2 in boxes:
                    mapped_cls = ycls
                    if ycls == pockmark_class_id:
                        key = (sample.index, row_idx)
                        if key not in pockmark_keep_keys:
                            mapped_cls = args.pockmark_unstable_class_id
                    inter = intersect_with_tile(x1, y1, x2, y2, tx1, ty1, tx2, ty2)
                    if inter is None:
                        continue
                    bx, by, bw, bh = inter
                    if bw * bh < args.min_box_area:
                        continue
                    tile_boxes.append((mapped_cls, bx, by, bw, bh))

                if not tile_boxes and not args.keep_empty_tiles:
                    split_stats[split]["skipped_empty_tiles"] += 1
                    continue

                tile = image.crop((tx1, ty1, tx2, ty2))
                tile_file_name = f"img{sample.index:05d}_r{row:02d}_c{col:02d}.jpg"
                tile_path = split_dirs[split] / tile_file_name
                tile.save(tile_path, format="JPEG", quality=95)

                image_id = image_id_counter[split]
                image_id_counter[split] += 1
                coco_by_split[split]["images"].append(
                    {
                        "id": image_id,
                        "file_name": tile_file_name,
                        "width": args.tile_size,
                        "height": args.tile_size,
                    }
                )

                for ycls, bx, by, bw, bh in tile_boxes:
                    if ycls not in yolo_to_coco:
                        continue
                    ann_id = ann_id_counter[split]
                    ann_id_counter[split] += 1
                    coco_by_split[split]["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": yolo_to_coco[ycls],
                            "bbox": [bx, by, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0,
                        }
                    )
                    if ycls == pockmark_class_id:
                        split_stats[split]["pockmark_kept"] += 1
                    if ycls == args.pockmark_unstable_class_id:
                        split_stats[split]["pockmark_unstable"] += 1

                split_stats[split]["saved_tiles"] += 1
                split_stats[split]["saved_annotations"] += len(tile_boxes)
                manifest_rows.append(
                    {
                        "split": split,
                        "tile_file_name": tile_file_name,
                        "source_image": sample.image_path.name,
                        "tile_row": row,
                        "tile_col": col,
                        "num_boxes": len(tile_boxes),
                        "dominant_class": sample.dominant_class,
                    }
                )

    for split in SPLITS:
        ann_path = split_dirs[split] / "_annotations.coco.json"
        ann_path.write_text(
            json.dumps(coco_by_split[split], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    metadata_dir = output_root / "metadata"
    summary = {
        "source_root": str(source_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_root": str(output_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "resize_size": args.resize_size,
            "grid_size": args.grid_size,
            "tile_size": args.tile_size,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "min_box_area": args.min_box_area,
            "split_strategy": args.split_strategy,
            "keep_empty_tiles": args.keep_empty_tiles,
            "pockmark_class_id": pockmark_class_id,
            "pockmark_unstable_class_id": args.pockmark_unstable_class_id,
            "pockmark_top_percent": args.pockmark_top_percent,
            "pockmark_border_px": args.pockmark_border_px,
        },
        "totals": {
            "source_images_with_labels": len(samples),
            "source_images_missing_labels": len(missing_labels),
            "split_source_images": {s: split_stats[s]["source_images"] for s in SPLITS},
            "split_saved_tiles": {s: split_stats[s]["saved_tiles"] for s in SPLITS},
            "split_saved_annotations": {s: split_stats[s]["saved_annotations"] for s in SPLITS},
            "split_skipped_empty_tiles": {s: split_stats[s]["skipped_empty_tiles"] for s in SPLITS},
            "split_pockmark_kept": {s: split_stats[s]["pockmark_kept"] for s in SPLITS},
            "split_pockmark_unstable": {s: split_stats[s]["pockmark_unstable"] for s in SPLITS},
            "strata_split_counts": strata_stats,
            "pockmark_filter": pockmark_filter_stats,
        },
    }
    (metadata_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "class_mapping.json").write_text(
        json.dumps(
            {
                "yolo_to_coco": {str(k): v for k, v in yolo_to_coco.items()},
                "class_names": {str(k): class_name_map[k] for k in yolo_ids},
                "pockmark_filter": pockmark_filter_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "missing_source_labels.txt").write_text(
        "\n".join(missing_labels),
        encoding="utf-8",
    )
    with (metadata_dir / "tile_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "tile_file_name",
                "source_image",
                "tile_row",
                "tile_col",
                "num_boxes",
                "dominant_class",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Preprocessing complete.")
    print(f"Output: {output_root}")
    print(f"Source images with labels: {len(samples)}")
    print(f"Source images missing labels: {len(missing_labels)}")
    for s in SPLITS:
        print(
            f"[{s}] source_images={split_stats[s]['source_images']} "
            f"tiles={split_stats[s]['saved_tiles']} "
            f"annotations={split_stats[s]['saved_annotations']} "
            f"pockmark={split_stats[s]['pockmark_kept']} "
            f"pockmark_unstable={split_stats[s]['pockmark_unstable']}"
        )


if __name__ == "__main__":
    main()
