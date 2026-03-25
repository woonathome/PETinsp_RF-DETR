#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate image files referenced by COCO annotations and optionally clean broken entries."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
        help="Dataset root that contains train/valid/test folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Which splits to validate.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove broken image entries from annotations (writes backup .bak first).",
    )
    parser.add_argument(
        "--warn-path-len",
        type=int,
        default=220,
        help="Warn if absolute path length is >= this value.",
    )
    return parser.parse_args()


def load_coco(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_image(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "file_not_found"
    try:
        with Image.open(path) as img:
            img.verify()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    total_bad = 0
    for split in args.splits:
        split_dir = dataset_dir / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"[{split}] annotation not found: {ann_path}")
            continue

        coco = load_coco(ann_path)
        images: List[Dict] = coco.get("images", [])
        annotations: List[Dict] = coco.get("annotations", [])

        bad_image_ids = set()
        bad_rows = []
        long_path_rows = []

        for img in images:
            image_id = img["id"]
            file_name = img["file_name"]
            image_path = split_dir / file_name

            if len(str(image_path)) >= args.warn_path_len:
                long_path_rows.append((image_id, file_name, str(image_path), len(str(image_path))))

            ok, reason = validate_image(image_path)
            if not ok:
                bad_image_ids.add(image_id)
                bad_rows.append((image_id, file_name, str(image_path), reason))

        print(
            f"[{split}] total_images={len(images)} bad_images={len(bad_rows)} "
            f"long_paths(>={args.warn_path_len})={len(long_path_rows)}"
        )

        if long_path_rows:
            print(f"[{split}] sample long paths:")
            for row in long_path_rows[:5]:
                print(f"  id={row[0]} len={row[3]} file={row[1]}")

        if bad_rows:
            bad_report_path = split_dir / "_bad_images_report.txt"
            with bad_report_path.open("w", encoding="utf-8") as f:
                for image_id, file_name, image_path, reason in bad_rows:
                    f.write(f"id={image_id}\tfile={file_name}\tpath={image_path}\treason={reason}\n")
            print(f"[{split}] bad report saved: {bad_report_path}")

        if args.clean and bad_image_ids:
            backup_path = ann_path.with_suffix(ann_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(ann_path, backup_path)
                print(f"[{split}] backup saved: {backup_path}")

            new_images = [img for img in images if img["id"] not in bad_image_ids]
            new_annotations = [ann for ann in annotations if ann["image_id"] not in bad_image_ids]
            coco["images"] = new_images
            coco["annotations"] = new_annotations
            ann_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{split}] cleaned annotation written: images {len(images)}->{len(new_images)}, "
                f"annotations {len(annotations)}->{len(new_annotations)}"
            )

        total_bad += len(bad_rows)

    print(f"Done. total_bad_images={total_bad}")


if __name__ == "__main__":
    main()
