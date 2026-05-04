#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


MODEL_CLASS_BY_SIZE = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (231, 111, 81),
    (102, 45, 145),
    (0, 128, 255),
    (255, 99, 71),
    (46, 204, 113),
    (241, 196, 15),
]

CLASS_COLOR_BY_NAME = {
    "airbubble": (245, 235, 0),          # yellow
    "blackspot": (22, 219, 189),         # mint/teal
    "color-distribution": (220, 0, 220), # magenta
    "dust": (255, 128, 0),               # orange
    "gasbubble": (255, 0, 96),           # pink-red
    "pockmark": (122, 44, 230),          # purple
    "scratch": (173, 235, 0),            # lime
    "unknown": (0, 170, 220),            # sky blue
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw COCO GT bboxes and/or RF-DETR prediction bboxes for a split."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
        help="COCO dataset root that contains train/valid/test folders.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="test",
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "vis" / "test_vis",
        help="Root directory to save visualizations.",
    )
    parser.add_argument(
        "--mode",
        choices=["gt", "pred", "both"],
        default="gt",
        help="Visualization mode.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Maximum number of images to visualize (use <=0 for all).",
    )
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip images that have no GT annotations (applies when GT is drawn).",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_CLASS_BY_SIZE.keys()),
        default="medium",
        help="RF-DETR model size for prediction mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (.pth/.ckpt). If omitted, uses --run-dir/checkpoint_best_total.pth.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs") / "rfdetr-medium-7cls",
        help="Training run directory that contains checkpoint_best_total.pth.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Prediction confidence threshold.",
    )
    parser.add_argument(
        "--class-threshold-json",
        type=Path,
        default=None,
        help=(
            "Optional path to pr_curves_<split>.json for class-wise best thresholds. "
            "If omitted, auto-searches common runs/pr_auc_eval paths."
        ),
    )
    parser.add_argument(
        "--skip-gt-only-classes",
        nargs="+",
        default=None,
        help=(
            "Skip images when all GT boxes belong to these class names "
            "(case-insensitive, space/comma separated). "
            "Example: --skip-gt-only-classes unknown pockmark_unstable"
        ),
    )
    return parser.parse_args()


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    return split_dir / p


def parse_class_tokens(raw_values: List[str] | None) -> set[str]:
    if not raw_values:
        return set()
    out = set()
    for raw in raw_values:
        for token in str(raw).split(","):
            name = token.strip().lower()
            if name:
                out.add(name)
    return out


def _normalize_class_key(name: Any) -> str:
    return str(name).strip().lower()


def _parse_threshold_value(value: Any) -> float | None:
    try:
        thr = float(value)
    except Exception:
        return None
    if not math.isfinite(thr):
        return None
    if thr < 0.0 or thr > 1.0:
        return None
    return thr


def find_pr_curve_json_path(
    run_dir: Path,
    split: str,
    model_size: str,
    preferred_json: Path | None = None,
) -> Path | None:
    run_dir = run_dir.resolve()
    file_names = [f"pr_curves_{split}.json"]
    if str(split).lower() != "test":
        file_names.append("pr_curves_test.json")

    candidate_subdirs = [run_dir.name, model_size]
    if model_size == "medium":
        candidate_subdirs.append("medium-v2")

    raw_candidates: List[Path] = []
    if preferred_json is not None:
        raw_candidates.append(preferred_json)

    for file_name in file_names:
        raw_candidates.append(run_dir / file_name)
        for subdir in candidate_subdirs:
            raw_candidates.extend(
                [
                    run_dir.parent / "pr_auc_eval" / subdir / file_name,
                    Path.cwd() / "runs" / "pr_auc_eval" / subdir / file_name,
                    Path("runs") / "pr_auc_eval" / subdir / file_name,
                ]
            )

    seen: set[str] = set()
    for cand in raw_candidates:
        resolved = cand.expanduser().resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def load_classwise_best_thresholds(
    run_dir: Path,
    split: str,
    model_size: str,
    preferred_json: Path | None = None,
) -> tuple[Dict[str, float], Path | None]:
    json_path = find_pr_curve_json_path(
        run_dir=run_dir,
        split=split,
        model_size=model_size,
        preferred_json=preferred_json,
    )
    if json_path is None:
        return {}, None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read PR-curve json: {json_path} ({exc})")
        return {}, json_path

    class_to_thr: Dict[str, float] = {}

    summary = payload.get("summary", [])
    if isinstance(summary, list):
        for row in summary:
            if not isinstance(row, dict):
                continue
            class_name = row.get("class_name", None)
            if class_name is None:
                continue
            thr = _parse_threshold_value(row.get("best_threshold_by_f1", None))
            if thr is None:
                continue
            class_to_thr[_normalize_class_key(class_name)] = float(thr)

    curves = payload.get("curves", {})
    if isinstance(curves, dict):
        for class_name, curve in curves.items():
            key = _normalize_class_key(class_name)
            if key in class_to_thr:
                continue
            if not isinstance(curve, dict):
                continue
            thr = _parse_threshold_value(curve.get("best_threshold", None))
            if thr is None:
                continue
            class_to_thr[key] = float(thr)

    return class_to_thr, json_path


def threshold_for_class(
    class_name: str,
    default_threshold: float,
    classwise_thresholds: Dict[str, float] | None,
) -> float:
    if not classwise_thresholds:
        return float(default_threshold)
    return float(classwise_thresholds.get(_normalize_class_key(class_name), float(default_threshold)))


def color_for_category(category_id: int) -> Tuple[int, int, int]:
    return PALETTE[(int(category_id) - 1) % len(PALETTE)]


def color_for_class_name(class_name: str, category_id: int | None = None) -> Tuple[int, int, int]:
    key = str(class_name).strip().lower()
    if key in CLASS_COLOR_BY_NAME:
        return CLASS_COLOR_BY_NAME[key]
    if category_id is not None:
        return color_for_category(category_id)
    return (255, 255, 255)


def draw_box_with_label(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=max(1, line_width))
    text_pos = (x1 + 2, max(0.0, y1 - 12))
    draw.text(text_pos, label, fill=color, font=font)


def resolve_model_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        ckpt = args.checkpoint.resolve()
    else:
        ckpt = (args.run_dir.resolve() / "checkpoint_best_total.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def load_predictor(args: argparse.Namespace):
    try:
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            "rfdetr package is not installed. Install dependencies first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    class_name = MODEL_CLASS_BY_SIZE[args.model_size]
    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        raise AttributeError(
            f"Cannot find {class_name} in rfdetr package. "
            "Please update rfdetr to a newer version."
        )

    ckpt = resolve_model_checkpoint(args)
    init_sig = inspect.signature(model_cls.__init__)
    init_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values()
    )
    init_kwargs = {}
    resolved_ckpt = str(ckpt)
    if "pretrain_weights" in init_sig.parameters or init_has_var_kwargs:
        init_kwargs["pretrain_weights"] = resolved_ckpt
    elif "weights" in init_sig.parameters:
        init_kwargs["weights"] = resolved_ckpt
    elif "checkpoint_path" in init_sig.parameters:
        init_kwargs["checkpoint_path"] = resolved_ckpt
    else:
        raise TypeError(
            "This rfdetr version does not expose a known checkpoint init argument. "
            "Please update the script for your rfdetr version."
        )

    print(f"Loading prediction model from: {ckpt}")
    model = model_cls(**init_kwargs)
    return model


def load_run_class_names(run_dir: Path, checkpoint_path: Path) -> List[str] | None:
    meta_path = run_dir / "class_selection.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            selected = payload.get("selected_class_names", [])
            if isinstance(selected, list) and selected:
                return [str(x) for x in selected]
        except Exception:
            pass

    try:
        import torch

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        args_obj = ckpt.get("args", ckpt.get("hyper_parameters", {}))
        if isinstance(args_obj, dict):
            names = args_obj.get("class_names", None)
            if isinstance(names, list) and names:
                return [str(x) for x in names]
    except Exception:
        pass

    return None


def extract_predictions(detections: Any) -> List[Tuple[int, float, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float, float]] = []
    if detections is None:
        return rows

    xyxy = getattr(detections, "xyxy", None)
    class_id = getattr(detections, "class_id", None)
    confidence = getattr(detections, "confidence", None)
    if xyxy is not None:
        total = len(xyxy)
        for i in range(total):
            box = xyxy[i]
            x1, y1, x2, y2 = [float(v) for v in box]
            cid = int(class_id[i]) if class_id is not None else -1
            conf = float(confidence[i]) if confidence is not None else -1.0
            rows.append((cid, conf, x1, y1, x2, y2))
        return rows

    if isinstance(detections, list):
        for d in detections:
            if not isinstance(d, dict):
                continue
            box = d.get("xyxy", d.get("bbox"))
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            cid = int(d.get("class_id", d.get("class", -1)))
            conf = float(d.get("confidence", d.get("score", -1.0)))
            rows.append((cid, conf, x1, y1, x2, y2))
    return rows


def map_pred_class_name(
    pred_class_id: int,
    class_names: List[str],
) -> Tuple[str, int | None]:
    # RF-DETR predict() class_id is treated as 0-based index.
    idx = pred_class_id
    if 0 <= idx < len(class_names):
        return class_names[idx], idx
    return f"class_{pred_class_id}", None


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    ann_path = split_dir / "_annotations.coco.json"
    output_dir = args.output_dir.resolve()

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_gt = args.mode in {"gt", "both"}
    mode_pred = args.mode in {"pred", "both"}
    skip_gt_only_set = parse_class_tokens(args.skip_gt_only_classes)
    gt_dir = output_dir / "gt" if mode_gt else None
    pred_dir = output_dir / "pred" if mode_pred else None
    if gt_dir is not None:
        gt_dir.mkdir(parents=True, exist_ok=True)
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])

    id_to_name: Dict[int, str] = {int(c["id"]): str(c["name"]) for c in categories}
    ordered_dataset_class_names = [
        str(c["name"]) for c in sorted(categories, key=lambda c: int(c["id"]))
    ]
    ann_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in annotations:
        ann_by_image[int(ann["image_id"])].append(ann)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    predictor = load_predictor(args) if mode_pred else None
    pred_class_names = ordered_dataset_class_names
    checkpoint_path = None
    classwise_thresholds: Dict[str, float] = {}
    class_threshold_json_used: Path | None = None
    threshold_mode = "global"
    if mode_pred:
        checkpoint_path = resolve_model_checkpoint(args)
        run_class_names = load_run_class_names(args.run_dir.resolve(), checkpoint_path)
        if run_class_names:
            pred_class_names = run_class_names
            print(f"Using class order from run/checkpoint: {pred_class_names}")
        else:
            print(f"Using class order from dataset categories: {pred_class_names}")

        classwise_thresholds, class_threshold_json_used = load_classwise_best_thresholds(
            run_dir=args.run_dir.resolve(),
            split=args.split,
            model_size=args.model_size,
            preferred_json=args.class_threshold_json,
        )
        if class_threshold_json_used is not None and classwise_thresholds:
            covered = sorted(
                [name for name in pred_class_names if name.strip().lower() in classwise_thresholds]
            )
            print(f"Using class-wise thresholds from: {class_threshold_json_used}")
            print(f"Class-wise threshold coverage: {len(covered)}/{len(pred_class_names)}")
            threshold_mode = "classwise_best_f1"
        elif class_threshold_json_used is not None:
            print(
                "PR-curve json was found, but no usable class thresholds were parsed: "
                f"{class_threshold_json_used}"
            )
        else:
            print(
                "Class-wise threshold json not found. "
                f"Using global threshold={float(args.threshold):.4f}"
            )

    max_images = args.max_images
    limit = len(images) if max_images <= 0 else min(len(images), max_images)

    saved_gt = 0
    saved_pred = 0
    skipped = 0
    skipped_gt_only = 0
    for img in images[:limit]:
        image_id = int(img["id"])
        file_name = str(img["file_name"])
        image_path = resolve_image_path(split_dir, file_name)
        anns = ann_by_image.get(image_id, [])
        if skip_gt_only_set and anns:
            gt_names = [
                id_to_name.get(int(ann.get("category_id", -1)), f"class_{ann.get('category_id', -1)}").lower()
                for ann in anns
            ]
            if gt_names and all(name in skip_gt_only_set for name in gt_names):
                skipped_gt_only += 1
                continue
        if args.skip_empty and not anns:
            skipped += 1
            continue
        if not image_path.exists():
            print(f"[WARN] missing image: {image_path}")
            skipped += 1
            continue

        with Image.open(image_path) as im:
            base_image = im.convert("RGB")

        stem = Path(file_name).stem
        if mode_gt:
            canvas_gt = base_image.copy()
            draw_gt = ImageDraw.Draw(canvas_gt)
            for ann in anns:
                cat_id = int(ann["category_id"])
                cat_name = id_to_name.get(cat_id, f"class_{cat_id}")
                x, y, w, h = [float(v) for v in ann["bbox"]]
                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = max(x1 + 1.0, x + w)
                y2 = max(y1 + 1.0, y + h)
                draw_box_with_label(
                    draw=draw_gt,
                    box=(x1, y1, x2, y2),
                    label=cat_name,
                    color=color_for_class_name(cat_name, category_id=cat_id),
                    line_width=args.line_width,
                    font=font,
                )
            out_gt = gt_dir / f"{stem}_gt.jpg"  # type: ignore[arg-type]
            canvas_gt.save(out_gt, quality=95)
            saved_gt += 1

        if mode_pred and predictor is not None:
            canvas_pred = base_image.copy()
            draw_pred = ImageDraw.Draw(canvas_pred)
            infer_threshold = min(float(args.threshold), 0.001) if classwise_thresholds else float(args.threshold)
            detections = predictor.predict(base_image, threshold=infer_threshold)
            raw_pred_rows = extract_predictions(detections)
            pred_rows = []
            for cid, conf, x1, y1, x2, y2 in raw_pred_rows:
                if 0 <= cid < len(pred_class_names):
                    cname = pred_class_names[cid]
                    thr = threshold_for_class(
                        class_name=cname,
                        default_threshold=float(args.threshold),
                        classwise_thresholds=classwise_thresholds,
                    )
                else:
                    thr = float(args.threshold)
                if float(conf) >= float(thr):
                    pred_rows.append((cid, conf, x1, y1, x2, y2))
            for cid, conf, x1, y1, x2, y2 in pred_rows:
                name, class_idx = map_pred_class_name(
                    pred_class_id=cid,
                    class_names=pred_class_names,
                )
                label = f"{name} {conf:.2f}" if conf >= 0 else name
                color_id = (class_idx + 1) if class_idx is not None else (cid + 1)
                draw_box_with_label(
                    draw=draw_pred,
                    box=(x1, y1, x2, y2),
                    label=label,
                    color=color_for_class_name(name, category_id=color_id),
                    line_width=args.line_width,
                    font=font,
                )
            out_pred = pred_dir / f"{stem}_pred.jpg"  # type: ignore[arg-type]
            canvas_pred.save(out_pred, quality=95)
            saved_pred += 1

    print(
        f"Done. split={args.split} mode={args.mode} "
        f"saved_gt={saved_gt} saved_pred={saved_pred} "
        f"skipped={skipped} skipped_gt_only={skipped_gt_only} output_dir={output_dir}"
    )
    if mode_pred:
        if threshold_mode == "classwise_best_f1":
            print(
                f"Threshold mode: {threshold_mode} "
                f"(json={class_threshold_json_used})"
            )
        else:
            print(f"Threshold mode: {threshold_mode} (global={float(args.threshold):.4f})")


if __name__ == "__main__":
    main()
