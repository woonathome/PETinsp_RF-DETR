#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

MODEL_CLASS_BY_SIZE = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build confusion matrix on tiled COCO split using RF-DETR predictions.\n"
            "Detection matching uses greedy IoU matching (one GT <-> one prediction)."
        )
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
        help="COCO dataset root that contains train/valid/test folders.",
    )
    p.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="test",
        help="Split to evaluate.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs") / "rfdetr-medium-7cls",
        help="Training run directory containing checkpoint_best_total.pth.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. If omitted, uses --run-dir/checkpoint_best_total.pth",
    )
    p.add_argument(
        "--model-size",
        choices=sorted(MODEL_CLASS_BY_SIZE.keys()),
        default="medium",
        help="RF-DETR model size for prediction.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Prediction confidence threshold.",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for GT/pred matching.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to process (<=0 means all).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "confusion" / "test",
        help="Directory to save confusion matrix csv/json/png.",
    )
    p.add_argument(
        "--skip-gt-only-classes",
        nargs="+",
        default=None,
        help=(
            "Skip images when all GT boxes belong to these classes "
            "(case-insensitive, space/comma separated). "
            "Example: --skip-gt-only-classes unknown pockmark_unstable"
        ),
    )
    return p.parse_args()


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


def resolve_model_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        ckpt = args.checkpoint.resolve()
    else:
        ckpt = args.run_dir.resolve() / "checkpoint_best_total.pth"
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
    return model_cls(**init_kwargs)


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


def bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_thr: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou = bbox_iou_xyxy(g, p)
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def save_matrix_csv(path: Path, matrix: np.ndarray, labels: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i, name in enumerate(labels):
            writer.writerow([name] + [int(x) for x in matrix[i].tolist()])


def plot_heatmap(path: Path, matrix: np.ndarray, labels: List[str], title: str, normalize: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to save confusion matrix heatmap.\n"
            "Install with: python -m pip install matplotlib"
        ) from exc

    mat = matrix.astype(np.float64)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    fig_w = max(10, 1.2 * len(labels))
    fig_h = max(8, 1.0 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            txt = f"{val:.3f}" if normalize else str(int(val))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    ann_path = split_dir / "_annotations.coco.json"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not categories:
        raise ValueError("No categories found in annotation.")

    categories_sorted = sorted(categories, key=lambda c: int(c["id"]))
    eval_class_names = [str(c["name"]) for c in categories_sorted]
    category_id_to_eval_idx = {int(c["id"]): i for i, c in enumerate(categories_sorted)}
    category_id_to_name = {int(c["id"]): str(c["name"]) for c in categories_sorted}
    eval_name_to_idx = {n: i for i, n in enumerate(eval_class_names)}
    skip_gt_only_set = parse_class_tokens(args.skip_gt_only_classes)

    ann_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        ann_by_image[int(ann["image_id"])].append(ann)

    predictor = load_predictor(args)
    checkpoint_path = resolve_model_checkpoint(args)
    pred_class_names = load_run_class_names(args.run_dir.resolve(), checkpoint_path)
    if pred_class_names:
        print(f"Using class order from run/checkpoint: {pred_class_names}")
    else:
        pred_class_names = eval_class_names
        print(f"Using class order from dataset categories: {pred_class_names}")

    num_classes = len(eval_class_names)
    bg_idx = num_classes
    labels = eval_class_names + ["background"]
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    matched_only = np.zeros((num_classes, num_classes), dtype=np.int64)

    skipped_missing_image = 0
    skipped_unmapped_pred = 0
    skipped_gt_only = 0
    processed_images = 0

    max_images = args.max_images
    limit = len(images) if max_images <= 0 else min(len(images), max_images)
    for img in tqdm(images[:limit], desc=f"Evaluating {args.split}"):
        image_id = int(img["id"])
        image_path = resolve_image_path(split_dir, str(img["file_name"]))
        if not image_path.exists():
            skipped_missing_image += 1
            continue

        with Image.open(image_path) as im:
            pil_img = im.convert("RGB")

        gt_boxes: List[List[float]] = []
        gt_cls: List[int] = []
        gt_names_for_filter: List[str] = []
        for ann in ann_by_image.get(image_id, []):
            cat_id = int(ann["category_id"])
            gt_names_for_filter.append(category_id_to_name.get(cat_id, f"class_{cat_id}").lower())
            if cat_id not in category_id_to_eval_idx:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            if x2 <= x1 or y2 <= y1:
                continue
            gt_boxes.append([x1, y1, x2, y2])
            gt_cls.append(category_id_to_eval_idx[cat_id])

        if skip_gt_only_set and gt_names_for_filter:
            if all(name in skip_gt_only_set for name in gt_names_for_filter):
                skipped_gt_only += 1
                continue

        pred_rows = extract_predictions(predictor.predict(pil_img, threshold=args.threshold))
        pred_boxes: List[List[float]] = []
        pred_cls: List[int] = []
        for cid, conf, x1, y1, x2, y2 in pred_rows:
            _ = conf
            if not (0 <= cid < len(pred_class_names)):
                skipped_unmapped_pred += 1
                continue
            pred_name = pred_class_names[cid]
            if pred_name not in eval_name_to_idx:
                skipped_unmapped_pred += 1
                continue
            pred_boxes.append([x1, y1, x2, y2])
            pred_cls.append(eval_name_to_idx[pred_name])

        matches, unmatched_gt, unmatched_pred = greedy_match(
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            iou_thr=args.iou_threshold,
        )

        for gi, pi, _ in matches:
            t = gt_cls[gi]
            p = pred_cls[pi]
            cm[t, p] += 1
            matched_only[t, p] += 1
        for gi in unmatched_gt:
            cm[gt_cls[gi], bg_idx] += 1
        for pi in unmatched_pred:
            cm[bg_idx, pred_cls[pi]] += 1

        processed_images += 1

    # Save outputs
    raw_csv = output_dir / f"confusion_{args.split}_raw.csv"
    raw_png = output_dir / f"confusion_{args.split}_raw.png"
    norm_png = output_dir / f"confusion_{args.split}_row_norm.png"
    matched_csv = output_dir / f"confusion_{args.split}_matched_only_raw.csv"
    matched_png = output_dir / f"confusion_{args.split}_matched_only_raw.png"
    summary_json = output_dir / f"confusion_{args.split}_summary.json"

    save_matrix_csv(raw_csv, cm, labels)
    save_matrix_csv(matched_csv, matched_only, eval_class_names)
    plot_heatmap(
        raw_png,
        cm,
        labels,
        title=f"RF-DETR Confusion Matrix ({args.split}, IoU>={args.iou_threshold}, thr>={args.threshold})",
        normalize=False,
    )
    plot_heatmap(
        norm_png,
        cm,
        labels,
        title=f"RF-DETR Confusion Matrix Row-Norm ({args.split})",
        normalize=True,
    )
    plot_heatmap(
        matched_png,
        matched_only,
        eval_class_names,
        title=f"RF-DETR Matched-Only Class Confusion ({args.split})",
        normalize=False,
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "model_size": args.model_size,
        "threshold": float(args.threshold),
        "iou_threshold": float(args.iou_threshold),
        "processed_images": int(processed_images),
        "requested_images": int(limit),
        "skipped_missing_image": int(skipped_missing_image),
        "skipped_unmapped_prediction": int(skipped_unmapped_pred),
        "skipped_gt_only_classes": int(skipped_gt_only),
        "skip_gt_only_classes": sorted(skip_gt_only_set),
        "class_names": eval_class_names,
        "labels_with_background": labels,
        "raw_csv": str(raw_csv),
        "raw_png": str(raw_png),
        "row_norm_png": str(norm_png),
        "matched_only_csv": str(matched_csv),
        "matched_only_png": str(matched_png),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nConfusion matrix saved:")
    print(f"  raw_csv: {raw_csv}")
    print(f"  raw_png: {raw_png}")
    print(f"  row_norm_png: {norm_png}")
    print(f"  matched_only_csv: {matched_csv}")
    print(f"  matched_only_png: {matched_png}")
    print(f"  summary_json: {summary_json}")
    print(
        f"Processed images={processed_images}/{limit}, "
        f"missing_images={skipped_missing_image}, "
        f"unmapped_preds={skipped_unmapped_pred}, "
        f"skipped_gt_only={skipped_gt_only}"
    )


if __name__ == "__main__":
    main()
