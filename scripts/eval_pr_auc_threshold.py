#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

MODEL_VERSION_TO_CLASS = {
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "medium-v2": "RFDETRMedium",
    "large": "RFDETRLarge",
}

MODEL_VERSION_TO_DEFAULT_RUN_DIR = {
    "small": Path("runs") / "rfdetr-small",
    "medium": Path("runs") / "rfdetr-medium",
    "medium-v2": Path("runs") / "rfdetr-medium-v2",
    "large": Path("runs") / "rfdetr-large",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate class-wise PR curve, PR-AUC, and best threshold on tiled COCO split.\n"
            "Supports RF-DETR small / medium / medium-v2 / large runs."
        )
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
        help="COCO dataset root containing train/valid/test folders.",
    )
    p.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="test",
        help="Split to evaluate.",
    )
    p.add_argument(
        "--model-version",
        choices=["small", "medium", "medium-v2", "large"],
        default="medium",
        help="Model run preset.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override run directory. Default is mapped from --model-version.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. If omitted, script searches best/latest checkpoint in run-dir.",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.05,
        help="IoU threshold for TP matching.",
    )
    p.add_argument(
        "--infer-threshold",
        type=float,
        default=0.001,
        help="Low predictor threshold to collect PR points.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max number of images to evaluate (<=0 means all).",
    )
    p.add_argument(
        "--exclude-classes",
        nargs="+",
        default=None,
        help="Class names to exclude (case-insensitive, space/comma separated).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "pr_auc_eval",
        help="Output directory for summary/plots.",
    )
    return p.parse_args()


def parse_class_tokens(raw_values: Sequence[str] | None) -> List[str]:
    if not raw_values:
        return []
    out: List[str] = []
    seen = set()
    for raw in raw_values:
        for tok in str(raw).split(","):
            name = tok.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
    return out


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    return split_dir / p


def find_checkpoint_in_run_dir(run_dir: Path) -> Path:
    preferred = [
        run_dir / "checkpoint_best_total.pth",
        run_dir / "checkpoint_best_regular.pth",
        run_dir / "checkpoint_best_ema.pth",
        run_dir / "checkpoint_last.ckpt",
        run_dir / "checkpoint_last.pth",
    ]
    for p in preferred:
        if p.exists():
            return p.resolve()

    best_like = sorted(
        [p for p in run_dir.glob("checkpoint_best*.pth") if p.is_file()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if best_like:
        return best_like[0].resolve()

    any_ckpt = sorted(
        [p for p in run_dir.glob("*.pth") if p.is_file()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if any_ckpt:
        return any_ckpt[0].resolve()
    raise FileNotFoundError(f"No checkpoint found under run dir: {run_dir}")


def load_predictor(model_version: str, checkpoint: Path):
    try:
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            "rfdetr package is not installed. Install dependencies first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    class_name = MODEL_VERSION_TO_CLASS[model_version]
    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        raise AttributeError(
            f"Cannot find {class_name} in rfdetr package. Please update rfdetr."
        )

    init_sig = inspect.signature(model_cls.__init__)
    init_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values()
    )
    init_kwargs = {}
    ckpt = str(checkpoint)
    if "pretrain_weights" in init_sig.parameters or init_has_var_kwargs:
        init_kwargs["pretrain_weights"] = ckpt
    elif "weights" in init_sig.parameters:
        init_kwargs["weights"] = ckpt
    elif "checkpoint_path" in init_sig.parameters:
        init_kwargs["checkpoint_path"] = ckpt
    else:
        raise TypeError("Unsupported rfdetr init signature for checkpoint loading.")

    print(f"Loading {class_name} from: {checkpoint}")
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
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
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
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_pr_for_class(
    gt_by_image: Dict[int, List[List[float]]],
    pred_list: List[Tuple[int, float, List[float]]],  # (image_id, score, box)
    iou_thr: float,
) -> Dict[str, Any]:
    n_gt = sum(len(v) for v in gt_by_image.values())
    if n_gt == 0:
        return {
            "num_gt": 0,
            "num_pred": len(pred_list),
            "thresholds": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "ap_auc": 0.0,
            "best_threshold": None,
            "best_f1": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
        }

    preds = sorted(pred_list, key=lambda x: x[1], reverse=True)
    matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_by_image.items()}
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)

    for i, (img_id, score, pbox) in enumerate(preds):
        _ = score
        gt_boxes = gt_by_image.get(img_id, [])
        if len(gt_boxes) == 0:
            fp[i] = 1.0
            continue

        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_boxes):
            if matched[img_id][j]:
                continue
            iou = bbox_iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1.0
            matched[img_id][best_j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / max(n_gt, 1)
    thresholds = np.array([p[1] for p in preds], dtype=np.float64)
    f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)

    # Interpolated AP (VOC/COCO-like envelope integration over recall axis)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap_auc = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))

    best_idx = int(np.argmax(f1)) if len(f1) > 0 else -1
    if best_idx >= 0:
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1[best_idx])
        best_precision = float(precision[best_idx])
        best_recall = float(recall[best_idx])
    else:
        best_threshold = None
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0

    return {
        "num_gt": int(n_gt),
        "num_pred": int(len(preds)),
        "thresholds": thresholds.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "ap_auc": ap_auc,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "best_precision": best_precision,
        "best_recall": best_recall,
    }


def save_curve_plot(
    out_path: Path,
    class_to_curve: Dict[str, Dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to save PR curve plots. "
            "Install with: python -m pip install matplotlib"
        ) from exc

    plt.figure(figsize=(11, 8))
    for cname, curve in class_to_curve.items():
        rec = np.array(curve["recall"], dtype=np.float64)
        pre = np.array(curve["precision"], dtype=np.float64)
        auc = curve["ap_auc"]
        if len(rec) == 0:
            continue
        plt.plot(rec, pre, linewidth=2, label=f"{cname} (AUC={auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Class-wise PR Curves")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=8, loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    run_dir = args.run_dir.resolve() if args.run_dir is not None else MODEL_VERSION_TO_DEFAULT_RUN_DIR[args.model_version].resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else find_checkpoint_in_run_dir(run_dir)

    output_dir = args.output_dir.resolve() / args.model_version
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = sorted(payload.get("categories", []), key=lambda c: int(c["id"]))
    if not categories:
        raise ValueError("No categories in annotation.")

    class_names = [str(c["name"]) for c in categories]
    cid_to_name = {int(c["id"]): str(c["name"]) for c in categories}
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    exclude_tokens = parse_class_tokens(args.exclude_classes)
    exclude_set = {t.lower() for t in exclude_tokens}
    selected_classes = [n for n in class_names if n.lower() not in exclude_set]
    if not selected_classes:
        raise ValueError("No classes left after --exclude-classes.")

    image_id_list = [int(im["id"]) for im in images]
    image_meta = {int(im["id"]): im for im in images}
    limit = len(image_id_list) if args.max_images <= 0 else min(len(image_id_list), args.max_images)
    image_id_list = image_id_list[:limit]
    image_id_set = set(image_id_list)

    gt_by_class: Dict[str, Dict[int, List[List[float]]]] = {
        cname: defaultdict(list) for cname in selected_classes
    }
    for ann in annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_id_set:
            continue
        cname = cid_to_name.get(int(ann["category_id"]), None)
        if cname is None or cname not in gt_by_class:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        x1, y1, x2, y2 = x, y, x + w, y + h
        if x2 <= x1 or y2 <= y1:
            continue
        gt_by_class[cname][image_id].append([x1, y1, x2, y2])

    predictor = load_predictor(args.model_version, checkpoint)
    run_class_names = load_run_class_names(run_dir, checkpoint)
    pred_class_names = run_class_names if run_class_names else class_names
    print(f"Using prediction class order: {pred_class_names}")

    pred_by_class: Dict[str, List[Tuple[int, float, List[float]]]] = {
        cname: [] for cname in selected_classes
    }

    missing_images = 0
    for image_id in tqdm(image_id_list, desc=f"Inference ({args.model_version}, {args.split})"):
        im = image_meta[image_id]
        image_path = resolve_image_path(split_dir, str(im["file_name"]))
        if not image_path.exists():
            missing_images += 1
            continue
        with Image.open(image_path) as pil:
            rgb = pil.convert("RGB")
        dets = extract_predictions(predictor.predict(rgb, threshold=args.infer_threshold))
        for cid, score, x1, y1, x2, y2 in dets:
            if not (0 <= cid < len(pred_class_names)):
                continue
            pred_name = pred_class_names[cid]
            if pred_name not in pred_by_class:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            pred_by_class[pred_name].append((image_id, float(score), [float(x1), float(y1), float(x2), float(y2)]))

    class_to_curve: Dict[str, Dict[str, Any]] = {}
    summary_rows = []
    for cname in selected_classes:
        curve = compute_pr_for_class(
            gt_by_image=gt_by_class[cname],
            pred_list=pred_by_class[cname],
            iou_thr=args.iou_threshold,
        )
        class_to_curve[cname] = curve
        summary_rows.append(
            {
                "class_name": cname,
                "num_gt": curve["num_gt"],
                "num_pred": curve["num_pred"],
                "pr_auc": curve["ap_auc"],
                "best_threshold_by_f1": curve["best_threshold"],
                "best_f1": curve["best_f1"],
                "best_precision": curve["best_precision"],
                "best_recall": curve["best_recall"],
            }
        )

    # Save outputs
    summary_csv = output_dir / f"pr_auc_summary_{args.split}.csv"
    curves_json = output_dir / f"pr_curves_{args.split}.json"
    plot_png = output_dir / f"pr_curves_{args.split}.png"

    import csv

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "num_gt",
                "num_pred",
                "pr_auc",
                "best_threshold_by_f1",
                "best_f1",
                "best_precision",
                "best_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    payload_out = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "model_version": args.model_version,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "iou_threshold": float(args.iou_threshold),
        "infer_threshold": float(args.infer_threshold),
        "max_images": int(args.max_images),
        "exclude_classes": exclude_tokens,
        "selected_classes": selected_classes,
        "missing_images": int(missing_images),
        "summary": summary_rows,
        "curves": class_to_curve,
    }
    curves_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")

    save_curve_plot(plot_png, class_to_curve)

    print("\nSaved outputs:")
    print(f"  summary_csv : {summary_csv}")
    print(f"  curves_json : {curves_json}")
    print(f"  plot_png    : {plot_png}")
    print("\nTop-level summary:")
    for row in summary_rows:
        print(
            f"  {row['class_name']}: AUC={row['pr_auc']:.4f}, "
            f"best_thr={row['best_threshold_by_f1']}, F1={row['best_f1']:.4f}, "
            f"P={row['best_precision']:.4f}, R={row['best_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
