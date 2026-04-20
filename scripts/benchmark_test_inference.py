#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
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
        description="Report RF-DETR parameter count and avg inference time per tile on a COCO split."
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
        help="Which split to benchmark.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs") / "rfdetr-medium-7cls",
        help="Training run directory (used for default checkpoint path).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. If omitted, --run-dir/checkpoint_best_total.pth is used.",
    )
    p.add_argument(
        "--model-size",
        choices=sorted(MODEL_CLASS_BY_SIZE.keys()),
        default="medium",
    )
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to benchmark (<=0 means all).",
    )
    p.add_argument(
        "--warmup-images",
        type=int,
        default=20,
        help="Number of warmup images before timing.",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional JSON output path for benchmark summary.",
    )
    return p.parse_args()


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
    ckpt_str = str(ckpt)
    if "pretrain_weights" in init_sig.parameters or init_has_var_kwargs:
        init_kwargs["pretrain_weights"] = ckpt_str
    elif "weights" in init_sig.parameters:
        init_kwargs["weights"] = ckpt_str
    elif "checkpoint_path" in init_sig.parameters:
        init_kwargs["checkpoint_path"] = ckpt_str
    else:
        raise TypeError(
            "This rfdetr version does not expose a known checkpoint init argument."
        )
    print(f"Loading model from: {ckpt}")
    return model_cls(**init_kwargs), ckpt


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    return split_dir / p


def try_get_torch_module(obj: Any) -> torch.nn.Module | None:
    if isinstance(obj, torch.nn.Module):
        return obj
    for attr in ("model", "module", "_model", "net", "detr"):
        cand = getattr(obj, attr, None)
        if isinstance(cand, torch.nn.Module):
            return cand
    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            if isinstance(v, torch.nn.Module):
                return v
    return None


def _module_param_count(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def find_largest_module_in_object(root_obj: Any, max_depth: int = 5) -> tuple[torch.nn.Module | None, int, int]:
    """
    Some rfdetr versions wrap nn.Module deeply. Recursively traverse object graph
    (with cycle protection) and return the module with the largest parameter count.
    """
    visited = set()
    stack: List[Tuple[Any, int]] = [(root_obj, 0)]
    best_module: torch.nn.Module | None = None
    best_total = 0
    best_trainable = 0

    while stack:
        cur, depth = stack.pop()
        oid = id(cur)
        if oid in visited:
            continue
        visited.add(oid)

        if isinstance(cur, torch.nn.Module):
            total, trainable = _module_param_count(cur)
            if total > best_total:
                best_module = cur
                best_total = total
                best_trainable = trainable
            # No need to recurse through module internals here.
            continue

        if depth >= max_depth:
            continue

        children: List[Any] = []
        if isinstance(cur, dict):
            children.extend(cur.values())
        elif isinstance(cur, (list, tuple, set)):
            children.extend(list(cur))
        else:
            if hasattr(cur, "__dict__"):
                try:
                    children.extend(vars(cur).values())
                except Exception:
                    pass

        for ch in children:
            if ch is None:
                continue
            # avoid exploding traversal on trivial scalar types
            if isinstance(ch, (str, bytes, int, float, bool)):
                continue
            stack.append((ch, depth + 1))

    return best_module, best_total, best_trainable


def _safe_torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # for older torch versions without weights_only argument
        return torch.load(str(path), map_location="cpu")


def _count_tensors_in_mapping(mapping: Dict[Any, Any]) -> tuple[int, int]:
    total = 0
    n_tensors = 0
    for v in mapping.values():
        if torch.is_tensor(v):
            total += int(v.numel())
            n_tensors += 1
    return total, n_tensors


def count_parameters_from_checkpoint(ckpt_path: Path, max_depth: int = 5) -> Dict[str, int] | None:
    """
    Fallback path: parse checkpoint object and search for tensor mappings
    (state_dict/model/model_state_dict/ema_state_dict/etc.).
    """
    try:
        obj = _safe_torch_load(ckpt_path)
    except Exception:
        return None

    best_total = 0
    best_num_tensors = 0

    stack: List[Tuple[Any, int]] = [(obj, 0)]
    visited = set()
    preferred_keys = {"state_dict", "model", "model_state_dict", "ema_state_dict", "net"}
    while stack:
        cur, depth = stack.pop()
        oid = id(cur)
        if oid in visited:
            continue
        visited.add(oid)

        if isinstance(cur, dict):
            total, n_tensors = _count_tensors_in_mapping(cur)
            if total > best_total and n_tensors > 0:
                best_total = total
                best_num_tensors = n_tensors

            if depth < max_depth:
                # preferred keys first
                for k in preferred_keys:
                    if k in cur:
                        stack.append((cur[k], depth + 1))
                # then generic traversal
                for v in cur.values():
                    if isinstance(v, (dict, list, tuple)):
                        stack.append((v, depth + 1))
            continue

        if depth >= max_depth:
            continue
        if isinstance(cur, (list, tuple)):
            for v in cur:
                if isinstance(v, (dict, list, tuple)):
                    stack.append((v, depth + 1))

    if best_total <= 0:
        return None
    return {"total": int(best_total), "trainable": -1, "num_tensors": int(best_num_tensors)}


def count_parameters(model_obj: Any, checkpoint_path: Path) -> Dict[str, Any] | None:
    # 1) direct shallow check
    module = try_get_torch_module(model_obj)
    if module is not None:
        total, trainable = _module_param_count(module)
        return {"total": total, "trainable": trainable, "source": "predictor_shallow"}

    # 2) recursive object search
    module2, total2, trainable2 = find_largest_module_in_object(model_obj)
    if module2 is not None and total2 > 0:
        return {"total": total2, "trainable": trainable2, "source": "predictor_recursive"}

    # 3) checkpoint fallback
    ckpt_stats = count_parameters_from_checkpoint(checkpoint_path)
    if ckpt_stats is not None:
        ckpt_stats["source"] = "checkpoint_state"
        return ckpt_stats
    return None


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    arr = sorted(values)
    idx = (len(arr) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    frac = idx - lo
    return float(arr[lo] * (1 - frac) + arr[hi] * frac)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    predictor, checkpoint_path = load_predictor(args)
    param_stats = count_parameters(predictor, checkpoint_path)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    if not images:
        raise RuntimeError(f"No images in annotation: {ann_path}")

    limit = len(images) if args.max_images <= 0 else min(len(images), args.max_images)
    selected = images[:limit]
    warmup_n = max(0, min(args.warmup_images, limit))

    print(f"Benchmark split={args.split} images={limit} warmup={warmup_n}")
    if param_stats is not None:
        tr = int(param_stats.get("trainable", -1))
        tr_str = f"{tr:,}" if tr >= 0 else "N/A"
        print(
            f"Parameter count: total={int(param_stats['total']):,}, "
            f"trainable={tr_str}, source={param_stats.get('source', 'unknown')}"
        )
    else:
        print("Parameter count: unavailable (torch.nn.Module handle not found).")

    use_cuda_sync = torch.cuda.is_available()

    def infer_one(image_path: Path) -> float:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
        if use_cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predictor.predict(rgb, threshold=args.threshold)
        if use_cuda_sync:
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    # Warmup
    for img in tqdm(selected[:warmup_n], desc="Warmup"):
        p = resolve_image_path(split_dir, str(img["file_name"]))
        if not p.exists():
            continue
        _ = infer_one(p)

    # Timed run
    times_sec: List[float] = []
    missing = 0
    for img in tqdm(selected[warmup_n:], desc="Benchmark"):
        p = resolve_image_path(split_dir, str(img["file_name"]))
        if not p.exists():
            missing += 1
            continue
        dt = infer_one(p)
        times_sec.append(dt)

    if not times_sec:
        raise RuntimeError("No timed inference samples were collected.")

    mean_s = float(statistics.mean(times_sec))
    median_s = float(statistics.median(times_sec))
    p90_s = percentile(times_sec, 0.90)
    p95_s = percentile(times_sec, 0.95)
    fps = 1.0 / mean_s if mean_s > 0 else 0.0

    print("\nInference timing (per tile, model.predict only):")
    print(f"  mean   : {mean_s * 1000:.3f} ms")
    print(f"  median : {median_s * 1000:.3f} ms")
    print(f"  p90    : {p90_s * 1000:.3f} ms")
    print(f"  p95    : {p95_s * 1000:.3f} ms")
    print(f"  fps    : {fps:.2f}")
    print(f"  timed_samples={len(times_sec)}, missing_images={missing}")

    summary = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "model_size": args.model_size,
        "threshold": float(args.threshold),
        "max_images": int(args.max_images),
        "warmup_images": int(warmup_n),
        "timed_samples": int(len(times_sec)),
        "missing_images": int(missing),
        "parameter_count": param_stats,
        "timing_ms": {
            "mean": mean_s * 1000.0,
            "median": median_s * 1000.0,
            "p90": p90_s * 1000.0,
            "p95": p95_s * 1000.0,
        },
        "fps": fps,
    }

    if args.save_json is not None:
        save_path = args.save_json.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON: {save_path}")


if __name__ == "__main__":
    main()
