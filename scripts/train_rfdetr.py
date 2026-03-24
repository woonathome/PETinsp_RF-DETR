#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

MODEL_CLASS_BY_SIZE = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

MODEL_CONFIG_CLASS_BY_SIZE = {
    "nano": "RFDETRNanoConfig",
    "small": "RFDETRSmallConfig",
    "medium": "RFDETRMediumConfig",
    "large": "RFDETRLargeConfig",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR on custom COCO dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
        help="COCO dataset root with train/valid/test split folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "rfdetr-medium",
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_CLASS_BY_SIZE.keys()),
        default="medium",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers. RTX3090에서는 8~12 권장.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from output_dir/checkpoint.pth if present.",
    )
    parser.add_argument(
        "--pretrain-weights",
        type=Path,
        default=None,
        help="Optional local checkpoint path to initialize from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility and epoch-wise augmentation randomness.",
    )
    parser.add_argument(
        "--disable-augment",
        action="store_true",
        help="Disable custom Albumentations training augmentations.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logger (epoch metrics are always printed to console).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable training progress bar.",
    )
    parser.add_argument(
        "--force-high-level-api",
        action="store_true",
        help="Force model.train(...) path instead of PTL custom API.",
    )
    return parser.parse_args()


def ensure_dataset_layout(dataset_dir: Path) -> None:
    expected = [
        dataset_dir / "train" / "_annotations.coco.json",
        dataset_dir / "valid" / "_annotations.coco.json",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        joined = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Dataset layout is incomplete. Missing files:\n"
            f"{joined}\n"
            "Run scripts/prepare_tiled_coco_dataset.py first."
        )


def summarize_split(dataset_dir: Path, split: str) -> str:
    ann_path = dataset_dir / split / "_annotations.coco.json"
    if not ann_path.exists():
        return f"{split}: not found"
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    num_images = len(payload.get("images", []))
    num_annotations = len(payload.get("annotations", []))
    num_categories = len(payload.get("categories", []))
    return (
        f"{split}: images={num_images}, annotations={num_annotations}, "
        f"categories={num_categories}"
    )


def load_coco_category_info(dataset_dir: Path) -> tuple[int, List[str]]:
    ann_path = dataset_dir / "train" / "_annotations.coco.json"
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if not categories:
        raise ValueError("No categories found in train/_annotations.coco.json")
    sorted_cats = sorted(categories, key=lambda c: int(c["id"]))
    class_names = [str(cat["name"]) for cat in sorted_cats]
    return len(sorted_cats), class_names


def build_requested_aug_config() -> Dict[str, Any]:
    """
    Requested policy:
    - HorizontalFlip p=0.2
    - RGBShift p=0.2 OR HSV(HueSaturationValue) p=0.2 OR ChannelShuffle p=0.2

    Implementation detail:
    - RF-DETR docs state OneOf container always fires, and child `p` is used as
      selection weight.
    - We add NoOp with weight 0.4, so each color transform is selected with
      exact relative probability 0.2, and 0.4 means no color transform.
    """
    return {
        "HorizontalFlip": {"p": 0.2},
        "OneOf": {
            "transforms": [
                {
                    "RGBShift": {
                        "r_shift_limit": 20,
                        "g_shift_limit": 20,
                        "b_shift_limit": 20,
                        "p": 0.2,
                    }
                },
                {
                    "HueSaturationValue": {
                        "hue_shift_limit": 10,
                        "sat_shift_limit": 20,
                        "val_shift_limit": 20,
                        "p": 0.2,
                    }
                },
                {"ChannelShuffle": {"p": 0.2}},
                {"NoOp": {"p": 0.4}},
            ],
        },
    }


def resolve_resume_path(output_dir: Path, do_resume: bool) -> str | None:
    if not do_resume:
        return None
    checkpoint_path = output_dir / "checkpoint.pth"
    if checkpoint_path.exists():
        return str(checkpoint_path.resolve())
    print(
        f"Warning: --resume was set but checkpoint not found: {checkpoint_path}. "
        "Starting from scratch."
    )
    return None


def run_high_level_train(
    rfdetr: Any,
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
    aug_config: Dict[str, Any] | None,
    class_count: int,
) -> None:
    class_name = MODEL_CLASS_BY_SIZE[args.model_size]
    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        raise AttributeError(
            f"Cannot find {class_name} in rfdetr package. "
            "Please update rfdetr to a newer version."
        )

    init_sig = inspect.signature(model_cls.__init__)
    init_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values()
    )
    init_kwargs = {}
    if "num_classes" in init_sig.parameters or init_has_var_kwargs:
        init_kwargs["num_classes"] = class_count
    if args.pretrain_weights is not None:
        resolved_ckpt = str(args.pretrain_weights.resolve())
        if "pretrain_weights" in init_sig.parameters or init_has_var_kwargs:
            init_kwargs["pretrain_weights"] = resolved_ckpt
        elif "weights" in init_sig.parameters:
            init_kwargs["weights"] = resolved_ckpt
        elif "checkpoint_path" in init_sig.parameters:
            init_kwargs["checkpoint_path"] = resolved_ckpt
        else:
            raise TypeError(
                "This rfdetr version does not expose a known checkpoint init argument. "
                "Please initialize with pretrained defaults or update script for your version."
            )
    model = model_cls(**init_kwargs)

    requested_train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "output_dir": str(output_dir),
        "eval_interval": 1,
        "log_per_class_metrics": True,
        "progress_bar": None if args.no_progress_bar else "tqdm",
        "num_workers": args.num_workers,
        "seed": args.seed,
        "tensorboard": args.tensorboard,
        "aug_config": {} if args.disable_augment else aug_config,
    }
    resume_path = resolve_resume_path(output_dir, args.resume)
    if resume_path:
        requested_train_kwargs["resume"] = resume_path

    train_sig = inspect.signature(model.train)
    train_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in train_sig.parameters.values()
    )
    if train_has_var_kwargs:
        train_kwargs = dict(requested_train_kwargs)
    else:
        supported_names = set(train_sig.parameters.keys())
        train_kwargs = {
            key: value
            for key, value in requested_train_kwargs.items()
            if key in supported_names
        }

    dropped = sorted(set(requested_train_kwargs.keys()) - set(train_kwargs.keys()))
    if dropped:
        print(
            "Warning: current rfdetr version does not support these train args and they were skipped: "
            + ", ".join(dropped)
        )

    print("Starting RF-DETR training (high-level API) with arguments:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")
    print(
        "Note: per-epoch metrics are produced by RF-DETR COCOEvalCallback "
        "(val/mAP_50_95, val/F1, etc.) when eval_interval=1."
    )
    model.train(**train_kwargs)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dataset_layout(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_count, class_names = load_coco_category_info(dataset_dir)
    aug_config = build_requested_aug_config()
    if args.disable_augment:
        print("Augmentation is disabled by --disable-augment")
    else:
        aug_path = output_dir / "augmentation_config.json"
        aug_path.write_text(
            json.dumps(aug_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved augmentation config: {aug_path}")

    print("Dataset summary")
    print(summarize_split(dataset_dir, "train"))
    print(summarize_split(dataset_dir, "valid"))
    print(summarize_split(dataset_dir, "test"))
    print(f"Detected class count: {class_count}")

    try:
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            "rfdetr package is not installed. Install dependencies first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    if args.force_high_level_api:
        run_high_level_train(
            rfdetr=rfdetr,
            args=args,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            aug_config=aug_config,
            class_count=class_count,
        )
        return

    # Prefer PTL custom API because it allows explicit epoch metric printing callbacks.
    try:
        from rfdetr.config import TrainConfig
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
    except Exception as exc:
        print(
            "Warning: custom training API import failed. Falling back to high-level model.train()."
        )
        print(f"Reason: {exc}")
        run_high_level_train(
            rfdetr=rfdetr,
            args=args,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            aug_config=aug_config,
            class_count=class_count,
        )
        return

    model_config_name = MODEL_CONFIG_CLASS_BY_SIZE[args.model_size]
    config_module = importlib.import_module("rfdetr.config")
    model_config_cls = getattr(config_module, model_config_name, None)
    if model_config_cls is None:
        print(
            f"Warning: {model_config_name} not found in rfdetr.config. "
            "Falling back to high-level model.train()."
        )
        run_high_level_train(
            rfdetr=rfdetr,
            args=args,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            aug_config=aug_config,
            class_count=class_count,
        )
        return

    model_config_kwargs = {"num_classes": class_count}
    if args.pretrain_weights is not None:
        model_config_kwargs["pretrain_weights"] = str(args.pretrain_weights.resolve())

    model_config_sig = inspect.signature(model_config_cls.__init__)
    model_config_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in model_config_sig.parameters.values()
    )
    if not model_config_has_var_kwargs:
        model_config_kwargs = {
            k: v
            for k, v in model_config_kwargs.items()
            if k in model_config_sig.parameters
        }
    model_config = model_config_cls(**model_config_kwargs)

    train_config_kwargs = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "eval_interval": 1,
        "log_per_class_metrics": True,
        "tensorboard": args.tensorboard,
        "progress_bar": None if args.no_progress_bar else "tqdm",
        "seed": args.seed,
        "class_names": class_names,
        "aug_config": {} if args.disable_augment else aug_config,
        "resume": resolve_resume_path(output_dir, args.resume),
    }
    train_config_sig = inspect.signature(TrainConfig.__init__)
    train_config_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in train_config_sig.parameters.values()
    )
    if not train_config_has_var_kwargs:
        train_config_kwargs = {
            k: v
            for k, v in train_config_kwargs.items()
            if k in train_config_sig.parameters
        }
    train_config = TrainConfig(**train_config_kwargs)

    # Callback imports: support both old/new lightning package names.
    try:
        from pytorch_lightning import Callback
    except Exception:
        from lightning.pytorch.callbacks import Callback  # type: ignore

    class EpochMetricsPrinter(Callback):
        def __init__(self) -> None:
            super().__init__()
            self._preferred = [
                "val/mAP_50_95",
                "val/mAP_50",
                "val/F1",
                "val/precision",
                "val/recall",
                "val/loss",
                "train/loss",
            ]

        @staticmethod
        def _to_float(v: Any) -> float | None:
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    return None
                return float(v.detach().cpu().item())
            if isinstance(v, (float, int)):
                return float(v)
            return None

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if getattr(trainer, "sanity_checking", False):
                return
            metrics = trainer.callback_metrics
            epoch = int(trainer.current_epoch) + 1
            chunks = [f"epoch={epoch:03d}"]
            for key in self._preferred:
                if key in metrics:
                    fv = self._to_float(metrics.get(key))
                    if fv is not None:
                        chunks.append(f"{key}={fv:.4f}")
            if len(chunks) > 1:
                print("[EpochMetrics] " + " | ".join(chunks))
            else:
                print(f"[EpochMetrics] epoch={epoch:03d} | no validation metrics found")

    class EpochAugmentationSeedCallback(Callback):
        """
        Ensures a different RNG seed per epoch so Albumentations random outcomes
        change across epochs.
        """

        def __init__(self, base_seed: int) -> None:
            super().__init__()
            self.base_seed = int(base_seed)

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            epoch = int(trainer.current_epoch)
            epoch_seed = self.base_seed + epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed % (2**32 - 1))
            torch.manual_seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(epoch_seed)
            print(f"[AugmentSeed] epoch={epoch + 1:03d} seed={epoch_seed}")

    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config)
    trainer.callbacks.extend(
        [
            EpochAugmentationSeedCallback(base_seed=args.seed),
            EpochMetricsPrinter(),
        ]
    )

    print("Starting RF-DETR training (custom PTL API) with arguments:")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  model_size: {args.model_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  grad_accum_steps: {args.grad_accum_steps}")
    print(f"  lr: {args.lr}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  eval_interval: 1")
    print(f"  log_per_class_metrics: True")
    print(f"  tensorboard: {args.tensorboard}")
    print(f"  progress_bar: {None if args.no_progress_bar else 'tqdm'}")
    print(f"  seed(base): {args.seed}")
    print(f"  augmentation: {'disabled' if args.disable_augment else 'enabled'}")

    trainer.fit(module, datamodule, ckpt_path=getattr(train_config, "resume", None) or None)


if __name__ == "__main__":
    main()
