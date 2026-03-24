# RF-DETR Fine-tuning Repo (Torch)

This repo trains RF-DETR for defect detection from your dataset:
- Original image size: `2046x2046`
- Resize to: `2048x2048`
- Tile strategy: `8x8` (`256x256`)
- Keep only defect tiles by default (tiles with at least one box)

## 1) Project Layout

```text
2603 Tester Model/
├─ dataset/
│  ├─ image/
│  └─ label/
├─ scripts/
│  ├─ prepare_tiled_coco_dataset.py
│  ├─ train_rfdetr.py
│  └─ predict_tile.py
├─ requirements.txt
└─ README.md
```

## 2) Environment (RTX3090)

Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

## 3) Preprocess Dataset (tiling + split)

Run from project root (`2603 Tester Model`):

```bash
python scripts/prepare_tiled_coco_dataset.py ^
  --dataset-root ./dataset ^
  --output-root ./data/rfdetr_tiled_coco ^
  --resize-size 2048 ^
  --grid-size 8 ^
  --tile-size 256 ^
  --val-ratio 0.15 ^
  --test-ratio 0.10 ^
  --seed 42 ^
  --overwrite
```

Output:

```text
data/rfdetr_tiled_coco/
├─ train/
│  ├─ *.jpg
│  └─ _annotations.coco.json
├─ valid/
│  ├─ *.jpg
│  └─ _annotations.coco.json
├─ test/
│  ├─ *.jpg
│  └─ _annotations.coco.json
└─ metadata/
   ├─ preprocess_summary.json
   ├─ class_mapping.json
   ├─ missing_labels.txt
   └─ tile_manifest.csv
```

## 4) Train RF-DETR (Epoch Metrics + Albumentations)

```bash
python scripts/train_rfdetr.py ^
  --dataset-dir ./data/rfdetr_tiled_coco ^
  --output-dir ./runs/rfdetr-medium ^
  --model-size medium ^
  --epochs 100 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --num-workers 8 ^
  --lr 1e-4 ^
  --tensorboard
```

### Added features in this trainer

1. Epoch-wise performance is printed in console:
- `val/mAP_50_95`
- `val/mAP_50`
- `val/F1`
- `val/precision`
- `val/recall`
- loss metrics when available

2. Albumentations policy is enabled by default:
- `HorizontalFlip(p=0.2)`
- color branch with OR behavior:
  - `RGBShift` (~0.2)
  - `HueSaturationValue` (~0.2)
  - `ChannelShuffle` (~0.2)

Implementation note:
- The OR branch is implemented with `OneOf` + `NoOp`.
- Weights are set to `0.2, 0.2, 0.2, 0.4` for `RGBShift`, `HueSaturationValue`, `ChannelShuffle`, `NoOp`.
- This matches your requested probabilities while keeping OR behavior.

3. Different augmentation randomness every epoch:
- Trainer includes an epoch callback that applies a different random seed each epoch.
- Seed log example: `[AugmentSeed] epoch=003 seed=44`

### Useful flags

- Disable augmentations:
```bash
python scripts/train_rfdetr.py --disable-augment ...
```

- Resume from `output_dir/checkpoint.pth`:
```bash
python scripts/train_rfdetr.py --resume ...
```

- Force high-level API fallback:
```bash
python scripts/train_rfdetr.py --force-high-level-api ...
```

## 5) Single-image Inference Test

```bash
python scripts/predict_tile.py ^
  --image-path ./data/rfdetr_tiled_coco/test/some_tile.jpg ^
  --model-size medium ^
  --checkpoint ./runs/rfdetr-medium/checkpoint_best_total.pth ^
  --threshold 0.3
```

## 6) Optional: Custom Class Names

Default names are generated from YOLO ids (`class_0`, `class_1`, ...).  
To override, create a text file with one class name per line:

```bash
python scripts/prepare_tiled_coco_dataset.py --class-names-file ./class_names.txt ...
```
