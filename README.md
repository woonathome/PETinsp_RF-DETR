# RF-DETR Pipeline for 2603 Tester Model2

This project builds a training dataset from:
- `Dataset-v3.v1i.yolov5pytorch/train/images`
- `Dataset-v3.v1i.yolov5pytorch/train/labels`

Workflow:
1. Resize full images from `2046x2046` to `2048x2048`
2. Tile to `8x8` (`256x256`)
3. For `pockmark`, compute contrast = `|mean(inner bbox) - mean(outer 2px ring)|`
4. Keep top 10% pockmark boxes as `pockmark`, relabel remaining as class `8` (`pockmark_unstable`)
5. Keep defect tiles only by default
6. Create COCO splits (`train/valid/test`)
7. Train RF-DETR

## 1) Install

```bash
python -m pip install -r requirements.txt
```

## 2) Preprocess and Split (same ratio policy)

```bash
python scripts/prepare_tiled_coco_dataset.py ^
  --source-root ./Dataset-v3.v1i.yolov5pytorch ^
  --images-subdir train/images ^
  --labels-subdir train/labels ^
  --output-root ./data/rfdetr_tiled_coco ^
  --val-ratio 0.15 ^
  --test-ratio 0.10 ^
  --split-strategy dominant_class ^
  --seed 42 ^
  --overwrite
```

Notes:
- `split-strategy=dominant_class` keeps split ratios per dominant-class stratum.
- This is used to keep train/valid/test ratio behavior consistent across subgroups.
- New pockmark filter options (defaults shown):
  - `--pockmark-class-id` (auto detect from `data.yaml`, fallback to class `5`)
  - `--pockmark-unstable-class-id 8`
  - `--pockmark-top-percent 0.10`
  - `--pockmark-border-px 2`

## 3) Train

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

Train only 7 classes (exclude `pockmark_unstable`, `unknown`):

```bash
python scripts/train_rfdetr.py ^
  --dataset-dir ./data/rfdetr_tiled_coco ^
  --output-dir ./runs/rfdetr-medium-7cls ^
  --model-size medium ^
  --epochs 100 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --num-workers 8 ^
  --lr 1e-4 ^
  --exclude-classes pockmark_unstable unknown ^
  --tensorboard
```

You can also choose explicit class names:

```bash
python scripts/train_rfdetr.py ... --include-classes airbubble blackspot color-distribution dust gasbubble pockmark scratch
```

## 4) Resume

Auto latest checkpoint in output dir:

```bash
python scripts/train_rfdetr.py ... --resume
```

From specific checkpoint:

```bash
python scripts/train_rfdetr.py ... --resume-from ./runs/rfdetr-medium/checkpoint_best_total.pth
```

## 5) Check broken images if DataLoader fails

```bash
python scripts/check_coco_images.py --dataset-dir ./data/rfdetr_tiled_coco
python scripts/check_coco_images.py --dataset-dir ./data/rfdetr_tiled_coco --clean
```

## 6) Class count notebook

Use `data_preprocess.ipynb` to inspect:
- source YOLO class bbox counts
- processed COCO class bbox counts
- per-split class distribution
