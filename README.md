# RF-DETR Pipeline for PET inspection Pipeline

This project builds a training dataset from:
- `Dataset-v3.v1i.yolov5pytorch/train/images`
- `Dataset-v3.v1i.yolov5pytorch/train/labels`

Workflow:
1. Resize full images from `2046x2046` to `2048x2048`
2. Filename rule relabel to `unknown`:
   - `color-distribution`, `gasbubble`, `airbubble` labels become `unknown` when filename does not contain `colordistribution`, `gas`, `air`
3. Size filter for `airbubble / gasbubble / color-distribution`:
   - keep only when bbox width `>=10px` OR height `>=10px` (resized pixel coordinates), otherwise relabel to `unknown`
4. For `pockmark` and `blackspot`, compute contrast = `|mean(inner bbox) - mean(outer 2px ring)|`
5. Keep top 50% `pockmark` boxes, relabel remaining to `unknown`
6. Keep top 20% `blackspot` boxes, relabel remaining to `unknown`
7. Save refined secondary YOLO dataset
8. Tile to `8x8` (`256x256`)
9. Create COCO splits (`train/valid/test`)
10. Train RF-DETR

## 1) Install

```bash
python -m pip install -r requirements.txt
```

## 2) Preprocess and Split (same ratio policy)

```bash
python scripts/prepare_tiled_coco_dataset.py 
  --source-root ./Dataset-v3.v1i.yolov5pytorch 
  --images-subdir train/images 
  --labels-subdir train/labels 
  --secondary-root ./data/dataset_stage2_refined 
  --output-root ./data/rfdetr_tiled_coco 
  --val-ratio 0.15 
  --test-ratio 0.10 
  --split-strategy dominant_class 
  --min-air-side-px 10 
  --min-gas-side-px 20 
  --min-color-side-px 40 
  --pockmark-top-percent 0.50 
  --blackspot-top-percent 0.20 
  --pockmark-border-px 2 
  --seed 42 
  --overwrite
```

Notes:
- `split-strategy=dominant_class` keeps split ratios per dominant-class stratum.
- This is used to keep train/valid/test ratio behavior consistent across subgroups.
- `air/gas/color-distribution` are size-filtered by class-specific thresholds and small boxes are relabeled to `unknown`:
  - `--min-air-side-px` (default `10`)
  - `--min-gas-side-px` (default `20`)
  - `--min-color-side-px` (default `40`)
- `pockmark` keeps only top contrast ratio by `--pockmark-top-percent` (default `0.50`).
- `blackspot` keeps only top contrast ratio by `--blackspot-top-percent` (default `0.20`).
- Secondary refined YOLO dataset is saved at `--secondary-root` (workflow step-7 artifact).

## 3) Train

```bash
python scripts/train_rfdetr.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --output-dir ./runs/rfdetr-medium 
  --model-size medium 
  --epochs 100 
  --batch-size 8 
  --grad-accum-steps 2 
  --num-workers 8 
  --lr 1e-4 
  --tensorboard
```

Best-checkpoint behavior (default):
- `--save-best-metric class_mean`
- best file: `checkpoint_best_class_mean.ckpt` (and `.pth`)
- excluded from class-mean by default: `unknown` (`--best-class-mean-exclude-classes unknown`)

Use old aggregate behavior if needed:

```bash
python scripts/train_rfdetr.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --output-dir ./runs/rfdetr-medium 
  --model-size medium 
  --epochs 100 
  --batch-size 8 
  --grad-accum-steps 2 
  --num-workers 8 
  --lr 1e-4 
  --save-best-metric total 
  --tensorboard
```

Train only 7 classes (exclude `unknown`):

```bash
python scripts/train_rfdetr.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --output-dir ./runs/rfdetr-medium 
  --model-size medium 
  --epochs 100 
  --batch-size 8 
  --grad-accum-steps 2 
  --num-workers 8 
  --lr 1e-4 
  --exclude-classes unknown 
  --tensorboard
```

You can also choose explicit class names:

```bash
python scripts/train_rfdetr.py ... --include-classes airbubble blackspot color-distribution dust gasbubble pockmark scratch
```

## 4) Resume

`last` checkpoint is now saved every epoch as:
- `checkpoint_last.ckpt`

Auto resume (prefers `checkpoint_last.ckpt`):

```bash
python scripts/train_rfdetr.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --output-dir ./runs/rfdetr-medium 
  --model-size medium 
  --epochs 100 
  --batch-size 8 
  --grad-accum-steps 2 
  --num-workers 8 
  --lr 1e-4 
  --exclude-classes unknown 
  --resume 
  --tensorboard
```

Resume from best checkpoint (`--resume-best` prefers `checkpoint_best_class_mean.*`):

```bash
python scripts/train_rfdetr.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --output-dir ./runs/rfdetr-medium 
  --model-size medium 
  --epochs 100 
  --batch-size 8 
  --grad-accum-steps 2 
  --num-workers 8 
  --lr 1e-4 
  --exclude-classes unknown 
  --resume-best 
  --tensorboard
```

From specific checkpoint:

```bash
python scripts/train_rfdetr.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --output-dir ./runs/rfdetr-medium 
  --model-size medium 
  --epochs 100 
  --batch-size 8 
  --grad-accum-steps 2 
  --num-workers 8 
  --lr 1e-4 
  --exclude-classes unknown 
  --resume-from ./runs/rfdetr-medium/checkpoint_best_class_mean.ckpt 
  --tensorboard
```

## 5) Check broken images if DataLoader fails

```bash
python scripts/check_coco_images.py --dataset-dir ./data/rfdetr_tiled_coco
python scripts/check_coco_images.py --dataset-dir ./data/rfdetr_tiled_coco --clean
```

## 6) Visualization class-index note

Prediction visualization uses zero-based class index mapping (RF-DETR `predict()` output convention).
No class-index-base option is needed.

## 7) Split Visualization (Best Model)

Use `checkpoint_best_total.pth` to visualize `train/valid/test` splits.
Class-wise best thresholds (from PR-curve json) are auto-applied when found.

```bash
# train split
python scripts/visualize_coco_bboxes.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --split train 
  --mode both 
  --run-dir ./runs/rfdetr-medium 
  --checkpoint ./runs/rfdetr-medium/checkpoint_best_total.pth 
  --model-size medium 
  --threshold 0.3 
  --skip-gt-only-classes unknown pockmark_unstable 
  --max-images 0 
  --skip-empty 
  --output-dir ./runs/vis/train_both_best

# valid split
python scripts/visualize_coco_bboxes.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --split valid 
  --mode both 
  --run-dir ./runs/rfdetr-medium 
  --checkpoint ./runs/rfdetr-medium/checkpoint_best_total.pth 
  --model-size medium 
  --threshold 0.3 
  --skip-gt-only-classes unknown pockmark_unstable 
  --max-images 0 
  --skip-empty 
  --output-dir ./runs/vis/valid_both_best

# test split
python scripts/visualize_coco_bboxes.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --split test 
  --mode both 
  --run-dir ./runs/rfdetr-medium 
  --checkpoint ./runs/rfdetr-medium/checkpoint_best_total.pth 
  --model-size medium 
  --threshold 0.3 
  --skip-gt-only-classes unknown pockmark_unstable 
  --max-images 0 
  --skip-empty 
  --output-dir ./runs/vis/test_both_best
```

Optional: force threshold json path explicitly:

```bash
python scripts/visualize_coco_bboxes.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --split test 
  --mode both 
  --run-dir ./runs/rfdetr-medium 
  --checkpoint ./runs/rfdetr-medium/checkpoint_best_total.pth 
  --model-size medium 
  --threshold 0.3 
  --class-threshold-json ./runs/pr_auc_eval/medium/pr_curves_test.json 
  --output-dir ./runs/vis/test_both_best
```

## 8) Stage2 Refinement Before/After Compare

For the specific stage2 refinement:
- filename-rule relabel (`air/gas/color -> unknown`)
- size filter (`air/gas/color`, min side px) then relabel small boxes to unknown
- pockmark contrast top-50% keep (`pockmark -> unknown` for the rest)
- blackspot contrast top-20% keep (`blackspot -> unknown` for the rest)

compare **original YOLO vs `data/dataset_stage2_refined` YOLO**:

```bash
python scripts/compare_stage2_bbox_counts.py 
  --source-root ./Dataset-v3.v1i.yolov5pytorch 
  --source-labels-dir train/labels 
  --stage2-root ./data/dataset_stage2_refined 
  --stage2-labels-dir train/labels
```

Save transition summary JSON:

```bash
python scripts/compare_stage2_bbox_counts.py 
  --source-root ./Dataset-v3.v1i.yolov5pytorch 
  --source-labels-dir train/labels 
  --stage2-root ./data/dataset_stage2_refined 
  --stage2-labels-dir train/labels 
  --save-json ./runs/stage2_compare_summary.json
```

## 9) Test Confusion Matrix (After Tiling)

Generate confusion matrix on tiled COCO `test` split using best RF-DETR checkpoint.
Class-wise best thresholds (from PR-curve json) are auto-applied when found:

```bash
python scripts/eval_confusion_matrix.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --split test 
  --run-dir ./runs/rfdetr-medium 
  --checkpoint ./runs/rfdetr-medium/checkpoint_best_total.pth 
  --model-size medium 
  --threshold 0.5 
  --iou-threshold 0.3 
  --skip-gt-only-classes unknown pockmark_unstable 
  --max-images 0 
  --output-dir ./runs/confusion/test_best
```

Optional: force threshold json path explicitly:

```bash
python scripts/eval_confusion_matrix.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --split test 
  --run-dir ./runs/rfdetr-medium 
  --checkpoint ./runs/rfdetr-medium/checkpoint_best_total.pth 
  --model-size medium 
  --threshold 0.5 
  --class-threshold-json ./runs/pr_auc_eval/medium/pr_curves_test.json 
  --iou-threshold 0.3 
  --output-dir ./runs/confusion/test_best
```

Outputs:
- raw confusion csv/png (with background row/col)
- row-normalized confusion png
- matched-only class confusion csv/png
- summary json
- summary includes `threshold_mode` (`classwise_best_f1` or `global`)

## 10) Class-wise PR Curve / PR-AUC / Best Threshold

Evaluate class-wise PR curves on tiled COCO split and compute:
- PR-AUC (AP-style PR integration)
- best threshold per class (max F1)

Default model-version to run-dir mapping:
- `medium` -> `./runs/rfdetr-medium`
- `medium-v2` -> `./runs/rfdetr-medium-v2`
- `large` -> `./runs/rfdetr-large`

```bash
# medium
python scripts/eval_pr_auc_threshold.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --split test 
  --model-version medium 
  --run-dir ./runs/rfdetr-medium 
  --exclude-classes unknown 
  --output-dir ./runs/pr_auc_eval

# large
python scripts/eval_pr_auc_threshold.py 
  --dataset-dir ./data/rfdetr_tiled_coco 
  --split test 
  --model-version large 
  --run-dir ./runs/rfdetr-large 
  --exclude-classes unknown 
  --output-dir ./runs/pr_auc_eval
```

Optional:
- specify checkpoint directly: `--checkpoint ./runs/rfdetr-medium/checkpoint_best_total.pth`
- adjust IoU threshold for TP matching: `--iou-threshold 0.5`
- collect denser PR points with low predictor threshold: `--infer-threshold 0.001`
- subset check: `--max-images 200`

Outputs (per model version under output dir):
- `pr_auc_summary_<split>.csv`
- `pr_curves_<split>.json`
- `pr_curves_<split>.png`

