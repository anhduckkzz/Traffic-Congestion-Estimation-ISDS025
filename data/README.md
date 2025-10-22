# Dataset Preparation

The project uses two datasets: one for **road marking segmentation** (RLMD-style) and another for **vehicle detection / tracking & speed estimation**. Paths are configurable via the YAML files.

---
## 1. Road Marking Segmentation (RLMD)

### Directory Layout
```
data/segmentation/
  images/
    train/
      *.jpg
    val/
      *.jpg
  labels/
    train/
      *.png
    val/
      *.png
  rlmd.csv            # color palette (class_id,name,R,G,B)
```
- `images/{split}` — RGB road-scene frames.
- `labels/{split}` — RGB masks whose colors encode RLMD classes.
- `rlmd.csv` — Palette shipped with the dataset; required for color?class conversion.

### Configuration (`configs/segmentation.yaml`)
- `format: directory` tells the loader to scan the folders above for train/val splits.
- `palette_csv` points to `rlmd.csv`, enabling palette-driven decoding.
- `include_class_ids` **defaults to `[4, 5, 6, 7, 8, 9, 10, 16]`**, matching Table 2 of the paper (solid/dashed white & yellow lines, solid red, and channelizing lines). Only these markings populate the binary mask; edit the list if you need additional symbols (e.g., stop lines or arrows).
- `binary_mask: true` merges the selected classes into `{0,1}` (background vs. marking).
- Augmentations such as `horizontal_flip`, `color_jitter`, and `gaussian_noise` are configured under `dataset.augmentation`.

### Notes
- Images and masks must share dimensions; resizing is handled inside the training pipeline.
- The loader converts RLMD RGB masks to class IDs via the palette, then to binary masks using `include_class_ids`.

---
## 2. Vehicle Detection, Tracking, and Speed

### Directory Layout
```
data/detection/
  images/
    train/
    val/
  labels/
    train/
    val/
  meta/
    speeds.csv        # optional
    calibration.json  # optional
```

#### Bounding Boxes
- Default annotation format: YOLO TXT (`<class_id> <cx> <cy> <w> <h>`, values normalized to `[0,1]`). Class IDs must align with `configs/detection.yaml:data.class_names` (e.g., `0=motorbike`, `1=car`).
- To use COCO JSON instead, set `data.format: "coco"` and point the label paths to the JSON files.

#### Optional Metadata
- `meta/speeds.csv`: per-clip ground-truth speed, columns `video_id,timestamp_start,timestamp_end,mean_speed_kph` (add more columns as needed).
- `meta/calibration.json`: camera parameters if you prefer fixed meter-per-pixel ratios over ROI-derived calibration.

### Video Clips
- Training consumes still images, but the inference pipeline expects video. Keep the source `.mp4` / `.avi` clips available and provide their paths when running `scripts/run_pipeline.py`.

---
## Configuration Tips
- Adjust the YAML files if your directory names, extensions, or formats differ.
- Map each YOLO class ID to `"two_wheeler"` or `"four_wheeler"` in `configs/pipeline.yaml:reporting.class_map` so LoS thresholds apply correctly.
- Update `roi.dash_lengths_m` if your local road markings follow different physical dimensions.

---
## Data Quality Checklist
- Ensure lane markings are clearly visible; heavy wear can break ROI extraction.
- Keep annotation timestamps aligned with any speed measurements.
- Balance vehicle classes (especially motorbikes vs. cars) for stable detector training.

Once the folders follow these layouts, the provided training and inference scripts will discover the datasets automatically.
