# Towards End-to-End Traffic Congestion Estimation Using Learned ROI and Vehicle Object Dynamics

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://doi.org/10.1007/978-981-95-3358-9_17)
[![Demo](https://img.shields.io/badge/Demo-Coming%20Soon-blue.svg)](#live-demo)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation of the ISDS2025 paper **"Towards End-to-End Traffic Congestion Estimation Using Learned ROI and Vehicle Object Dynamics"**. The codebase reproduces the full pipeline described in the paper: lightweight road-marking segmentation, geometry-aware ROI derivation, YOLO11s-based vehicle analytics, and Level-of-Service (LoS) congestion classification tailored to motorbike-dominant cities such as Ho Chi Minh City.

## Highlights
- **Lightweight segmentation**  DeepLabV3+ head with a ResNet34 backbone enhanced by squeeze-and-excitation and depthwise-separable convolutions, delivering high Dice/mIoU at edge-device speed.
- **Automatic ROI discovery**  Geometry-aware post-processing that converts dashed lane markings into calibrated ROI lines and meter-per-pixel ratios, removing manual camera setup.
- **Multi-object analytics**  YOLO11s fine-tuning, BoT-SORT tracking, and displacement-based speed estimation provide per-lane traffic counts and velocities.
- **LoS classification**  Implements the Highway Capacity Manual thresholds with dual speed tables (two-wheel vs. four-wheel) to output congestion levels AF.
- **Production-ready project**  Structured Python package with configs, scripts, tests, and utilities for training, inference, and evaluation.

## Live Demo
- Coming soon. After training your models, plug the exported checkpoints into `configs/pipeline.yaml` and use `scripts/run_pipeline.py` to generate annotated summaries per 30?s window.

## Architecture Overview
```
Raw Video
   
   +-? DeepLabV3 + SEResNet34 ? segmentation mask
           +-? Lane ROI extractor ? ROI lines + meter/pixel
   
   +-? YOLO11s + BoT-SORT ? tracked vehicle trajectories
           +-? Speed estimator ? per-track speed (km/h)
   
   +-? Window aggregator ? per-lane density, mean speed ? LoS label AF
```

## 📁 Repository Structure

```text
Traffic-Congestion-Estimation-ISDS2025/
├── src/                      # Core Python source package
│   ├── config/               # YAML configuration helpers
│   ├── data/                 # Dataset loaders, preprocessors, and utilities
│   ├── models/               # Deep learning model definitions
│   │   ├── segmentation/     # SEResNet34 + DeepLabV3 implementation
│   │   └── detection/        # YOLOv11s wrapper + BoT-SORT for real-time tracking
│   ├── pipeline/             # ROI extraction, speed estimation, and congestion logic
│   └── utils/                # Logging, visualization, and video helper functions
│
├── configs/                  # YAML presets for segmentation, detection, and pipeline
├── scripts/                  # CLI tools for training, evaluation, and inference
├── tests/                    # Unit tests for core components
├── data/README.md            # Notes on dataset structure and RLMD metadata
│
├── README.md                 # Project overview and usage instructions
├── requirements.txt          # Python dependencies
└── LICENSE                   # License file (if applicable)


## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Training
**Segmentation (RLMD-ready):**
```bash
python -m scripts.train_segmentation --config configs/segmentation.yaml
```
- Defaults to the RLMD dataset layout (`data/segmentation/images/{train,val}`, `data/segmentation/labels/{train,val}`) with class colors specified in `data/segmentation/rlmd.csv`.
- Config defaults to the eight RLMD classes highlighted in Table 2 (IDs 4, 5, 6, 7, 8, 9, 10, 16) so only lane-separator markings populate the binary mask; adjust the list if your deployment needs other road symbols.
- Toggle wandb.enabled in configs/segmentation.yaml to stream metrics to Weights & Biases; adjust project/run metadata in the same block.

**Vehicle Detection & Tracking:**
```bash
python -m scripts.finetune_yolo --config configs/detection.yaml
```
Both scripts rely on the dataset layouts described in [`data/README.md`](data/README.md) and export checkpoints to `outputs/` by default.

## Inference Pipeline
Once checkpoints are ready, update `configs/pipeline.yaml` with their paths and run:
```bash
python -m scripts.run_pipeline --config configs/pipeline.yaml --video path/to/clip.mp4
```
The script prints per-window LoS summaries and (optionally) saves JSON reports.

## Datasets
- **Road Marking Segmentation (RLMD example):** Drop the RLMD dataset under `data/segmentation/` so that images live in `images/{train,val}` and label PNGs in `labels/{train,val}`. The provided `rlmd.csv` palette maps RGB colors to class IDs.
- **Detection & Speed:** Use a YOLO-compatible dataset with class labels for motorbikes, cars, trucks, etc. Optional `speeds.csv` can store ground-truth flow speed for validation.

See [`data/README.md`](data/README.md) for exact folder expectations, palette handling, and how to override them via YAML.

## Results
| Component | Metric | Target (Paper) |
|-----------|--------|----------------|
| Road segmentation | Dice / mIoU | 75.4% / 73.9% |
| Vehicle detection | mAP@50 | 83.1% (YOLO11s finetuned) |
| Speed estimation | % Error | 7.82% |
| Congestion mapping | Macro F1 | 85.3% |

Actual performance depends on dataset quality, training length, and hardware.

## Analysis & Future Work
- **Camera shifts:** Current ROI lines are computed once per run; integrating temporal ROI alignment (e.g., Temporal ROI Align) would improve robustness to jitter.
- **Weather robustness:** Augmentations partially mimic rain/fog, but real-world dynamics may require collecting adverse-weather footage.
- **Dynamic windows:** The fixed 30?s window can be adaptive to capture abrupt congestion changes.
- **Visualization:** Overlaying ROIs and track info on video output can assist operators; extend `run_pipeline.py` to use `VideoWriter` for this.

## Citation
If you use this code or the accompanying dataset, please cite the original paper and acknowledge this implementation.

```
@inproceedings{le2025congestion,
  title={Towards End-to-End Traffic Congestion Estimation Using Learned ROI and Vehicle Object Dynamics},
  author={Le, Nguyen Khang and Tran, Anh Duc and Tran, Thi Minh Tam and Do, Khanh Ngoc},
  booktitle={International Conference on Intelligent Systems and Data Science (ISDS)},
  year={2025}
}
```

## License
MIT License. Dataset licenses may impose additional restrictionsconsult the original sources before redistribution.

