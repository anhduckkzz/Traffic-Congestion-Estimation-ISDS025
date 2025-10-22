from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console

from src.config import load_config
from src.models.detection import YOLODetector
from src.utils import configure_logging, get_logger

console = Console()


def write_dataset_yaml(cfg: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "dataset.yaml"
    content = f"""
path: {cfg['root']}
train: {cfg['train_images']}
val: {cfg['val_images']}
names: {cfg['class_names']}
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO detector using Ultralytics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_logging()
    logger = get_logger("finetune_yolo")

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    logging_cfg = cfg.get("logging", {})

    project_dir = Path(logging_cfg.get("project_dir", "outputs/detection"))
    experiment_name = logging_cfg.get("experiment_name", "yolo_finetune")
    dataset_yaml = write_dataset_yaml(data_cfg, project_dir)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    detector = YOLODetector(
        weights=model_cfg["pretrained_weights"],
        device=device,
        conf_threshold=model_cfg.get("conf_threshold", 0.25),
        iou_threshold=model_cfg.get("iou_threshold", 0.45),
    )

    train_kwargs = {
        "epochs": model_cfg.get("epochs", 100),
        "batch": model_cfg.get("batch_size", 16),
        "img_size": model_cfg.get("img_size", 1280),
        "project_dir": project_dir,
        "name": experiment_name,
        "lr0": model_cfg.get("learning_rate", 0.002),
        "momentum": model_cfg.get("momentum", 0.937),
        "weight_decay": model_cfg.get("weight_decay", 0.0005),
        "warmup_epochs": model_cfg.get("warmup_epochs", 3),
    }

    extra_args = {}
    augment_cfg = model_cfg.get("augment", {})
    if augment_cfg.get("hsv") is False:
        extra_args["hsv"] = 0.0
    if augment_cfg.get("mosaic") is False:
        extra_args["mosaic"] = 0.0
    if augment_cfg.get("mixup") is True:
        extra_args["mixup"] = 0.2

    train_kwargs.update(extra_args)

    logger.info("Starting YOLO fine-tuning -> %s", project_dir)
    detector.finetune(data_cfg=dataset_yaml, **train_kwargs)
    console.print(f"Fine-tuning complete. Checkpoints saved to {project_dir / experiment_name}")


if __name__ == "__main__":
    main()
