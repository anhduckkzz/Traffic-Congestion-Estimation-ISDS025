from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from rich.console import Console
from rich.progress import Progress
from torch.cuda.amp import GradScaler, autocast

from src.config import load_config
from src.data import SegmentationDataset, load_split_file
from src.data.datasets import collect_directory_samples, segmentation_collate
from src.models.segmentation.deeplab_seresnet import build_segmentation_model
from src.utils import configure_logging, get_logger

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore

console = Console()


def focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 1.5,
) -> torch.Tensor:
    if logits.shape[1] > 1:
        probs = torch.softmax(logits, dim=1)[:, 1:2]
    else:
        probs = torch.sigmoid(logits)
    targets = targets.float()
    if targets.ndim == 4 and targets.shape[1] != probs.shape[1]:
        targets = targets.squeeze(1).unsqueeze(1)

    true_pos = torch.sum(probs * targets)
    false_neg = torch.sum(targets * (1 - probs))
    false_pos = torch.sum((1 - targets) * probs)
    tversky = (true_pos + 1e-6) / (true_pos + alpha * false_neg + beta * false_pos + 1e-6)
    loss = torch.pow((1 - tversky), gamma)
    return loss


def dice_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    if preds.shape[1] > 1:
        probs = torch.softmax(preds, dim=1)[:, 1]
    else:
        probs = torch.sigmoid(preds[:, 0])
    preds_bin = (probs > threshold).float()
    targets_bin = targets.float().squeeze(1)
    intersection = (preds_bin * targets_bin).sum(dim=(1, 2))
    union = preds_bin.sum(dim=(1, 2)) + targets_bin.sum(dim=(1, 2))
    dice = ((2 * intersection + 1e-6) / (union + 1e-6)).mean().item()
    return dice


def _resolve_samples(dataset_cfg: Dict[str, Any]) -> Tuple[Sequence, Sequence]:
    root = dataset_cfg["root"]
    image_dir = dataset_cfg.get("image_dir", "images")
    mask_dir = dataset_cfg.get("mask_dir", "masks")
    image_suffix = dataset_cfg.get("image_suffix", ".jpg")
    mask_suffix = dataset_cfg.get("mask_suffix", ".png")
    format_hint = dataset_cfg.get("format", "splits").lower()

    if format_hint == "splits":
        train_split = load_split_file(dataset_cfg["train_split"])
        val_split = load_split_file(dataset_cfg["val_split"])
        return train_split, val_split

    if format_hint == "directory":
        train_dir = dataset_cfg.get("train_subdir", "train")
        val_dir = dataset_cfg.get("val_subdir", "val")
        train_samples = collect_directory_samples(
            root=root,
            image_dir=image_dir,
            mask_dir=mask_dir,
            split_subdir=train_dir,
            image_suffix=image_suffix,
            mask_suffix=mask_suffix,
        )
        val_samples = collect_directory_samples(
            root=root,
            image_dir=image_dir,
            mask_dir=mask_dir,
            split_subdir=val_dir,
            image_suffix=image_suffix,
            mask_suffix=mask_suffix,
        )
        return train_samples, val_samples

    raise ValueError(f"Unsupported dataset format: {format_hint}")


def create_dataloaders(cfg: Dict[str, Any], device: torch.device):
    dataset_cfg = cfg["dataset"]
    train_samples, val_samples = _resolve_samples(dataset_cfg)

    common_kwargs = dict(
        root=dataset_cfg["root"],
        image_dir=dataset_cfg.get("image_dir", "images"),
        mask_dir=dataset_cfg.get("mask_dir", "masks"),
        image_suffix=dataset_cfg.get("image_suffix", ".jpg"),
        mask_suffix=dataset_cfg.get("mask_suffix", ".png"),
        palette_csv=dataset_cfg.get("palette_csv"),
        include_class_ids=dataset_cfg.get("include_class_ids"),
        background_class_id=dataset_cfg.get("background_class_id", 0),
        binary_mask=dataset_cfg.get("binary_mask", True),
    )

    train_dataset = SegmentationDataset(
        samples=train_samples,
        augmentations=dataset_cfg.get("augmentation", {}),
        **common_kwargs,
    )
    val_dataset = SegmentationDataset(
        samples=val_samples,
        **common_kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
        collate_fn=segmentation_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
        collate_fn=segmentation_collate,
    )
    return train_loader, val_loader


def _init_wandb(cfg: Dict[str, Any], model: torch.nn.Module) -> Any:
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    if wandb is None:  # pragma: no cover - runtime guard
        raise ImportError("wandb is not installed. Install it or disable wandb in the config.")

    run = wandb.init(
        project=wandb_cfg.get("project", "isds-traffic-ai"),
        name=wandb_cfg.get("run_name"),
        entity=wandb_cfg.get("entity"),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        config={
            "dataset": cfg.get("dataset", {}),
            "model": cfg.get("model", {}),
            "training": cfg.get("training", {}),
            "loss": cfg.get("loss", {}),
        },
    )
    if wandb_cfg.get("watch", True):
        wandb.watch(
            model,
            log=wandb_cfg.get("watch_log", "gradients"),
            log_freq=int(wandb_cfg.get("watch_log_freq", 100)),
        )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DeepLabV3 SEResNet34 segmentation model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    configure_logging()
    logger = get_logger("train_segmentation")

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_loader, val_loader = create_dataloaders(cfg, device)

    model = build_segmentation_model(
        num_classes=cfg["dataset"].get("num_classes", 2),
        output_stride=cfg["model"]["backbone"].get("output_stride", 16),
    )
    model.to(device)

    training_cfg = cfg["training"]
    epochs = training_cfg["epochs"]
    lr = training_cfg["learning_rate"]
    weight_decay = training_cfg.get("weight_decay", 0.0)
    optimizer_name = training_cfg.get("optimizer", "adamw").lower()

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_cfg.get("scheduler", {}).get("t_initial", 10),
        eta_min=training_cfg.get("scheduler", {}).get("eta_min", 1e-5),
    )

    scaler = GradScaler(enabled=training_cfg.get("mixed_precision", True) and device.type == "cuda")

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "outputs/segmentation"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    wandb_run = _init_wandb(cfg, model)

    try:
        with Progress(console=console) as progress:
            train_task = progress.add_task("train", total=len(train_loader))
            val_task = progress.add_task("val", total=len(val_loader))

            for epoch in range(1, epochs + 1):
                model.train()
                progress.reset(train_task, total=len(train_loader))
                train_loss_total = 0.0
                train_batches = 0

                for batch in train_loader:
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=scaler.is_enabled()):
                        logits = model(images)["out"]
                        loss = focal_tversky_loss(
                            logits,
                            masks,
                            alpha=cfg["loss"].get("tversky_alpha", 0.7),
                            beta=cfg["loss"].get("tversky_beta", 0.3),
                            gamma=cfg["loss"].get("focal_gamma", 1.5),
                        )
                    loss_value = float(loss.detach().item())
                    scaler.scale(loss).backward()
                    if training_cfg.get("gradient_clip_norm"):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg["gradient_clip_norm"])
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_total += loss_value
                    train_batches += 1
                    progress.advance(train_task)

                train_loss = train_loss_total / max(train_batches, 1)

                model.eval()
                progress.reset(val_task, total=len(val_loader))
                dice_scores = []
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch["image"].to(device)
                        masks = batch["mask"].to(device)
                        logits = model(images)["out"]
                        dice_scores.append(dice_score(logits, masks))
                        progress.advance(val_task)
                mean_dice = float(sum(dice_scores) / max(len(dice_scores), 1))
                current_lr = optimizer.param_groups[0]["lr"]

                logger.info("Epoch %d | TrainLoss=%.4f | Dice=%.4f | LR=%.6f", epoch, train_loss, mean_dice, current_lr)

                if wandb_run is not None:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/dice": mean_dice,
                        "lr": current_lr,
                    })

                scheduler.step()

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    ckpt_path = checkpoint_dir / "best.ckpt"
                    torch.save({"epoch": epoch, "dice": mean_dice, "state_dict": model.state_dict()}, ckpt_path)
                    logger.info("Saved checkpoint to %s", ckpt_path)
                    if wandb_run is not None:
                        wandb_run.summary["best_val_dice"] = best_dice
                        wandb_run.summary["best_checkpoint"] = str(ckpt_path)
    finally:
        if wandb_run is not None:
            wandb.finish()

    console.print(f"Training completed. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
