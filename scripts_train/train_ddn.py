"""Fine-tune DDN-M36 on the project's image/edge manifest."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rockedgesdetectors import DDNBSDS
from rockedgesdetectors.ddn.training import (
    DDNLoss,
    DDNTrainer,
    EdgeManifestDataset,
    create_ddn_optimizer,
    restore_training_checkpoint,
    save_training_checkpoint,
)


# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MANIFEST = Path(r"D:\Data\Outcrops\train.lst")
INITIAL_CHECKPOINT = PROJECT_ROOT / "models" / "ddn_bsds500.pth"
RESUME_CHECKPOINT: Path | None = None
CHECKPOINT_FOLDER = PROJECT_ROOT / "save_models" / "ddn"

EPOCHS = 40
CROP_SIZE = 320
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_WORKERS = 2
VALIDATION_FRACTION = 0.1
SEED = 42

LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE_SCALE = 0.1
WEIGHT_DECAY = 5e-4
LR_STEP_SIZE = 3
LR_GAMMA = 0.1

NEGATIVE_PIXEL_WEIGHT = 1.1
KL_WEIGHT = 1e-2
LABEL_THRESHOLD = 0.5
MIN_EDGE_PIXELS_PER_CROP = 5
CROP_ATTEMPTS = 10

GRADIENT_CLIP_NORM: float | None = None
AUGMENT_FLIPS = True
SAVE_EVERY_EPOCHS = 1


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DDN training is configured for CUDA")
    if not 0 <= VALIDATION_FRACTION < 1:
        raise ValueError("VALIDATION_FRACTION must be in [0, 1)")

    seed_everything(SEED)
    device = torch.device("cuda")
    train_loader, validation_loader = create_loaders()

    source_checkpoint = RESUME_CHECKPOINT or INITIAL_CHECKPOINT
    model = DDNBSDS(source_checkpoint, trainable=True).to(device)
    optimizer = create_ddn_optimizer(
        model,
        learning_rate=LEARNING_RATE,
        encoder_lr_scale=ENCODER_LEARNING_RATE_SCALE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )
    loss_function = DDNLoss(
        negative_weight=NEGATIVE_PIXEL_WEIGHT,
        kl_weight=KL_WEIGHT,
    )
    trainer = DDNTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        # The original DDN training runs entirely in float32 and does not use AMP.
        use_amp=False,
        accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
    )

    start_epoch = 1
    if RESUME_CHECKPOINT is not None:
        completed_epoch = restore_training_checkpoint(
            RESUME_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scaler=trainer.scaler,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = completed_epoch + 1
        print(f"Resumed DDN training after epoch {completed_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        validation_metrics = (
            trainer.validate(validation_loader, epoch)
            if validation_loader is not None
            else None
        )
        scheduler.step()

        print_metrics(epoch, "train", train_metrics)
        if validation_metrics is not None:
            print_metrics(epoch, "validation", validation_metrics)

        if epoch % SAVE_EVERY_EPOCHS == 0 or epoch == EPOCHS:
            checkpoint_path = CHECKPOINT_FOLDER / f"checkpoint_{epoch:03d}.pth"
            saved_path = save_training_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=trainer.scaler,
                scheduler=scheduler,
                metrics={
                    "train": vars(train_metrics),
                    "validation": (
                        vars(validation_metrics)
                        if validation_metrics is not None
                        else None
                    ),
                },
            )
            print(f"Saved DDN checkpoint: {saved_path}")


def create_loaders() -> tuple[DataLoader, DataLoader | None]:
    train_dataset = EdgeManifestDataset(
        TRAIN_MANIFEST,
        crop_size=CROP_SIZE,
        crop_mode="random",
        label_threshold=LABEL_THRESHOLD,
        augment=AUGMENT_FLIPS,
        min_edge_pixels=MIN_EDGE_PIXELS_PER_CROP,
        crop_attempts=CROP_ATTEMPTS,
    )
    validation_dataset = EdgeManifestDataset(
        TRAIN_MANIFEST,
        crop_size=CROP_SIZE,
        crop_mode="center",
        label_threshold=LABEL_THRESHOLD,
        augment=False,
        min_edge_pixels=0,
    )
    train_indices, validation_indices = split_indices(
        len(train_dataset),
        VALIDATION_FRACTION,
        SEED,
    )
    train_subset = Subset(train_dataset, train_indices)
    validation_subset = (
        Subset(validation_dataset, validation_indices)
        if validation_indices
        else None
    )

    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(train_subset, shuffle=True, **common)
    validation_loader = (
        DataLoader(validation_subset, shuffle=False, **common)
        if validation_subset is not None
        else None
    )
    print(
        f"DDN dataset: train={len(train_subset)}, "
        f"validation={len(validation_subset) if validation_subset else 0}"
    )
    return train_loader, validation_loader


def split_indices(
    length: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    validation_size = int(round(length * validation_fraction))
    if validation_fraction > 0 and length > 1:
        validation_size = max(1, min(validation_size, length - 1))
    return indices[validation_size:], indices[:validation_size]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_metrics(epoch: int, split: str, metrics) -> None:
    print(
        f"epoch={epoch} split={split} loss={metrics.loss:.4f} "
        f"bce={metrics.bce:.4f} kl={metrics.kl:.4f} "
        f"positive_pixels={metrics.positive_pixels:.1f}"
    )


if __name__ == "__main__":
    main()
