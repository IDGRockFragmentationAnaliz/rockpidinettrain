"""Fine-tune RCF on image/edge[/mask] manifest pairs."""

from pathlib import Path

from rockedgesdetectors.pyrcf.train import TrainingConfig, run_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = TrainingConfig(
    train_manifest=Path(r"D:\Data\Outcrops\train.lst"),
    initial_checkpoint=PROJECT_ROOT / "models" / "bsds500_pascal_model.pth",
    resume_checkpoint=None,
    checkpoint_folder=PROJECT_ROOT / "save_models" / "rcf",
    epochs=10,
    crop_size=320,
    batch_size=1,
    accumulation_steps=10,
    num_workers=2,
    validation_fraction=0.1,
    seed=42,
    learning_rate=1e-6,
    momentum=0.9,
    weight_decay=2e-4,
    lr_step_size=3,
    lr_gamma=0.1,
    negative_pixel_weight=1.1,
    label_threshold=0.5,
    ignore_ambiguous=True,
    min_edge_pixels_per_crop=5,
    crop_attempts=10,
    gradient_clip_norm=None,
    augment_flips=True,
    save_every_epochs=1,
    device="cuda",
)


if __name__ == "__main__":
    run_training(CONFIG)
