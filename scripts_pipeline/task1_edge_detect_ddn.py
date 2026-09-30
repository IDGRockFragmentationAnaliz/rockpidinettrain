"""Run the pure-PyTorch DDN-M36 BSDS500 model over the dataset."""

import tomllib
from pathlib import Path
import cv2
import torch
import numpy as np

from rockedgesdetectors import Cropper, DDN, NumpyDDNAdapter, BatchedCropper, BlendingCropper
from storage_manager import Storage


def normalize_image_min_max(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image_min = np.min(image)
    image_max = np.max(image)
    if image_max == image_min:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - image_min) / (image_max - image_min) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def main() -> None:
    project_path = Path(__file__).resolve().parents[1]
    config_path = project_path / "config.toml"

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    # dataset_path = Path(config["preparation"]["folder_dataset"])
    dataset_path = Path(config["validation"]["folder_validation"])
    if not dataset_path.is_absolute():
        dataset_path = project_path / dataset_path

    #vcheckpoint_path = project_path / "models" / "models_ddn"  / "ddn_bsds500.pth"
    checkpoint_path = project_path / "models" / "models_ddn" / "50_16" / "checkpoint_013.pth"
    model = load_model(checkpoint_path)
    model = BlendingCropper(
        model,
        crop=350,
        pad_mode="reflect",
        display=True,
    )
    for folder_path in sorted(dataset_path.iterdir()):
        if not folder_path.is_dir():
            continue
        storage = Storage.from_folder_path(folder_path)
        image = storage.load_image()

        # Запоминаем исходный размер
        height, width = image.shape[:2]

        # Уменьшаем в 2 раза
        image = cv2.resize(
            image,
            (width // 1, height // 1),
            interpolation=cv2.INTER_AREA
        )

        edges = model(image)

        # Возвращаем ТОЧНО исходный размер
        edges = cv2.resize(
            edges,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )
        edges = normalize_image_min_max(edges)

        storage.save_grayscale(edges, suffix="_50_16", subfolder="ddn/edges")

def load_model(checkpoint_path: Path):
    model = DDN(checkpoint_path).cuda().eval()
    model = NumpyDDNAdapter(model)
    return model

if __name__ == "__main__":
    main()
