"""Run the official RCF BSDS500+PASCAL model over the configured dataset."""

import tomllib
from pathlib import Path

import torch

from rockedgesdetectors import Cropper, RCFBSDS, NumpyRCFAdapter
from storage_manager import Storage


def main() -> None:
    project_path = Path(__file__).resolve().parents[1]
    with (project_path / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)
    # dataset_path = Path(config["preparation"]["folder_dataset"])
    dataset_path = Path(config["validation"]["folder_validation"])
    if not dataset_path.is_absolute():
        dataset_path = project_path / dataset_path

    checkpoint_path = project_path / "models" / "bsds500_pascal_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = RCFBSDS(checkpoint_path).to(device).eval()
    model = Cropper(
        NumpyRCFAdapter(module),
        crop=512,
        pad=128,
        pad_mode="reflect",
        display=True,
    )

    for folder_path in sorted(dataset_path.iterdir()):
        if not folder_path.is_dir():
            continue
        storage = Storage.from_folder_path(folder_path)
        edges = model(storage.load_image())
        storage.save_grayscale(edges, suffix="_edges_rcf")


if __name__ == "__main__":
    main()
