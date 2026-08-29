"""Run the pure-PyTorch DDN-M36 BSDS500 model over the dataset."""

from pathlib import Path

import torch

from rockedgesdetectors import Cropper, DDNBSDS, NumpyDDNAdapter
from storage_manager import Storage


def main() -> None:
    project_path = Path(__file__).resolve().parents[1]
    config_path = project_path / "config.toml"

    checkpoint_path = project_path / "models" / "ddn_bsds500.pth"
    dataset_path = Path(r"D:\Data\Outcrops\unmark")

    module = DDNBSDS(checkpoint_path).cuda().eval()
    model = NumpyDDNAdapter(module)
    model = Cropper(
        model,
        crop=512,
        pad=128,
        pad_mode="reflect",
        display=True,
    )

    for folder_path in sorted(dataset_path.iterdir()):
        if not folder_path.is_dir():
            continue
        storage = Storage.from_folder_path(folder_path)
        image = storage.load_image()
        edges = model(image)
        edges -= edges.min()
        maximum = edges.max()
        if maximum > 0:
            edges /= maximum
        storage.save_grayscale(edges, suffix="_edges_ddn")


if __name__ == "__main__":
    main()
