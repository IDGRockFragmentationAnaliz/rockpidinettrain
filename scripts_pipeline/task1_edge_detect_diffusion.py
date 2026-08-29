"""Run the standard PyTorch DiffusionEdge BSDS model over the dataset."""

import tomllib
from pathlib import Path

import torch

from rockedgesdetectors import (
	Cropper,
	DiffusionEdgeBSDS,
	NumpyDiffusionEdgeAdapter,
)
from storage_manager import Storage


def main() -> None:
	project_path = Path(__file__).resolve().parents[1]
	config_path = project_path / "config.toml"

	with config_path.open("rb") as config_file:
		config = tomllib.load(config_file)
	dataset_path = Path(config["preparation"]["folder_dataset"])
	if not dataset_path.is_absolute():
		dataset_path = project_path / dataset_path

	checkpoint_path = project_path / "models" / "diffusion_edge_bsds.pt"

	torch.manual_seed(42)
	module = DiffusionEdgeBSDS(
		checkpoint_path,
		sampling_timesteps=5,
	).cuda().eval()
	model = NumpyDiffusionEdgeAdapter(module)
	model = Cropper(
		model,
		crop=320,
		pad=40,
		display=True,
	)

	for folder_path in sorted(dataset_path.iterdir()):
		if not folder_path.is_dir():
			continue
		storage = Storage.from_folder_path(folder_path)
		image = storage.load_image()
		edges = model(image)
		storage.save_grayscale(
			edges,
			suffix="_edges_diffusion",
		)


if __name__ == "__main__":
	main()
