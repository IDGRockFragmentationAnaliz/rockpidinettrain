"""Run the standard PyTorch DiffusionEdge BSDS model over the dataset."""

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
	checkpoint_path = project_path / "models" / "diffusion_edge_bsds.pt"
	dataset_path = Path(r"D:\Data\Outcrops\unmark")

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
