import tomllib
from pathlib import Path

import torch

from rockedgesdetectors import Cropper, ModelRCF, NumpyImagenetAdapter
from rockedgesdetectors.pidinet.config import config_model
from rockedgesdetectors.pidinet.models import PiDiNet
from rocknetmanager.manager_shapefile import label_load
from storage_manager import Storage
import matplotlib.pyplot as plt



def main():
	project_path = Path(__file__).resolve().parents[1]
	config_path = project_path / "config.toml"

	with config_path.open("rb") as config_file:
		config = tomllib.load(config_file)
	dataset_path = Path(config["preparation"]["folder_dataset"])
	if not dataset_path.is_absolute():
		dataset_path = project_path / dataset_path

	checkpoint_path = "../models/table7_pidinet.pth"
	checkpoint_path = Path(r"D:\Data\Outcrops\models\save_models\checkpoint_000.pth")

	model = create_pidinet_adapter(checkpoint_path)
	model = Cropper(model, crop=512, pad=64)

	for folder_path in dataset_path.iterdir():
		storage = Storage.from_folder_path(folder_path)
		image = storage.load_image()
		edges = model(image)
		storage.save_grayscale(edges, suffix="_edges")


def create_pidinet_adapter(checkpoint_path):
	module = PiDiNet(60, config_model("carv4"), dil=24, sa=True)
	module = torch.nn.DataParallel(module).cuda()
	checkpoint = torch.load(checkpoint_path, map_location="cuda")
	module.load_state_dict(checkpoint["state_dict"])
	return NumpyImagenetAdapter(
		module,
		output_selector=lambda outputs: outputs[-1],
	)

if __name__ == "__main__":
	main()
