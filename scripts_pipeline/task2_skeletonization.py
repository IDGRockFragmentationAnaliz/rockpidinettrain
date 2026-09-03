import cv2
from pathlib import Path
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from pygradskeleton import couprie
from storage_manager import Storage


def main():
	project_path = Path(__file__).resolve().parents[1]
	config_path = project_path / "config.toml"

	with config_path.open("rb") as config_file:
		config = tomllib.load(config_file)
	# dataset_path = Path(config["preparation"]["folder_dataset"])
	dataset_path = Path(config["validation"]["folder_validation"])

	for folder_path in dataset_path.iterdir():
		storage = Storage.from_folder_path(folder_path)
		edges = storage.load_grayscale(suffix="_edges_ddn12")
		edges_thin = couprie(edges, lam=50, threshold=128, progress=True)
		storage.save_grayscale(edges_thin, suffix="_thin_edges")


if __name__ == "__main__":
	main()
