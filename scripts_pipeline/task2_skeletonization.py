import cv2
from pathlib import Path
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from pygradskeleton import couprie
from storage_manager import Storage

LAMBDA = 0
THRESHOLD = 32

EDGES_SUFFIX = "_bsds500"
EDGES_SUBFOLDER = "rcf/edges"

THIN_EDGES_SUFFIX =  "_bsds500"
THIN_EDGES_SUBFOLDER = "rcf/edges_thin"

def main():
	print("LAMBDA", LAMBDA)
	print("THRESHOLD", THRESHOLD)
	project_path = Path(__file__).resolve().parents[1]
	config_path = project_path / "config.toml"

	with config_path.open("rb") as config_file:
		config = tomllib.load(config_file)
	# dataset_path = Path(config["preparation"]["folder_dataset"])
	dataset_path = Path(config["validation"]["folder_validation"])

	for folder_path in dataset_path.iterdir():
		storage = Storage.from_folder_path(folder_path)
		edges = storage.load_grayscale(suffix=EDGES_SUFFIX, subfolder=EDGES_SUBFOLDER)
		edges_thin = couprie(edges, lam=LAMBDA, threshold=THRESHOLD, progress=True)
		storage.save_grayscale(edges_thin, suffix=THIN_EDGES_SUFFIX,subfolder=THIN_EDGES_SUBFOLDER)


if __name__ == "__main__":
	main()
