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
		edges_thin = storage.load_grayscale(suffix="_edges_4", subfolder="ddn/bsds500/edges_thin")
		mask = storage.load_mask(shape=edges_thin.shape)
		# mask — uint8 (0/255); сравнение создаёт булеву маску для индексации.
		edges_thin[mask == 0] = 255
		# Все белые пиксели имеют расстояние 0, фон — расстояние до ближайшего белого.
		white_pixels = edges_thin > 128
		background = (~white_pixels).astype(np.uint8)
		distance_transform = cv2.distanceTransform(
			background, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
		)
		positive_distances = distance_transform[distance_transform > 0]
		mean_distance = float(positive_distances.mean()) if positive_distances.size else 0.0
		print(f"{folder_path.name}: среднее расстояние = {mean_distance:.3f} px")
		continue


if __name__ == "__main__":
	main()
