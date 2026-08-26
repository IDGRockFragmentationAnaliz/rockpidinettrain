import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pygradskeleton import couprie
from storage_manager import Storage


def main():
	dataset_path = Path(r"D:\Data\Outcrops\unmark")
	for folder_path in dataset_path.iterdir():
		storage = Storage.from_folder_path(folder_path)
		edges = storage.load_grayscale(suffix="_edges_original")
		edges_thin = couprie(edges, lam=5, threshold=128, progress=True)
		storage.save_grayscale(edges_thin, suffix="_edges_thin_original")


if __name__ == "__main__":
	main()
