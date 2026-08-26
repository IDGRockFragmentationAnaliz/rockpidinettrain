import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pygradskeleton import couprie
from storage_manager import Storage


def main():
	folder_path = Path(r"D:\Data\Outcrops\handmark\IMGP0146")
	storage = Storage.from_folder_path(folder_path)
	edges = storage.load_edges()
	edges_thin = couprie(edges, lam=20, threshold=128, progress=True)
	storage.save_thin_edges(edges_thin)


if __name__ == "__main__":
	main()
