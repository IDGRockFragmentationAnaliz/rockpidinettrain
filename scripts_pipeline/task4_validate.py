from pathlib import Path
import tomllib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pygradskeleton import couprie
from storage_manager import Storage
from rocknetmanager.manager_shapefile import label_load, mask_load
from rocknetmanager.metrics import boundary_f_score

def main():
	project_path = Path(__file__).resolve().parents[1]
	config_path = project_path / "config.toml"
	with config_path.open("rb") as config_file:
		config = tomllib.load(config_file)

	folder_validation = Path(config["validation"]["folder_validation"])

	folder_instance = folder_validation / "IMGP3286"
	edges_gt_path = folder_instance / Path(r"traces_gt\traces.shp")
	mask_path = folder_instance / Path(r"areas")

	storage = Storage.from_folder_path(folder_instance)
	image_edges_thin = storage.load_thin_edges()
	image_mask = mask_load(mask_path, image_edges_thin.shape)
	image_edges_gt = label_load(
		path=edges_gt_path,
		shape=image_edges_thin.shape,
		thickness=1
	)
	# image_edges_gt[image_mask == 0] = 0
	# image_edges_thin[image_mask == 0] = 0
	# f_score = boundary_f_score(edges_pred=image_edges_thin, edges_gt=image_edges_gt, tolerance_px=3)
	# print(f_score)

	test1, test2 = panoptic_quality(edges_pred=image_edges_thin, edges_gt=image_edges_gt)

	fig = plt.figure(figsize=(14, 9))
	axs = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
	axs[0].imshow(test1)
	axs[1].imshow(test2)
	axs[1].sharex(axs[0])
	axs[1].sharey(axs[0])
	plt.show()

	exit()

	fig = plt.figure(figsize=(14, 9))
	axs = [fig.add_subplot(1, 2, 1),fig.add_subplot(1, 2, 2)]
	axs[0].imshow(image_edges_thin)
	axs[1].imshow(image_edges_gt)
	plt.show()

def panoptic_quality(
	edges_pred: np.ndarray,
	edges_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Выделяет замкнутые области внутри предсказанных и эталонных границ.

	В выходных масках фон и сами границы имеют значение 0, а каждая
	замкнутая область получает отдельный положительный целочисленный ID.
	"""
	if edges_pred.ndim != 2 or edges_gt.ndim != 2:
		raise ValueError("Обе карты границ должны быть двумерными")

	if edges_pred.shape != edges_gt.shape:
		raise ValueError(
			f"Размеры карт границ не совпадают: "
			f"prediction={edges_pred.shape}, GT={edges_gt.shape}"
		)

	def closed_objects(edges: np.ndarray) -> np.ndarray:
		# Ненулевые пиксели считаются непроходимыми границами. 4-связность
		# не позволяет фону просачиваться через диагональное касание линий.
		background = (edges == 0).astype(np.uint8)
		component_count, components = cv2.connectedComponents(
			background,
			connectivity=4,
		)

		border_ids = np.unique(np.concatenate((
			components[0, :],
			components[-1, :],
			components[:, 0],
			components[:, -1],
		)))

		closed_component_ids = np.setdiff1d(
			np.arange(component_count),
			np.append(border_ids, 0),
		)

		object_ids_by_component = np.zeros(component_count, dtype=np.int32)
		object_ids_by_component[closed_component_ids] = np.arange(
			1,
			len(closed_component_ids) + 1,
		)

		return object_ids_by_component[components]

	return closed_objects(edges_pred), closed_objects(edges_gt)

if __name__ == "__main__":
	main()
