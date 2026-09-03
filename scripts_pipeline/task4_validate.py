from pathlib import Path
import tomllib
import matplotlib.pyplot as plt
from pygradskeleton import couprie
from storage_manager import Storage
from rocknetmanager.manager_shapefile import label_load, mask_load
from rocknetmanager.metrics import boundary_f_score

def main():
	project_path = Path(__file__).resolve().parents[1]
	config_path = project_path / "config.toml"
	with config_path.open("rb") as config_file:
		config = tomllib.load(config_file)

	folder_validation = Path(config["validation"]["validation_folder"])

	folder_instance = folder_validation / "IMGP0146"
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

	fig = plt.figure(figsize=(14, 9))
	axs = [fig.add_subplot(1, 2, 1),fig.add_subplot(1, 2, 2)]
	axs[0].imshow(image_edges_thin)
	axs[1].imshow(image_edges_gt)
	plt.show()


if __name__ == "__main__":
	main()
