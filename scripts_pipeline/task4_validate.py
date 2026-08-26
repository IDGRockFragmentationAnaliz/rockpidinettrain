from pathlib import Path

import matplotlib.pyplot as plt
from pygradskeleton import couprie
from storage_manager import Storage
from rocknetmanager.manager_shapefile import label_load, mask_load
from rocknetmanager.metrics import boundary_f_score

def main():
	folder_path = Path(r"D:\Data\Outcrops\handmark\IMGP0146")
	edges_gt_path = Path(r"D:\Data\Outcrops\handmark\IMGP0146\traces_gt\traces.shp")
	mask_path = Path(r"D:\Data\Outcrops\handmark\IMGP0146\areas")

	storage = Storage.from_folder_path(folder_path)
	image_edges_thin = storage.load_thin_edges()
	image_mask = mask_load(mask_path, image_edges_thin.shape)
	image_edges_gt = label_load(
		path=edges_gt_path,
		shape=image_edges_thin.shape,
		thickness=1
	)
	image_edges_gt[image_mask == 0] = 0
	image_edges_thin[image_mask == 0] = 0
	f_score = boundary_f_score(edges_pred=image_edges_thin, edges_gt=image_edges_gt, tolerance_px=3)
	print(f_score)

	fig = plt.figure(figsize=(14, 9))
	axs = [fig.add_subplot(1, 2, 1),fig.add_subplot(1, 2, 2)]
	axs[0].imshow(image_edges_thin)
	axs[1].imshow(image_edges_gt)
	plt.show()


if __name__ == "__main__":
	main()
