from pathlib import Path

import torch

from rockedgesdetectors import Cropper, ModelRCF, NumpyImagenetAdapter
from rockedgesdetectors.pidinet.config import config_model
from rockedgesdetectors.pidinet.models import PiDiNet
from storage_manager import Storage
import matplotlib.pyplot as plt



def main():
	checkpoint_path = "../models/table7_pidinet.pth"
	# image_path = Path("../test_images/test_01.png")

	image_path = Path(r"D:\Data\Outcrops\handmark\IMGP0146\IMGP0146.png")
	# storage = Storage.from_folder_path(image_path)
	storage = Storage.from_image_path(image_path)

	image = storage.load_image()
	model = create_pidinet_adapter(checkpoint_path)
	model = Cropper(model, crop=512, pad=64)
	edges = model(image)

	fig = plt.figure(figsize=(14, 9))
	axs = [fig.add_subplot(1, 2, 1),fig.add_subplot(1, 2, 2)]
	axs[0].imshow(image)
	axs[1].imshow(edges)
	plt.show()

	#storage.save_edges(edges)


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
