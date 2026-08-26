import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn

from rockedgesdetectors import NumpyImagenetAdapter
from rockedgesdetectors.pidinet.config import config_model
from rockedgesdetectors.pidinet.models import PiDiNet
from rocknetmanager.dataset import Dataset
from rocknetmanager.save_checkpoint import save_checkpoint
from rocknetmanager.train import ModelTrain


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_LST_PATH = Path(r"D:\Data\train_data\train.lst")
INITIAL_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "table7_pidinet.pth"
CHECKPOINT_FOLDER = PROJECT_ROOT / "save_models"
TEST_IMAGE_FOLDER = PROJECT_ROOT / "test_images"
TEST_OUTPUT_FOLDER = PROJECT_ROOT / "train_test"

MODEL_CONFIG = "carv4"
EPOCHS = 99
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4


def main() -> None:
	if not torch.cuda.is_available():
		raise RuntimeError("Для текущего ModelTrain требуется CUDA")

	dataset = Dataset(path_lst=DATASET_LST_PATH)

	seed = int(time.time())
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	model = load_pidinet(
		checkpoint_path=INITIAL_CHECKPOINT_PATH,
		device=torch.device("cuda"),
	)
	optimizer = create_optimizer(model.module)

	preview_model = NumpyImagenetAdapter(
		model,
		output_selector=lambda outputs: outputs[-1],
	)

	for image_path in iter_test_images(TEST_IMAGE_FOLDER):
		save_image_test(
			model=preview_model,
			image_path=image_path,
			save_folder=TEST_OUTPUT_FOLDER,
		)

	trainer = ModelTrain(
		dataset=dataset,
		model=model,
		optimizer=optimizer,
	)

	for epoch in range(1, EPOCHS + 1):
		trainer.train()

		save_checkpoint(
			{
				"epoch": epoch,
				"state_dict": model.state_dict(),
				"optimizer": optimizer.state_dict(),
			},
			epoch,
			CHECKPOINT_FOLDER,
		)

		epoch_folder = TEST_OUTPUT_FOLDER / f"epoch_{epoch}"
		for image_path in iter_test_images(TEST_IMAGE_FOLDER):
			save_image_test(
				model=preview_model,
				image_path=image_path,
				save_folder=epoch_folder,
			)


def load_pidinet(
	checkpoint_path: Path,
	device: torch.device,
) -> nn.DataParallel:
	"""Создаёт конкретный PiDiNet и загружает его checkpoint."""
	module = PiDiNet(
		60,
		config_model(MODEL_CONFIG),
		dil=24,
		sa=True,
	)
	model = nn.DataParallel(module).to(device)

	checkpoint = torch.load(
		checkpoint_path,
		map_location=device,
		weights_only=True,
	)
	model.load_state_dict(checkpoint["state_dict"])

	return model


def create_optimizer(module: PiDiNet) -> torch.optim.Optimizer:
	conv_weights, bn_weights, relu_weights = module.get_weights()

	param_groups = [
		{
			"params": conv_weights,
			"weight_decay": WEIGHT_DECAY,
			"lr": LEARNING_RATE,
		},
		{
			"params": bn_weights,
			"weight_decay": 0.1 * WEIGHT_DECAY,
			"lr": LEARNING_RATE,
		},
		{
			"params": relu_weights,
			"weight_decay": 0.0,
			"lr": LEARNING_RATE,
		},
	]

	return torch.optim.Adam(
		param_groups,
		betas=(0.9, 0.99),
	)


def iter_test_images(folder: Path):
	if not folder.is_dir():
		return

	yield from sorted(
		path
		for path in folder.iterdir()
		if path.is_file()
	)


def save_image_test(
	model: NumpyImagenetAdapter,
	image_path: Path,
	save_folder: Path,
) -> None:
	save_folder.mkdir(parents=True, exist_ok=True)

	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(
			f"Не удалось загрузить тестовое изображение: {image_path}"
		)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	model.eval()
	with torch.inference_mode():
		result = model(image)

	figure, axes = plt.subplots(1, 2, figsize=(7, 4))
	axes[0].imshow(result, cmap="gray")
	axes[0].set_title("PiDiNet")
	axes[1].imshow(image)
	axes[1].set_title("Image")

	for axis in axes:
		axis.axis("off")

	figure.tight_layout()
	figure.savefig(save_folder / image_path.name)
	plt.close(figure)


if __name__ == "__main__":
	main()
