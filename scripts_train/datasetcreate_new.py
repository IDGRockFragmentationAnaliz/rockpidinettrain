import csv
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from rocknetmanager import Sample, Tiler, save_tile
from rocknetmanager.sample_transform import Rotate, horizontal_flip


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"


def resolve_config_path(value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else PROJECT_ROOT / path


with CONFIG_PATH.open("rb") as config_file:
	PATH_CONFIG = tomllib.load(config_file)

DATASET_CONFIG = PATH_CONFIG["dataset"]

RAW_MANIFEST_PATHS = tuple(
	resolve_config_path(value)
	for value in DATASET_CONFIG["raw_manifest_paths"]
)
OUTPUT_FOLDER = resolve_config_path(DATASET_CONFIG["output_folder"])
OUTPUT_LST_PATH = resolve_config_path(DATASET_CONFIG["lst_path"])

TILE_SIZE = (512, 512)
TILE_STRIDE = (337, 337)
ROTATION_ANGLES = (0, 90, 180, 270)
INCLUDE_HORIZONTAL_FLIP = True

LABEL_THICKNESS = 1
MIN_MASK_FRACTION = 0.4
MIN_LABEL_PIXELS = 5


@dataclass(frozen=True)
class SamplePaths:
	name: str
	image: Path
	label: Path
	mask: Path

	def as_dict(self) -> dict[str, Path]:
		return {
			"image": self.image,
			"label": self.label,
			"mask": self.mask,
		}


def main() -> None:
	samples = []

	for manifest_path in RAW_MANIFEST_PATHS:
		samples.extend(load_manifest(manifest_path))

	saved_count, skipped_count = create_dataset(
		samples=samples,
		output_folder=OUTPUT_FOLDER,
		output_lst_path=OUTPUT_LST_PATH,
		tile_size=TILE_SIZE,
		tile_stride=TILE_STRIDE,
		rotation_angles=ROTATION_ANGLES,
		include_horizontal_flip=INCLUDE_HORIZONTAL_FLIP,
		label_thickness=LABEL_THICKNESS,
		min_mask_fraction=MIN_MASK_FRACTION,
		min_label_pixels=MIN_LABEL_PIXELS,
	)

	print(f"Обработано исходных сэмплов: {len(samples)}")
	print(
		"Сохранено тайлов с учётом "
		f"поворотов и отражений: {saved_count}"
	)
	print(f"Пропущено пустых или недоступных тайлов: {skipped_count}")
	print(f"Датасет сохранён в: {OUTPUT_FOLDER}")
	print(f"Список датасета сохранён в: {OUTPUT_LST_PATH}")


def load_manifest(
	manifest_path: Path,
	root_path: Path | None = None,
) -> list[SamplePaths]:
	manifest_path = Path(manifest_path).resolve()

	if not manifest_path.is_file():
		raise FileNotFoundError(f"Манифест не найден: {manifest_path}")

	root_path = Path(
		manifest_path.parent if root_path is None else root_path
	).resolve()

	samples = []

	with manifest_path.open(
		"r",
		encoding="utf-8-sig",
		newline="",
	) as manifest_file:
		reader = csv.reader(manifest_file, delimiter="\t")

		for line_number, row in enumerate(reader, start=1):
			if not row or all(not value.strip() for value in row):
				continue

			if len(row) != 3:
				raise ValueError(
					f"В {manifest_path}, строка {line_number}: "
					f"ожидалось 3 колонки image/label/mask, "
					f"получено {len(row)}"
				)

			image_value, label_value, mask_value = (
				value.strip() for value in row
			)

			if not image_value or not label_value or not mask_value:
				raise ValueError(
					f"В {manifest_path}, строка {line_number}: "
					"пути image, label и mask должны быть заданы"
				)

			image_path = resolve_path(root_path, image_value)
			label_path = resolve_path(root_path, label_value)
			mask_path = resolve_path(root_path, mask_value)

			if not image_path.is_file():
				raise FileNotFoundError(
					f"Image из строки {line_number} не найден: {image_path}"
				)

			for field_name, path in (
				("label", label_path),
				("mask", mask_path),
			):
				if not path.exists():
					raise FileNotFoundError(
						f"{field_name} из строки {line_number} "
						f"не найден: {path}"
					)

			sample_name = (
				f"{manifest_path.stem}_"
				f"{line_number:06d}_{image_path.stem}"
			)

			samples.append(SamplePaths(
				name=sample_name,
				image=image_path,
				label=label_path,
				mask=mask_path,
			))

	return samples


def resolve_path(root_path: Path, value: str) -> Path:
	path = Path(value)

	if not path.is_absolute():
		path = root_path / path

	return path.resolve()


def create_dataset(
	samples: Sequence[SamplePaths],
	output_folder: Path,
	output_lst_path: Path,
	tile_size: tuple[int, int],
	tile_stride: tuple[int, int],
	rotation_angles: tuple[int, ...] = (0,),
	include_horizontal_flip: bool = True,
	label_thickness: int = 1,
	min_mask_fraction: float = 0.4,
	min_label_pixels: int = 5,
) -> tuple[int, int]:
	validate_build_parameters(
		tile_size=tile_size,
		rotation_angles=rotation_angles,
		label_thickness=label_thickness,
		min_mask_fraction=min_mask_fraction,
		min_label_pixels=min_label_pixels,
	)

	output_folder = Path(output_folder)
	output_folder.mkdir(parents=True, exist_ok=True)
	output_lst_path = Path(output_lst_path).resolve()
	output_lst_path.parent.mkdir(parents=True, exist_ok=True)

	tiler = Tiler(
		size=tile_size,
		stride=tile_stride,
	)
	saved_count = 0
	skipped_count = 0

	with output_lst_path.open(
		"w",
		encoding="utf-8",
		newline="",
		buffering=1,
	) as output_lst:
		lst_writer = csv.writer(
			output_lst,
			delimiter="\t",
			lineterminator="\n",
		)

		progress = tqdm(
			samples,
			desc="Подготовка датасета",
			unit="sample",
			dynamic_ncols=True,
		)

		for sample_paths in progress:
			progress.set_postfix_str(sample_paths.image.stem)
			sample_output_folder = (
				output_folder / sample_paths.image.parent.name
			)

			sample = Sample.load_sample(
				sample_paths.as_dict(),
				thickness=label_thickness,
			)
			sample["image"].masked_fill_(
				sample["mask"] == 0,
				0,
			)

			for transformed, angle, is_flipped in iter_sample_variants(
				sample=sample,
				rotation_angles=rotation_angles,
				include_horizontal_flip=include_horizontal_flip,
			):
				for tile, top, left in tiler.iter_with_coordinates(transformed):
					if not is_accessible_tile(
						tile,
						min_mask_fraction=min_mask_fraction,
						min_label_pixels=min_label_pixels,
					):
						skipped_count += 1
						continue

					tile_name = (
						f"{sample_paths.name}_"
						f"r{angle:03d}_f{int(is_flipped)}_"
						f"y{top:05d}_x{left:05d}"
					)

					image_path, label_path = save_tile(
						tile=tile,
						output_folder=sample_output_folder,
						tile_name=tile_name,
					)
					lst_writer.writerow((
						image_path.resolve().as_posix(),
						label_path.resolve().as_posix(),
					))
					saved_count += 1

	return saved_count, skipped_count


def iter_sample_variants(
	sample: Sample,
	rotation_angles: tuple[int, ...],
	include_horizontal_flip: bool,
) -> Iterator[tuple[Sample, int, bool]]:
	"""
	Сначала создаёт ориентации всего Sample.

	Тайлинг выполняется уже над результатами этого
	генератора. При include_horizontal_flip=True для
	каждого угла возвращаются обычная и горизонтально
	отражённая версии.
	"""
	for angle in rotation_angles:
		rotated = sample if angle == 0 else Rotate(angle)(sample)

		yield rotated, angle, False

		if include_horizontal_flip:
			yield horizontal_flip(rotated), angle, True


def validate_build_parameters(
	tile_size: tuple[int, int],
	rotation_angles: tuple[int, ...],
	label_thickness: int,
	min_mask_fraction: float,
	min_label_pixels: int,
) -> None:
	if not rotation_angles:
		raise ValueError("Нужно задать хотя бы один угол поворота")

	if len(set(rotation_angles)) != len(rotation_angles):
		raise ValueError(
			f"Углы поворота не должны повторяться: {rotation_angles}"
		)

	if label_thickness <= 0:
		raise ValueError("label_thickness должен быть положительным")

	if not 0 <= min_mask_fraction <= 1:
		raise ValueError("min_mask_fraction должен находиться в [0, 1]")

	if min_label_pixels < 0:
		raise ValueError("min_label_pixels должен быть неотрицательным")


def is_accessible_tile(
	tile: Sample,
	min_mask_fraction: float,
	min_label_pixels: int,
) -> bool:
	mask = tile["mask"]
	label = tile["label"]

	mask_fraction = (
		torch.count_nonzero(mask).item()
		/ mask.numel()
	)
	label_pixels = torch.count_nonzero(label).item()

	return (
		mask_fraction > min_mask_fraction
		and label_pixels > min_label_pixels
	)


if __name__ == "__main__":
	main()
