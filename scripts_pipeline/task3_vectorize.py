"""Векторизация сохранённых скелетов из валидационных примеров."""

from __future__ import annotations

import tomllib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skelet_vectorize.extract_lines import extract_lines
from storage_manager import Storage
from storage_manager.tools import save_shplines


THIN_EDGES_SUBFOLDER = "rcf/edges_thin"
THIN_EDGES_SUFFIX = "_50_16"  # Тот же тип файлов, что в objects_visulise.py.
OUTPUT_SUBFOLDER = "edges_predict"
SIMPLIFY_TOLERANCE = 1


def vectorize_folder(folder_path: Path) -> int:
    """Сохранить полилинии в <папка примера>/edges_predict/."""
    storage = Storage.from_folder_path(folder_path)
    image_thin = storage.load_grayscale(
        suffix=THIN_EDGES_SUFFIX,
        subfolder=THIN_EDGES_SUBFOLDER,
    )
    polylines = extract_lines(image_thin, simplify_tol=SIMPLIFY_TOLERANCE)

    output_folder = folder_path / OUTPUT_SUBFOLDER
    save_shplines(output_folder, polylines)
    print(f"{folder_path.name}: {len(polylines)} линий -> {output_folder}")
    return len(polylines)


def main() -> None:
    with (PROJECT_ROOT / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    validation_folder = Path(config["validation"]["folder_validation"])
    if not validation_folder.is_absolute():
        validation_folder = PROJECT_ROOT / validation_folder
    if not validation_folder.is_dir():
        raise NotADirectoryError(f"Папка валидации не найдена: {validation_folder}")

    folders = sorted(path for path in validation_folder.iterdir() if path.is_dir())
    if not folders:
        raise ValueError(f"Нет примеров для векторизации: {validation_folder}")

    total = 0
    for folder_path in folders:
        total += vectorize_folder(folder_path)
    print(f"Всего: {len(folders)} примеров, {total} линий")


if __name__ == "__main__":
    main()
