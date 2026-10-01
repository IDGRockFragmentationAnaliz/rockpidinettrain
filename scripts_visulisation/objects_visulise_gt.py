"""Раскрасить замкнутые объекты Ground Truth внутри полигональной маски."""

from __future__ import annotations

import cv2

from objects_visulise import (
    INPUT_IMAGE_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_FOLDER,
    THRESHOLD,
    color_regions,
    find_closed_regions,
    regions_inside_mask,
)
from rocknetmanager.manager_shapefile import label_load
from storage_manager import Storage


GT_PATH = SAMPLE_FOLDER / "traces_gt" / "traces.shp"
OUTPUT_IMAGE_PATH = PROJECT_ROOT / f"{INPUT_IMAGE_PATH.stem}_gt_objects.png"


def main() -> None:
    # Размер берём из той же карты, что использует objects_visulise.py.
    image = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {INPUT_IMAGE_PATH}")
    shape = image.shape
    del image

    storage = Storage.from_folder_path(SAMPLE_FOLDER)
    mask = storage.load_mask(shape=shape)
    edges_gt = label_load(path=GT_PATH, shape=shape, thickness=1)
    labels, closed_labels, edge_mask = find_closed_regions(edges_gt, THRESHOLD)
    kept_labels = regions_inside_mask(labels, closed_labels, mask)

    result = color_regions(labels, kept_labels, edge_mask, RANDOM_SEED)
    result[mask == 0] = (0, 0, 0)

    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_IMAGE_PATH), result):
        raise OSError(f"Не удалось сохранить изображение: {OUTPUT_IMAGE_PATH}")

    print(f"Всего замкнутых областей Ground Truth: {len(closed_labels)}")
    print(f"Полностью внутри маски: {len(kept_labels)}")
    print(f"Результат сохранён: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
