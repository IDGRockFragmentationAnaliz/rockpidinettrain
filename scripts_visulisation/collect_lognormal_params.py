"""Собрать параметры логнормальной ММП для всех примеров handmark в CSV."""

from __future__ import annotations

import csv

import cv2
import numpy as np

import distrebution as distribution
from rocknetmanager.manager_shapefile import label_load
from storage_manager import Storage


OUTPUT_CSV_PATH = distribution.PROJECT_ROOT / "handmark_lognormal_mle.csv"
FIELDNAMES = (
    "example",
    "predict_mu_ln",
    "predict_sigma_ln",
    "gt_mu_ln",
    "gt_sigma_ln",
    "predict_n",
    "gt_n",
    "min_x",
    "max_x",
)


def areas_for_fit(edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Применить тот же поиск и отбор объектов, что и в distrebution.py."""
    labels, closed_labels, _ = distribution.find_closed_regions(
        edges, distribution.THRESHOLD
    )
    areas = distribution.object_areas_inside_mask(labels, closed_labels, mask)
    return areas[
        (areas >= distribution.MIN_X) & (areas <= distribution.MAX_X)
    ]


def main() -> None:
    examples = sorted(
        path for path in distribution.HANDMARK_FOLDER.iterdir() if path.is_dir()
    )
    if not examples:
        raise ValueError(f"Нет примеров в {distribution.HANDMARK_FOLDER}")

    rows: list[dict[str, str | int]] = []
    for folder in examples:
        name = folder.name
        edge_path = folder / "rcf" / "edges_thin" / f"{name}_50_16.png"
        edges = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        if edges is None:
            raise FileNotFoundError(f"Не удалось загрузить карту границ: {edge_path}")
        shape = edges.shape

        mask = Storage.from_folder_path(folder).load_mask(shape=shape)
        predict_areas = areas_for_fit(edges, mask)
        del edges

        gt_path = folder / "traces_gt" / "traces.shp"
        gt_edges = label_load(path=gt_path, shape=shape, thickness=1)
        gt_areas = areas_for_fit(gt_edges, mask)
        del gt_edges, mask

        predict_mu, predict_sigma = distribution.fit_lognormal_mle(predict_areas)
        gt_mu, gt_sigma = distribution.fit_lognormal_mle(gt_areas)
        rows.append({
            "example": name,
            "predict_mu_ln": f"{predict_mu:.8f}",
            "predict_sigma_ln": f"{predict_sigma:.8f}",
            "gt_mu_ln": f"{gt_mu:.8f}",
            "gt_sigma_ln": f"{gt_sigma:.8f}",
            "predict_n": int(predict_areas.size),
            "gt_n": int(gt_areas.size),
            "min_x": distribution.MIN_X,
            "max_x": distribution.MAX_X,
        })
        print(f"{name}: predict n={predict_areas.size}, GT n={gt_areas.size}")

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сохранено {len(rows)} примеров: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
