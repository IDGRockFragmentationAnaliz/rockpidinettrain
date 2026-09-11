"""Валидация сохранённых карт границ без загрузки сети и инференса."""

from pathlib import Path
import tomllib

import numpy as np

from pygradskeleton import couprie
from rocknetmanager.manager_shapefile import label_load, mask_load
from rocknetmanager.metrics import boundary_f_score, panoptic_quality
from storage_manager import Storage


# Настройки загружаемой карты
EDGES_SUFFIX = "_edges_bsds500"
EDGES_SUBFOLDER = "pidinet"
EDGES_EXTENSION = "png"
MASK_SUBFOLDER = "areas"

PROGRESS = True
USE_PQ = False  # False — boundary F-score; True — panoptic quality

SKELETON_LAM = 0
SKELETON_THRESHOLD = 128
F_SCORE_TOLERANCE_PX = 3


def main(*, use_pq: bool = USE_PQ) -> None:
    print(
        "EDGES_SUBFOLDER", EDGES_SUBFOLDER,
        "EDGES_SUFFIX", EDGES_SUFFIX,
        "EDGES_EXTENSION", EDGES_EXTENSION,
    )
    project_path = Path(__file__).resolve().parents[1]
    with (project_path / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    folder_validation = Path(config["validation"]["folder_validation"])
    if not folder_validation.is_absolute():
        folder_validation = project_path / folder_validation
    folders = sorted(path for path in folder_validation.iterdir() if path.is_dir())
    if not folders:
        raise ValueError(f"Нет примеров для валидации: {folder_validation}")

    scores: list[float] = []
    for folder in folders:
        storage = Storage.from_folder_path(folder)
        edges = storage.load_grayscale(
            suffix=EDGES_SUFFIX,
            ext=EDGES_EXTENSION,
            subfolder=EDGES_SUBFOLDER,
        )
        edges_thin = couprie(
            edges,
            lam=SKELETON_LAM,
            threshold=SKELETON_THRESHOLD,
            progress=PROGRESS,
        )
        image_mask = mask_load(folder / MASK_SUBFOLDER, edges_thin.shape)
        edges_gt = label_load(
            path=folder / "traces_gt" / "traces.shp",
            shape=edges_thin.shape,
            thickness=1,
        )

        mask_value = 255 if use_pq else 0
        edges_thin[image_mask == 0] = mask_value
        edges_gt[image_mask == 0] = mask_value

        if use_pq:
            score = panoptic_quality(pred=edges_thin, gt=edges_gt)
        else:
            score = boundary_f_score(
                edges_pred=edges_thin,
                edges_gt=edges_gt,
                tolerance_px=F_SCORE_TOLERANCE_PX,
            )
        scores.append(score)
        print(f"{folder.name}: {score:.9f}")

    print(f"Среднее: {float(np.mean(scores)):.9f}")


if __name__ == "__main__":
    main()
