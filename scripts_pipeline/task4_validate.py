from pathlib import Path
import tomllib

import numpy as np
import torch

from pygradskeleton import couprie
from rockedgesdetectors import Cropper
from rocknetmanager.manager_shapefile import label_load, mask_load
from rocknetmanager.metrics import boundary_f_score, panoptic_quality
from scripts_pipeline.task1_edge_detect_ddn import load_model as create_ddn_adapter
from scripts_pipeline.task1_edges_detect_pidi import create_pidinet_adapter
from storage_manager import Storage
from storage_manager.image_formatter import uint8_normalize


# Настройки
MODEL_TYPE = "ddn"  # "pidinet" или "ddn"
MODEL_NAME = "bsds500"  # PiDiNet: "64", "128", "192", "bsds500"; DDN: "bsds500", "outcrop_1", ...
CHECKPOINT_NUMBER = 1  # Только для PiDiNet, кроме "bsds500"
PROGRESS = True  # Показывать прогресс обработки кропов и скелетизации

CROP_SIZE = 350
PAD_SIZE = 150
SKELETON_LAM = 0
SKELETON_THRESHOLD = 128
F_SCORE_TOLERANCE_PX = 3


def get_checkpoint_path(project_path: Path) -> Path:
    if MODEL_TYPE == "ddn":
        return project_path / "models" / "models_ddn" / f"ddn_{MODEL_NAME}.pth"
    if MODEL_TYPE != "pidinet":
        raise ValueError(f"Неизвестный тип модели: {MODEL_TYPE}")

    model_dir = project_path / "models" / "models_pidinet"
    if MODEL_NAME == "bsds500":
        return model_dir / "pidinet_bsds500.pth"
    if MODEL_NAME not in {"64", "128", "192"}:
        raise ValueError(f"Неизвестная модель: {MODEL_NAME}")
    return (
        model_dir
        / MODEL_NAME
        / f"checkpoint_{CHECKPOINT_NUMBER:03d}_{MODEL_NAME}.pth"
    )

def main(*, use_pq: bool = False) -> None:
    print("MODEL_TYPE", MODEL_TYPE, "MODEL_NAME", MODEL_NAME, "CHECKPOINT", CHECKPOINT_NUMBER)
    project_path = Path(__file__).resolve().parents[1]
    with (project_path / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    folder_validation = Path(config["validation"]["folder_validation"])
    if not folder_validation.is_absolute():
        folder_validation = project_path / folder_validation

    checkpoint_path = get_checkpoint_path(project_path)
    if MODEL_TYPE == "ddn":
        adapter = create_ddn_adapter(checkpoint_path)
    else:
        adapter = create_pidinet_adapter(checkpoint_path)
    adapter.eval()
    model = Cropper(adapter, crop=CROP_SIZE, pad=PAD_SIZE, display=PROGRESS)

    scores: list[float] = []
    folders = sorted(path for path in folder_validation.iterdir() if path.is_dir())

    for folder in folders:
        storage = Storage.from_folder_path(folder)
        image = storage.load_image()

        with torch.inference_mode():
            edges = uint8_normalize(model(image))

        edges_thin = couprie(
            edges,
            lam=SKELETON_LAM,
            threshold=SKELETON_THRESHOLD,
            progress=PROGRESS,
        )
        image_mask = mask_load(folder / "areas", edges_thin.shape)
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
