"""Find and randomly color closed 4-connected regions in an edge image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Настройки запуска.
INPUT_IMAGE_PATH = Path(
    r"D:\Data\Outcrops\handmark\IMGP3353-3355\pidinet\edges_thin"
    r"\IMGP3353-3355_bsds500.png"
)
OUTPUT_IMAGE_PATH = PROJECT_ROOT / "IMGP3353-3355_bsds500_closed_4_connected.png"
THRESHOLD: int | None = None  # None — определить автоматически методом Otsu.
RANDOM_SEED: int | None = None  # Укажите число для повторяемых цветов.


def find_closed_regions(
    grayscale: np.ndarray,
    threshold: int | None = None,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Return component labels, closed label IDs, and the binary edge mask.

    Pixels brighter than ``threshold`` are treated as edges. If no threshold is
    supplied, Otsu's method selects it automatically. A background component is
    closed when it does not touch any image border. Background components are
    labelled using 4-connectivity.
    """
    if grayscale.ndim != 2:
        raise ValueError("Ожидалось одноканальное изображение.")
    if grayscale.size == 0:
        raise ValueError("Изображение пустое.")

    if threshold is None:
        _, edge_mask_u8 = cv2.threshold(
            grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
    else:
        if not 0 <= threshold <= 255:
            raise ValueError("Порог должен находиться в диапазоне от 0 до 255.")
        _, edge_mask_u8 = cv2.threshold(
            grayscale, threshold, 255, cv2.THRESH_BINARY
        )

    edge_mask = edge_mask_u8 != 0
    background = (~edge_mask).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(background, connectivity=4)

    border_labels = np.unique(
        np.concatenate(
            (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
        )
    )
    outside_labels = set(border_labels.tolist())
    closed_labels = [
        label for label in range(1, component_count) if label not in outside_labels
    ]
    return labels, closed_labels, edge_mask


def color_regions(
    labels: np.ndarray,
    closed_labels: list[int],
    edge_mask: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Create a BGR image with white edges and random closed-region colors."""
    rng = np.random.default_rng(seed)
    color_table = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)

    if closed_labels:
        closed_label_array = np.asarray(closed_labels, dtype=np.intp)
        # Exclude very dark colors so every detected region remains visible.
        color_table[closed_label_array] = rng.integers(
            40,
            256,
            size=(len(closed_labels), 3),
            dtype=np.uint8,
        )

    result = color_table[labels]
    result[edge_mask] = (255, 255, 255)

    return result


def main() -> None:
    grayscale = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise FileNotFoundError(
            f"Не удалось загрузить изображение: {INPUT_IMAGE_PATH}"
        )

    labels, closed_labels, edge_mask = find_closed_regions(
        grayscale, THRESHOLD
    )
    result = color_regions(labels, closed_labels, edge_mask, RANDOM_SEED)

    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_IMAGE_PATH), result):
        raise OSError(f"Не удалось сохранить изображение: {OUTPUT_IMAGE_PATH}")

    print(f"Найдено замкнутых областей: {len(closed_labels)}")
    print(f"Результат сохранён: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
