"""ECDF и гистограммы плотности площадей объектов внутри маски."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import truncnorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage_manager import Storage
from rocknetmanager.manager_shapefile import label_load
from objects_visulise import find_closed_regions


# Для другого примера достаточно изменить EXAMPLE_NAME.
HANDMARK_FOLDER = Path(r"D:\Data\Outcrops\handmark")
EXAMPLE_NAME = "IMGP3353-3355"
SAMPLE_FOLDER = HANDMARK_FOLDER / EXAMPLE_NAME
INPUT_IMAGE_PATH = (
    SAMPLE_FOLDER / "rcf" / "edges_thin" / f"{EXAMPLE_NAME}_50_16.png"
)
OUTPUT_IMAGE_PATH = PROJECT_ROOT / f"{INPUT_IMAGE_PATH.stem}_ecdf.png"
OUTPUT_PDF_IMAGE_PATH = PROJECT_ROOT / f"{INPUT_IMAGE_PATH.stem}_pdf.png"
THRESHOLD: int | None = None  # None — порог Otsu, как в objects_visulise.py.
GT_PATH = SAMPLE_FOLDER / "traces_gt" / "traces.shp"
MIN_X = 100
MAX_X = 100_000
N_BINS = 25  # Одинаковое число бинов для обоих наборов.


def object_areas_inside_mask(
    labels: np.ndarray, closed_labels: list[int], mask: np.ndarray
) -> np.ndarray:
    """Площади объектов, каждый пиксель которых лежит внутри маски."""
    if labels.shape != mask.shape:
        raise ValueError(
            f"Размер маски {mask.shape} не совпадает с изображением {labels.shape}"
        )
    if not closed_labels:
        return np.empty(0, dtype=np.int64)

    label_count = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=label_count)
    outside_counts = np.bincount(labels[mask == 0], minlength=label_count)
    ids = np.asarray(closed_labels, dtype=np.intp)
    return areas[ids[outside_counts[ids] == 0]]


def plot_ecdf(
    ax: plt.Axes,
    areas: np.ndarray,
    label: str,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
) -> None:
    """Добавить ECDF площадей на общий график."""
    if areas.size == 0:
        raise ValueError(f"Внутри маски не найдено замкнутых объектов: {label}")
    sizes, counts = np.unique(areas, return_counts=True)
    ax.step(
        sizes, np.cumsum(counts) / areas.size, where="post",
        label=f"{label}", color=color,
        linestyle=linestyle, linewidth=linewidth,
    )


def plot_pdf(
    ax: plt.Axes,
    areas: np.ndarray,
    bins: np.ndarray,
    label: str,
    *,
    linestyle: str,
    linewidth: float,
) -> None:
    """Построить гистограммную оценку плотности по общим границам бинов."""
    density, _ = np.histogram(areas, bins=bins, density=True)
    ax.stairs(
        density, bins, label=label, color="black",
        linestyle=linestyle, linewidth=linewidth,
    )


def fit_lognormal_mle(areas: np.ndarray) -> tuple[float, float]:
    """ММП для логнормального закона с усечением по [MIN_X, MAX_X]."""
    if areas.size < 2:
        raise ValueError("Для аппроксимации нужны хотя бы два объекта")

    log_areas = np.log(areas)
    log_min, log_max = np.log(MIN_X), np.log(MAX_X)
    span = log_max - log_min
    initial_sigma = max(float(np.std(log_areas)), 1e-3)

    def negative_log_likelihood(params: np.ndarray) -> float:
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        lower = (log_min - mu) / sigma
        upper = (log_max - mu) / sigma
        return float(-np.sum(truncnorm.logpdf(
            log_areas, lower, upper, loc=mu, scale=sigma
        )))

    fit = minimize(
        negative_log_likelihood,
        x0=(float(np.mean(log_areas)), np.log(initial_sigma)),
        method="L-BFGS-B",
        bounds=((log_min - 5 * span, log_max + 5 * span),
                (np.log(1e-3), np.log(10 * span))),
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Не удалось оценить параметры логнормального закона: {fit.message}")
    return float(fit.x[0]), float(np.exp(fit.x[1]))


def fitted_lognormal_curves(
    x: np.ndarray, mu: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """CDF и PDF логнормального закона, условного на [MIN_X, MAX_X]."""
    lower = (np.log(MIN_X) - mu) / sigma
    upper = (np.log(MAX_X) - mu) / sigma
    log_x = np.log(x)
    cdf = truncnorm.cdf(log_x, lower, upper, loc=mu, scale=sigma)
    pdf = truncnorm.pdf(log_x, lower, upper, loc=mu, scale=sigma) / x
    return cdf, pdf


def print_fit_summary(label: str, mu: float, sigma: float) -> None:
    """Показать параметры в основаниях e и 10 и максимум PDF по площади."""
    log10_factor = np.log(10)
    scale = np.exp(mu)
    mode = np.clip(np.exp(mu - sigma**2), MIN_X, MAX_X)
    print(
        f"ММП, {label}: mu_ln={mu:.4f}, sigma_ln={sigma:.4f}; "
        f"mu_log10={mu / log10_factor:.4f}, "
        f"sigma_log10={sigma / log10_factor:.4f}; "
        f"scale=e^mu={scale:.1f} px, mode_PDF={mode:.1f} px"
    )


def main() -> None:
    if MIN_X <= 0 or MAX_X <= MIN_X or N_BINS < 1:
        raise ValueError("Нужно задать 0 < MIN_X < MAX_X и N_BINS >= 1")

    grayscale = cv2.imread(str(INPUT_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {INPUT_IMAGE_PATH}")

    storage = Storage.from_folder_path(SAMPLE_FOLDER)
    mask = storage.load_mask(shape=grayscale.shape)
    labels, closed_labels, _ = find_closed_regions(grayscale, THRESHOLD)
    areas = object_areas_inside_mask(labels, closed_labels, mask)
    detected_inside_count = areas.size
    areas = areas[(areas >= MIN_X) & (areas <= MAX_X)]
    detected_count = len(closed_labels)
    del labels

    edges_gt = label_load(path=GT_PATH, shape=grayscale.shape, thickness=1)
    gt_labels, gt_closed_labels, _ = find_closed_regions(edges_gt, THRESHOLD)
    gt_areas = object_areas_inside_mask(gt_labels, gt_closed_labels, mask)
    gt_inside_count = gt_areas.size
    gt_areas = gt_areas[(gt_areas >= MIN_X) & (gt_areas <= MAX_X)]
    gt_count = len(gt_closed_labels)

    predicted_mu, predicted_sigma = fit_lognormal_mle(areas)
    gt_mu, gt_sigma = fit_lognormal_mle(gt_areas)
    fit_x = np.geomspace(MIN_X, MAX_X, 400)
    predicted_cdf, predicted_pdf = fitted_lognormal_curves(
        fit_x, predicted_mu, predicted_sigma
    )
    gt_cdf, gt_pdf = fitted_lognormal_curves(fit_x, gt_mu, gt_sigma)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_ecdf(ax, areas, "Найденные объекты", color="black",
              linestyle="-", linewidth=1.5)
    plot_ecdf(ax, gt_areas, "GT", color="black",
              linestyle="--", linewidth=3)
    ax.plot(fit_x, predicted_cdf, color="red", linestyle="-", linewidth=2,
            label="Логнормальная ММП: объекты")
    ax.plot(fit_x, gt_cdf, color="red", linestyle="--", linewidth=2.5,
            label="Логнормальная ММП: GT")
    ax.set(xlabel="Площадь объекта, пиксели", ylabel="Доля объектов (ECDF)")
    ax.set_xscale("log")
    ax.set_xlim([MIN_X, MAX_X])
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    bins = np.geomspace(MIN_X, MAX_X, N_BINS + 1)
    pdf_fig, pdf_ax = plt.subplots(figsize=(5, 5))
    plot_pdf(pdf_ax, areas, bins, "Predict",
             linestyle="-", linewidth=1.5)
    plot_pdf(pdf_ax, gt_areas, bins, "GT",
             linestyle="--", linewidth=3)
    pdf_ax.plot(fit_x, predicted_pdf, color="red", linestyle="-", linewidth=2)
    pdf_ax.plot(fit_x, gt_pdf, color="red", linestyle="--", linewidth=2.5)
    pdf_ax.set(xlabel="Площадь объекта, пиксели",
               ylabel="Плотность вероятности, 1/пиксель")
    pdf_ax.set_xscale("log")
    pdf_ax.set_xlim([MIN_X, MAX_X])
    pdf_ax.grid(alpha=0.3)
    pdf_ax.legend(loc="upper right")
    pdf_fig.tight_layout()

    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_IMAGE_PATH, dpi=600)
    pdf_fig.savefig(OUTPUT_PDF_IMAGE_PATH, dpi=600)
    plt.show()
    plt.close(fig)
    plt.close(pdf_fig)

    print(f"Найденные объекты: {areas.size} с площадью от {MIN_X} до {MAX_X} "
          f"из {detected_inside_count} внутри маски ({detected_count} всего)")
    print(f"Ground Truth: {gt_areas.size} с площадью от {MIN_X} до {MAX_X} "
          f"из {gt_inside_count} внутри маски ({gt_count} всего)")
    print_fit_summary("найденные объекты", predicted_mu, predicted_sigma)
    print_fit_summary("Ground Truth", gt_mu, gt_sigma)
    print(f"ECDF сохранена: {OUTPUT_IMAGE_PATH}")
    print(f"PDF сохранена: {OUTPUT_PDF_IMAGE_PATH}")


if __name__ == "__main__":
    main()
