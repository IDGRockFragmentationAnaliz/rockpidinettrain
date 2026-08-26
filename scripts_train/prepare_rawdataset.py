from pathlib import Path
from tqdm import tqdm
from rocknetmanager import DatasetPathList


def create_lst(root_path: str | Path, path_dataset = None, save_name: str = "raw_dataset.lst"):
    root_path = Path(root_path).resolve()
    if path_dataset is None:
        path_dataset = root_path / r"handmark"

    if not path_dataset.exists():
        raise FileNotFoundError(f"Папка не найдена: {path_dataset}")

    dataset_lst = DatasetPathList(root_path=root_path)

    sample_dirs = sorted(
        sample_dir
        for sample_dir in path_dataset.iterdir()
        if sample_dir.is_dir()
    )

    for sample_dir in tqdm(sample_dirs, desc="Создание .lst", unit="sample"):
        if not sample_dir.is_dir():
            continue
        sample_name = sample_dir.name

        # image_path = sample_dir / f"{sample_name}.png"
        # label_path = sample_dir / f"traces"
        # mask_path = sample_dir / "areas"

        image_path = sample_dir / f"{sample_name}.png"
        label_path = sample_dir / f"{sample_name}_edges_thin_original.png"
        mask_path = sample_dir / "areas"

        if not image_path.exists():
            raise FileNotFoundError(f"Не найден image: {image_path}")

        if not label_path.exists():
            raise FileNotFoundError(f"Не найден label: {label_path}")

        if not mask_path.exists():
            raise FileNotFoundError(f"Не найден mask: {mask_path}")

        # dataset_lst.add(
        #     path_image=image_path,
        #     path_label=label_path
        # )
        dataset_lst.add(
            path_image=image_path,
            path_label=label_path,
            path_mask=mask_path,
        )

    save_path = root_path / save_name
    dataset_lst.save(save_path)

    print(f"Создан файл: {save_path}")
    print(f"Добавлено записей: {len(dataset_lst)}")


if __name__ == "__main__":
    root = Path(r"D:\Data\Outcrops")
    path_dataset = root / "unmark"
    create_lst(root, path_dataset=path_dataset)