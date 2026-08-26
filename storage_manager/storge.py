import cv2
import numpy as np
from pathlib import Path
from typing import Self
from . import image_formatter

class Storage:
    IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm",
                      ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF")
    debug = True

    def __init__(self):
        self.base_name: str | None = None
        self.folder_path: Path | None = None
        self.image_path: Path | None = None
        self.edges_path: Path | None = None
        self.thin_edges_path: Path | None = None

    def load_thin_edges(self, suffix="_thin_edges", ext: str = "png") -> np.ndarray:
        if self.thin_edges_path is None:
            self.thin_edges_path = self.folder_path / (self.base_name + suffix)

        ext = "." + ext.lstrip(".")
        if not self.is_valid_extension(ext):
            raise ValueError(f"Invalid extension: {ext}")
        out_path = self.thin_edges_path.with_suffix(ext)

        edges = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
        if edges is None:
            raise FileNotFoundError(f"cv2.imread() failed for: {out_path}")
        edges = image_formatter.uint8_normalize(edges)
        return edges


    def save_thin_edges(self, edges: np.ndarray, suffix="_thin_edges", ext="png") -> None:
        if self.thin_edges_path is None:
            self.thin_edges_path = self.folder_path / (self.base_name + suffix)

        ext = "." + ext.lstrip(".")
        if not self.is_valid_extension(ext):
            raise ValueError(f"Invalid extension: {ext}")
        out_path = self.thin_edges_path.with_suffix(ext)

        edges = image_formatter.uint8_normalize(edges)
        ok = cv2.imwrite(str(out_path), edges)
        if not ok:
            raise IOError(f"cv2.imwrite failed for: {out_path}")
        if self.debug:
            print("Thin edges saved:", out_path)



    def load_edges(self, ext: str = "png") -> np.ndarray:
        if self.edges_path is None:
            self.edges_path = self.folder_path / (self.base_name + "_edges")

        ext = "." + ext.lstrip(".")
        if not self.is_valid_extension(ext):
            raise ValueError(f"Invalid extension: {ext}")
        out_path = self.edges_path.with_suffix(ext)

        edges = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
        if edges is None:
            raise FileNotFoundError(f"cv2.imread() failed for: {out_path}")
        edges = image_formatter.uint8_normalize(edges)
        return edges


    def save_edges(self, edges: np.ndarray, ext="png") -> None:
        if self.edges_path is None:
            self.edges_path = self.folder_path / (self.base_name + "_edges")

        ext = "." + ext.lstrip(".")
        if not self.is_valid_extension(ext):
            raise ValueError(f"Invalid extension: {ext}")
        out_path = self.edges_path.with_suffix(ext)

        edges = image_formatter.uint8_normalize(edges)
        ok = cv2.imwrite(str(out_path), edges)
        if not ok:
            raise IOError(f"cv2.imwrite failed for: {out_path}")

    def load_image(self, size=None):
        self.image_path = self._resolve_image_path()
        if self.debug:
            print("Found image file", self.image_path)
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"cv2.imread() failed for: {self.image_path}")
        image = image_formatter.uint8_normalize(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image if size is None else cv2.resize(image, size)
        return image

    def is_valid_extension(self, ext):
        return ext in self.IMAGE_SUFFIXES

    def _resolve_image_path(self) -> Path | None:
        if self.image_path is not None:
            return self.image_path
        if self.folder_path is None:
            raise ValueError("folder_path is not set")
        if not self.base_name:
            raise ValueError("base_name is not set")
        self.image_path = self._find_image_file(self.folder_path, self.base_name)
        if self.image_path is None:
            raise FileNotFoundError(
                f"Image file not found in {self.folder_path} with name {self.base_name}")
        return self.image_path

    @classmethod
    def _find_image_file(cls, folder_path: Path, name) -> Path | None:
        # Ищет рекурсивно по всей папке и всем подпапкам: **/name.ext
        for suf in cls.IMAGE_SUFFIXES:
            candidate = next(folder_path.rglob(f"{name}{suf}"), None)
            if candidate is not None and candidate.is_file():
                return candidate
        return None

    @classmethod
    def from_thin_edges_path(cls, thin_edges_path: Path) -> Self:
        thin_edges_path = Path(thin_edges_path)
        if not thin_edges_path.is_file():
            raise FileNotFoundError(f"Edges image file not found: {thin_edges_path}")

        base_name = thin_edges_path.stem
        suffix = "_thin_edges"
        if not base_name.endswith(suffix):
            raise ValueError(f"Edges file name must end with '{suffix}': {thin_edges_path.name}")

        base_name = base_name[: -len(suffix)]
        if not base_name:
            raise ValueError(f"Invalid base name after removing '{suffix}': {thin_edges_path.name}")

        storage = cls()
        storage.thin_edges_path = thin_edges_path
        storage.folder_path = thin_edges_path.parent
        storage.base_name = base_name
        return storage


    @classmethod
    def from_edges_path(cls, edges_path: Path) -> Self:
        edges_path = Path(edges_path)
        if not edges_path.is_file():
            raise FileNotFoundError(f"Edges image file not found: {edges_path}")

        base_name = edges_path.stem
        suffix = "_edges"
        if not base_name.endswith(suffix):
            raise ValueError(f"Edges file name must end with '{suffix}': {edges_path.name}")

        base_name = base_name[: -len(suffix)]
        if not base_name:
            raise ValueError(f"Invalid base name after removing '{suffix}': {edges_path.name}")

        storage = cls()
        storage.edges_path = edges_path
        storage.folder_path = edges_path.parent
        storage.base_name = base_name
        return storage

    @classmethod
    def from_image_path(cls, image_path: Path) -> Self:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        storage = cls()
        storage.image_path = image_path
        storage.base_name = image_path.stem
        storage.folder_path = image_path.parent
        return storage

    @classmethod
    def from_folder_path(cls, folder_path: Path) -> Self:
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Folder not found: {folder_path}")
        storage = cls()
        storage.folder_path = folder_path
        storage.base_name = folder_path.name
        return storage
