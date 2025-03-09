from pathlib import Path

import rasterio


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def read_tiff_file(file_path):
    tiff = rasterio.open(file_path)
    img = tiff.read(tiff.indexes)
    labels = tiff.descriptions
    return tiff, img, labels
