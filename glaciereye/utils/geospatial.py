from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def read_geotiff(path: str | Path) -> Tuple[np.ndarray, dict]:
    with rio.open(path) as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta


def write_geotiff(path: str | Path, arr: np.ndarray, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta.copy()
    meta.update({"count": arr.shape[0]})
    with rio.open(path, "w", **meta) as dst:
        dst.write(arr)


def reproject_match(src_path: Path, ref_path: Path, out_path: Path):
    with rio.open(src_path) as src, rio.open(ref_path) as ref:
        dst_crs = ref.crs
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

        with rio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
