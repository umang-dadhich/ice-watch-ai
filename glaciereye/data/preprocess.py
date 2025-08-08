from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from skimage.morphology import remove_small_objects

from .augmentations import build_train_aug, build_val_aug


def read_tiff(path: Path) -> Tuple[np.ndarray, Dict]:
    with rio.open(path) as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta


def write_tiff(path: Path, arr: np.ndarray, meta: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta.copy()
    meta.update({"count": arr.shape[0], "compress": "deflate"})
    with rio.open(path, "w", **meta) as dst:
        dst.write(arr)


def scl_cloud_mask(scl: np.ndarray) -> np.ndarray:
    # Sentinel-2 SCL classes: 8,9,10 are cloud/shadow/high-prob cloud
    cloud = np.isin(scl, [3, 8, 9, 10, 11])
    return cloud


def normalize_bands(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    p2 = np.percentile(arr, 2, axis=(1, 2), keepdims=True)
    p98 = np.percentile(arr, 98, axis=(1, 2), keepdims=True)
    arr = (arr - p2) / (p98 - p2 + 1e-6)
    return np.clip(arr, 0.0, 1.0)


def stack_multispectral(band_paths: List[Path]) -> Tuple[np.ndarray, Dict]:
    bands = []
    meta_ref = None
    for p in band_paths:
        arr, meta = read_tiff(p)
        if meta_ref is None:
            meta_ref = meta
        bands.append(arr[0:1])
    stacked = np.concatenate(bands, axis=0)
    return stacked, meta_ref  # C,H,W


def prepare_sample(scene_dir: Path, out_dir: Path, add_dem: Path | None = None) -> Path:
    # Expect files like *_B02_clip.tif, *_B03_clip.tif, *_B04_clip.tif, *_B08_clip.tif and *_SCL_clip.tif
    band_paths = sorted(scene_dir.glob("*_B0*_clip.tif"))
    scl_path = next(scene_dir.glob("*_SCL_clip.tif"), None)
    if not band_paths:
        raise FileNotFoundError(f"No bands found in {scene_dir}")

    bands, meta = stack_multispectral(band_paths)
    if scl_path is not None:
        scl, _ = read_tiff(scl_path)
        cloud = scl_cloud_mask(scl.squeeze())
    else:
        cloud = np.zeros_like(bands[0], dtype=bool)

    if add_dem is not None and Path(add_dem).exists():
        dem, _ = read_tiff(Path(add_dem))
        dem = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem) + 1e-6)
        bands = np.concatenate([bands, dem.astype(np.float32)], axis=0)

    bands = normalize_bands(bands)

    # Save prepared stack
    out_stack = out_dir / scene_dir.name / "stack.tif"
    write_tiff(out_stack, bands, meta)

    # Optional: placeholder label (to be replaced with real glacier polygon rasterization)
    label = np.zeros_like(bands[0], dtype=np.uint8)
    out_label = out_dir / scene_dir.name / "mask.tif"
    write_tiff(out_label, label[None, ...], meta)

    return out_stack


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Sentinel-2 scenes into model-ready stacks")
    parser.add_argument("--raw", required=True, help="Folder with raw clipped scenes (from download_stac)")
    parser.add_argument("--out", required=True, help="Output folder for prepared samples")
    parser.add_argument("--dem", default=None, help="Optional DEM GeoTIFF aligned to AOI")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    scenes = [p for p in raw.iterdir() if p.is_dir()]
    for s in scenes:
        try:
            p = prepare_sample(s, out, add_dem=args.dem)
            print(f"Prepared {p}")
        except Exception as e:
            print(f"Skip {s}: {e}")
