from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, mapping
from pystac_client import Client

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

S2_COG_COLLECTION = "sentinel-2-l2a"
L8_COLLECTION = "landsat-8-l2"

BASIC_S2_BANDS = ["B02", "B03", "B04", "B08", "SCL"]


def load_aoi(aoi_path: str | Path) -> dict:
    gdf = gpd.read_file(aoi_path)
    gdf = gdf.to_crs(epsg=4326)
    geom = json.loads(gdf.iloc[0].geometry.to_json())
    return geom


def search_s2(aoi_geojson: dict, start_date: str, end_date: str, max_items: int = 10):
    client = Client.open(EARTH_SEARCH)
    search = client.search(
        collections=[S2_COG_COLLECTION],
        intersects=aoi_geojson,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 30}},
        max_items=max_items,
    )
    return list(search.get_items())


def download_asset(asset_href: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    # Streamed download
    import requests

    with requests.get(asset_href, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


def clip_to_aoi(src_path: Path, aoi_geojson: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(src_path) as src:
        g = shape(aoi_geojson)
        out_image, out_transform = rio.mask.mask(src, [g], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })
        with rio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)
    return out_path


def fetch_s2_stack(
    aoi_path: str | Path,
    date_start: str,
    date_end: str,
    out_dir: str | Path = "data/raw/s2",
    bands: List[str] = BASIC_S2_BANDS,
    max_items: int = 3,
) -> List[Path]:
    aoi = load_aoi(aoi_path)
    items = search_s2(aoi, date_start, date_end, max_items=max_items)
    saved: List[Path] = []

    for it in items:
        sid = it.id
        for b in bands:
            if b not in it.assets:
                continue
            href = it.assets[b].href
            local = Path(out_dir) / sid / f"{sid}_{b}.tif"
            clipped = Path(out_dir) / sid / f"{sid}_{b}_clip.tif"
            try:
                p = download_asset(href, local)
                clip_to_aoi(p, aoi, clipped)
                saved.append(clipped)
            except Exception as e:
                print(f"Failed {sid} {b}: {e}")

    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Sentinel-2 COGs via STAC and clip to AOI")
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON/GeoPackage/Shapefile")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out", default="data/raw/s2", help="Output folder")
    parser.add_argument("--max", type=int, default=3, help="Max scenes")
    args = parser.parse_args()

    out = fetch_s2_stack(args.aoi, args.start, args.end, args.out, max_items=args.max)
    print(f"Saved {len(out)} assets to {args.out}")
