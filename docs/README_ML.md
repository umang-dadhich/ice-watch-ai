# GlacierEye Documentation

This repository contains:
- glaciereye/ — Python modules (data, models, training, eval)
- streamlit_app/ — Streamlit demo app

Quick start:
1. conda create -n glaciereye python=3.10 -y && conda activate glaciereye
2. pip install -r glaciereye/requirements.txt
3. Download data: python glaciereye/data/download_stac.py --aoi path/to/aoi.geojson --start 2019-06-01 --end 2019-09-30 --out data/raw/s2
4. Prepare samples: python glaciereye/data/preprocess.py --raw data/raw/s2 --out data/prep --dem data/dem.tif
5. Train: python glaciereye/train/train_segmentation.py --data data/prep --out outputs --in_ch 4
6. Run demo: streamlit run streamlit_app/app.py
