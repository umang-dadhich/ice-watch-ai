# GlacierEye — Intelligent Glacier Segmentation

This folder contains the core Python code for data, models, training, evaluation, and utilities.

Recommended environment:

```
conda create -n glaciereye python=3.10 -y
conda activate glaciereye
pip install -r requirements.txt
```

Project structure:
- data/ — dataset download and preprocessing
- models/ — segmentation architectures and losses
- train/ — training scripts (supervised & semi-supervised)
- eval/ — metrics, postprocessing, change detection, viz
- utils/ — geospatial helpers
