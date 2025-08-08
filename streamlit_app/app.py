import streamlit as st
import numpy as np
import torch
import rasterio as rio
from pathlib import Path

st.set_page_config(page_title="GlacierEye Demo", layout="wide")

@st.cache_resource
def load_model(ckpt_path: str | Path):
    try:
        from glaciereye.models.unet import UNet
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = UNet(in_channels=ckpt.get("in_channels", 4), num_classes=1)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Model not loaded: {e}. Using a dummy predictor.")
        return None


def predict(model, arr: np.ndarray):
    if model is None:
        # Dummy prediction for demo
        h, w = arr.shape[-2:]
        return (np.random.rand(h, w) > 0.7).astype(np.uint8)
    with torch.no_grad():
        x = torch.from_numpy(arr).float()[None, ...]
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].numpy()
        return (prob > 0.5).astype(np.uint8)


st.title("GlacierEye — Glacier Segmentation Demo")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload multi-band GeoTIFF (B02,B03,B04,B08[,DEM])", type=["tif", "tiff"])
    ckpt = st.text_input("Checkpoint path", value="outputs/best.ckpt")
    run = st.button("Run Segmentation")

with col2:
    view = st.selectbox("Band Composite", ["RGB (B04,B03,B02)", "NIR (B08,B04,B03)"])

if run and uploaded:
    model = load_model(ckpt)
    with rio.MemoryFile(uploaded.read()) as mem:
        with mem.open() as src:
            arr = src.read()
            meta = src.meta.copy()
    # Simple composite
    if arr.shape[0] >= 4:
        if view.startswith("RGB"):
            rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
        else:
            rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)
        rgb = (rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) - np.percentile(rgb, 2) + 1e-6)
        rgb = np.clip(rgb, 0, 1)
    else:
        rgb = np.repeat(arr[0][..., None], 3, axis=-1)

    mask = predict(model, arr)

    st.subheader("Results")
    colA, colB = st.columns(2)
    with colA:
        st.image(rgb, caption="Input Composite", use_column_width=True)
    with colB:
        st.image(mask * 255, caption="Segmentation Mask", use_column_width=True)

    # Download
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=".tif") as tmp:
        meta.update({"count": 1, "dtype": "uint8"})
        with rio.open(tmp.name, "w", **meta) as dst:
            dst.write(mask[None, ...].astype(np.uint8))
        with open(tmp.name, "rb") as f:
            st.download_button("Download Mask GeoTIFF", data=f, file_name="glacier_mask.tif", mime="image/tiff")
