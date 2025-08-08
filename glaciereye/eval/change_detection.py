from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def retreat_heatmap(mask_t1: np.ndarray, mask_t2: np.ndarray) -> np.ndarray:
    # 1 where glacier at t1, 0 else; 1 where glacier at t2
    lost = (mask_t1 == 1) & (mask_t2 == 0)
    gained = (mask_t1 == 0) & (mask_t2 == 1)
    heat = np.zeros((*mask_t1.shape, 3), dtype=np.float32)
    heat[..., 0] = lost.astype(np.float32)  # red
    heat[..., 1] = gained.astype(np.float32)  # green
    return heat


def plot_time_series_area(years, areas_km2):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(years, areas_km2, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Glacier area (km²)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
