from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def show_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.imshow(mask, alpha=alpha, cmap="Blues")
    ax.axis("off")
    fig.tight_layout()
    return fig
