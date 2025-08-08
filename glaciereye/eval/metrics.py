from __future__ import annotations
import numpy as np
import torch


def threshold_preds(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > thr).float()


def compute_confusion(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.view(-1)
    target = target.view(-1)
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    return tp, fp, fn, tn


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    tp, fp, fn, _ = compute_confusion(pred, target)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    tp, fp, fn, _ = compute_confusion(pred, target)
    return (tp + eps) / (tp + fp + fn + eps)


def precision_recall(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    tp, fp, fn, _ = compute_confusion(pred, target)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision, recall


def area_km2(mask: np.ndarray, pixel_size_m: float) -> float:
    # mask: HxW boolean/0-1; pixel_size_m: ground sampling distance (m per pixel)
    area_m2 = mask.sum() * (pixel_size_m ** 2)
    return area_m2 / 1e6
