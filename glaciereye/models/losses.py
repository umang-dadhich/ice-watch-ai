from __future__ import annotations
import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha=0.25, gamma=2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = torch.exp(-bce)
    loss = alpha * (1 - p_t) ** gamma * bce
    return loss.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return dice_loss(logits, targets) + focal_loss(logits, targets)
