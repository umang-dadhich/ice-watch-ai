from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
from tqdm import tqdm

from glaciereye.models.unet import UNet
from glaciereye.models.losses import combined_loss

import rasterio as rio


class GeoSegDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", img_size: int = 512):
        root = Path(root)
        self.samples = sorted((root / split).glob("*/stack.tif"))
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stack_path = self.samples[idx]
        mask_path = stack_path.parent / "mask.tif"
        with rio.open(stack_path) as src:
            img = src.read().astype(np.float32)
        with rio.open(mask_path) as srcm:
            m = srcm.read(1).astype(np.float32)
        img_t = torch.from_numpy(img)
        mask_t = torch.from_numpy(m)[None, ...]
        img_t = resize(img_t, [self.img_size, self.img_size])
        mask_t = resize(mask_t, [self.img_size, self.img_size])
        return img_t, mask_t


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = GeoSegDataset(args.data, "train", args.img_size)
    val_ds = GeoSegDataset(args.data, "val", args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = UNet(in_channels=args.in_ch, num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 1e9
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        loss_sum = 0.0
        for img, m in pbar:
            img, m = img.to(device), m.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = combined_loss(logits, m)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * img.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, m in val_loader:
                img, m = img.to(device), m.to(device)
                logits = model(img)
                val_loss += combined_loss(logits, m).item() * img.size(0)
        val_loss /= max(1, len(val_ds))

        print(f"Epoch {epoch}: train={loss_sum/len(train_ds):.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = out_dir / "best.ckpt"
            torch.save({"state_dict": model.state_dict(), "in_channels": args.in_ch}, ckpt)
            print(f"Saved {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train GlacierEye segmentation model")
    ap.add_argument("--data", required=True, help="Path to prepared dataset root containing train/ and val/")
    ap.add_argument("--out", default="outputs", help="Output folder for checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--in_ch", type=int, default=4, help="Number of input channels (e.g., 4 or 5 incl. DEM)")
    args = ap.parse_args()
    train(args)
