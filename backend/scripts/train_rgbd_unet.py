"""Training pipeline for the Late-Fusion RGB-D U-Net.

Supports two dataset modes:
  1. SyntheticWhealDataset — generates procedural training data on-the-fly
     for immediate pipeline validation without real imagery.
  2. WhealRGBDDataset — loads real paired RGB/depth/mask triplets from disk,
     expecting the directory layout:
       data_root/
         rgb/      *.png   (H×W×3, uint8)
         depth/    *.npy   (H×W, float32, meters)
         mask/     *.png   (H×W, uint8, 0 or 255)

Usage:
  # Synthetic quick-test (no data required):
  python -m backend.scripts.train_rgbd_unet --synthetic --epochs 10

  # Real data:
  python -m backend.scripts.train_rgbd_unet --data-dir ./data/wheal_rgbd --epochs 100
"""

import argparse
import os
import math
import time
import sys
from pathlib import Path

# Add backend directory to path so that 'core' can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ─── Attempt optional imports gracefully ─────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.models.unet_rgbd import LateFusionUNet


# ═══════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss — robust for imbalanced segmentation."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_iou(logits, targets, threshold=0.5):
    """Intersection-over-Union for binary masks."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_dice(logits, targets, threshold=0.5):
    """Dice coefficient for binary masks."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()
    if total == 0:
        return 1.0
    return (2.0 * intersection / total).item()


# ═══════════════════════════════════════════════════════════════════════
# Synthetic Dataset (Procedural — no files needed)
# ═══════════════════════════════════════════════════════════════════════

class SyntheticWhealDataset(Dataset):
    """Generates synthetic pairs of (RGB, depth, mask) on-the-fly.

    RGB:   Skin-toned background with a lighter circular wheal region.
    Depth: Flat baseline with a gaussian bump at the wheal location.
    Mask:  Binary circle at the wheal location.
    """

    # Representative skin-tone RGB ranges for diversity
    SKIN_TONES = [
        (210, 180, 140),  # light
        (180, 140, 100),  # medium-light
        (140, 100, 70),   # medium
        (100, 70, 45),    # medium-dark
        (70, 45, 25),     # dark
    ]

    def __init__(self, length=2000, img_size=256, seed=42):
        self.length = length
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        H = W = self.img_size

        # Random skin tone background
        base_rgb = np.array(
            self.SKIN_TONES[self.rng.randint(len(self.SKIN_TONES))], dtype=np.float32
        )

        # Build RGB canvas with slight noise
        rgb = np.tile(base_rgb, (H, W, 1)).astype(np.float32)
        rgb += self.rng.normal(0, 5, rgb.shape).astype(np.float32)
        rgb = np.clip(rgb, 0, 255)

        # Random wheal parameters
        cx = self.rng.randint(40, W - 40)
        cy = self.rng.randint(40, H - 40)
        radius = self.rng.randint(8, 35)

        # Create mask
        mask = np.zeros((H, W), dtype=np.float32)
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask[dist <= radius] = 1.0

        # Wheal region is slightly lighter / redder in RGB
        wheal_color_shift = self.rng.uniform(20, 60)
        rgb[mask > 0, 0] += wheal_color_shift  # more red
        rgb[mask > 0, 1] += wheal_color_shift * 0.3
        rgb = np.clip(rgb, 0, 255)

        # Build depth: flat baseline + gaussian bump at wheal
        baseline_depth = 0.25  # meters
        depth = np.full((H, W), baseline_depth, dtype=np.float32)
        bump_height = self.rng.uniform(0.001, 0.008)  # 1–8 mm elevation
        gaussian_bump = bump_height * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (radius * 0.7) ** 2)
        )
        # Wheal is CLOSER to the camera (lower depth) since it protrudes
        depth -= gaussian_bump.astype(np.float32)

        # Normalize
        rgb = rgb / 255.0
        # Depth normalization matches Metal shader: map [0.1, 0.5] → [0, 1]
        depth = np.clip((depth - 0.1) / 0.4, 0, 1)

        # Convert to tensors: (C, H, W)
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # (3, H, W)
        depth_tensor = torch.from_numpy(depth[np.newaxis]).float()     # (1, H, W)
        mask_tensor = torch.from_numpy(mask[np.newaxis]).float()       # (1, H, W)

        return rgb_tensor, depth_tensor, mask_tensor

class RealFinetuneDataset(Dataset):
    """Loads a real test photo and manual mask to overfit / finetune explicitly."""
    def __init__(self, img_path, length=20, img_size=256, seed=42):
        self.length = length
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        
        mask_path = img_path.replace(".jpg", "_mask.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Manual mask not found at {mask_path}")
            
        import cv2
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess to get the crop/resize the UNet will see
        from backend.services.preprocessing import preprocess
        prep = preprocess(img)
        rgb_img = cv2.cvtColor(prep["resized"], cv2.COLOR_BGR2RGB)
        self.base_rgb = cv2.resize(rgb_img, (img_size, img_size))
        
        H, W = prep["sam_ready_image"].shape[:2]
        self.base_mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        self.base_mask = cv2.resize(self.base_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        self.base_mask = (self.base_mask > 127).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rgb = self.base_rgb.copy().astype(np.float32)
        mask = self.base_mask.copy().astype(np.float32)
        
        if self.rng.rand() > 0.5:
            rgb = np.fliplr(rgb)
            mask = np.fliplr(mask)
        if self.rng.rand() > 0.5:
            rgb = np.flipud(rgb)
            mask = np.flipud(mask)
            
        noise = self.rng.normal(0, 5, rgb.shape).astype(np.float32)
        rgb = np.clip(rgb + noise, 0, 255)
        
        H, W = self.img_size, self.img_size
        depth = np.full((H, W), 0.25, dtype=np.float32)
        depth[mask > 0] -= 0.005 # Bump is closer (lower depth)
        
        rgb = rgb / 255.0
        depth = np.clip((depth - 0.1) / 0.4, 0, 1)
        
        rgb_tensor = torch.from_numpy(rgb.copy().transpose(2, 0, 1)).float()
        depth_tensor = torch.from_numpy(depth.copy()[np.newaxis]).float()
        mask_tensor = torch.from_numpy(mask.copy()[np.newaxis]).float()

        return rgb_tensor, depth_tensor, mask_tensor

# ═══════════════════════════════════════════════════════════════════════
# Real File-Based Dataset
# ═══════════════════════════════════════════════════════════════════════

class WhealRGBDDataset(Dataset):
    """Loads paired RGB / depth / mask triplets from disk.

    Expected directory structure:
      data_root/
        rgb/      *.png   (H×W×3, uint8)
        depth/    *.npy   (H×W, float32, meters)
        mask/     *.png   (H×W, uint8, 0 or 255)

    File stems must match across the three directories.
    """

    def __init__(self, data_root, img_size=256, augment=True):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment and HAS_ALBUMENTATIONS

        rgb_dir = self.data_root / "rgb"
        self.stems = sorted([p.stem for p in rgb_dir.glob("*.png")])

        if len(self.stems) == 0:
            raise FileNotFoundError(f"No PNG files found in {rgb_dir}")

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
                A.ElasticTransform(alpha=30, sigma=5, p=0.3),
                A.GaussNoise(var_limit=(5, 25), p=0.3),
            ], additional_targets={"depth": "image", "mask": "mask"})

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        H = W = self.img_size

        # Load RGB
        rgb_path = self.data_root / "rgb" / f"{stem}.png"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W, H))

        # Load depth
        depth_path = self.data_root / "depth" / f"{stem}.npy"
        depth = np.load(str(depth_path))
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # Load mask
        mask_path = self.data_root / "mask" / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        # Augmentation
        if self.augment:
            depth_3ch = np.stack([depth] * 3, axis=-1)
            augmented = self.transform(image=rgb, depth=depth_3ch, mask=mask)
            rgb = augmented["image"]
            depth = augmented["depth"][:, :, 0]
            mask = augmented["mask"]

        # Normalize
        rgb = rgb.astype(np.float32) / 255.0
        depth = np.clip((depth - 0.1) / 0.4, 0, 1).astype(np.float32)
        mask = mask.astype(np.float32)

        # To tensors
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        depth_tensor = torch.from_numpy(depth[np.newaxis]).float()
        mask_tensor = torch.from_numpy(mask[np.newaxis]).float()

        return rgb_tensor, depth_tensor, mask_tensor


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Dataset & DataLoader ──
    if args.finetune_real:
        print(f"[Train] Using REAL dataset for finetuning from {args.finetune_real}")
        dataset = RealFinetuneDataset(img_path=args.finetune_real, length=args.synthetic_size, img_size=args.img_size)
    elif args.synthetic:
        print("[Train] Using SYNTHETIC dataset for pipeline validation")
        dataset = SyntheticWhealDataset(
            length=args.synthetic_size, img_size=args.img_size
        )
    else:
        print(f"[Train] Loading real data from: {args.data_dir}")
        dataset = WhealRGBDDataset(
            args.data_dir, img_size=args.img_size, augment=True
        )

    # Train/val split (85/15)
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"[Train] Samples: {train_size} train, {val_size} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── Model ──
    model = LateFusionUNet(n_classes=1, bilinear=False).to(device)
    
    ckpt_dir = Path(args.checkpoint_dir)
    final_path = ckpt_dir / "final_rgbd_unet.pth"
    if final_path.exists():
        print(f"[Train] Resuming training from {final_path}")
        ckpt = torch.load(str(final_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {param_count:,}")

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    # ── TensorBoard ──
    writer = None
    if HAS_TENSORBOARD and args.log_dir:
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"[Train] TensorBoard logging to: {args.log_dir}")

    # ── Checkpoint dir ──
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        for rgb, depth, mask in train_loader:
            rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)

            logits = model(rgb, depth)
            loss = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for rgb, depth, mask in val_loader:
                rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
                logits = model(rgb, depth)
                val_loss += criterion(logits, mask).item()
                val_iou += compute_iou(logits, mask)
                val_dice += compute_dice(logits, mask)

        n_val = len(val_loader)
        val_loss /= n_val
        val_iou /= n_val
        val_dice /= n_val

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"IoU={val_iou:.4f}  Dice={val_dice:.4f}  "
            f"lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/IoU", val_iou, epoch)
            writer.add_scalar("Metrics/Dice", val_dice, epoch)
            writer.add_scalar("LR", lr_now, epoch)

        # ── Checkpoint ──
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_path = ckpt_dir / "best_rgbd_unet.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
            }, str(ckpt_path))
            print(f"  OK Saved best checkpoint (IoU={val_iou:.4f})")

        # Save periodic checkpoint every 25 epochs
        if epoch % 25 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, str(ckpt_dir / f"epoch_{epoch:03d}.pth"))

    # ── Final save ──
    final_path = ckpt_dir / "final_rgbd_unet.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "best_val_iou": best_val_iou,
    }, str(final_path))
    print(f"\n[Train] Complete. Best IoU: {best_val_iou:.4f}")
    print(f"[Train] Final model saved to: {final_path}")

    if writer:
        writer.close()

    return str(final_path)


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train the RGB-D Late-Fusion U-Net")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root directory with rgb/, depth/, mask/ subdirs")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic procedural data for testing")
    parser.add_argument("--finetune-real", type=str, default=None,
                        help="Path to real test photo to explicitly finetune on")
    parser.add_argument("--synthetic-size", type=int, default=2000,
                        help="Number of samples to generate")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Training image dimensions (square)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Output
    parser.add_argument("--checkpoint-dir", type=str,
                        default="backend/models/checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log-dir", type=str,
                        default="backend/models/runs",
                        help="TensorBoard log directory")

    args = parser.parse_args()

    if not args.synthetic and args.data_dir is None and args.finetune_real is None:
        parser.error("Must specify --data-dir, --synthetic, or --finetune-real")

    train(args)


if __name__ == "__main__":
    main()
