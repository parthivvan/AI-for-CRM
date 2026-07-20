"""train.py — Fine-tune EfficientNet-B0 for multi-label GenX 360 classification.

Usage:
    python training/train.py \
        --splits data/splits/ \
        --images data/images/ \
        --epochs 20 \
        --batch-size 16 \
        --lr 1e-4 \
        --out checkpoints/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL_COLS = [
    "pigmentation",
    "redness",
    "uneven_texture",
    "dryness",
    "hair_density",
    "other_or_unclear",
]


def build_transforms(train: bool):
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class GenXDataset:
    def __init__(self, csv_path: Path, images_root: Path, transform) -> None:
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image
        row = self.df.iloc[idx]
        img = Image.open(self.images_root / row.image_path).convert("RGB")
        labels = torch.tensor([float(row[col]) for col in LABEL_COLS], dtype=torch.float32)
        return self.transform(img), labels


def train(
    splits_dir: Path,
    images_root: Path,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_ds = GenXDataset(splits_dir / "train.csv", images_root, build_transforms(train=True))
    val_ds = GenXDataset(splits_dir / "val.csv", images_root, build_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Build model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABEL_COLS))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_loss += criterion(model(imgs), labels).item() * len(imgs)
        val_loss /= len(val_ds)

        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  -> Saved best checkpoint (val_loss={best_val_loss:.4f})")

    torch.save(model.state_dict(), out_dir / "last.pth")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 for GenX 360.")
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    train(args.splits, args.images, args.out, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
