"""train.py — Fine-tune MobileNetV3-Large for 5-class single-label classification.

Deployed Model Target:
    mobilenetv3_large_skin_ai.onnx (Softmax single-label top prediction)

Label vocabulary (5 classes):
    ['acne', 'dryness', 'hair_thinning', 'pigmentation', 'redness']

Usage:
    python training/train.py \
        --dataset-dir data/Skin_AI_Dataset/ \
        --splits-dir data/splits/ \
        --epochs 20 \
        --batch-size 16 \
        --lr 1e-4 \
        --out checkpoints/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABEL_COLS: list[str] = [
    "acne",
    "dryness",
    "hair_thinning",
    "pigmentation",
    "redness",
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


class GenXSingleLabelDataset:
    def __init__(self, json_path: Path, dataset_root: Path, transform) -> None:
        from PIL import Image
        self.samples: list[dict[str, str]] = json.loads(json_path.read_text(encoding="utf-8"))
        self.dataset_root = dataset_root
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(LABEL_COLS)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image
        sample = self.samples[idx]
        img_path = self.dataset_root / sample["image_path"]
        
        # Fallback dummy image if file is missing
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        label_idx = self.label_to_idx.get(sample["label"], 0)
        return self.transform(img), torch.tensor(label_idx, dtype=torch.long)


def train(
    dataset_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MobileNetV3-Large on {device}")

    train_ds = GenXSingleLabelDataset(splits_dir / "train.json", dataset_dir, build_transforms(train=True))
    val_ds = GenXSingleLabelDataset(splits_dir / "val.json", dataset_dir, build_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build MobileNetV3-Large
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(LABEL_COLS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / max(len(val_ds), 1)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_mobilenetv3.pth")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GenX MobileNetV3-Large Classifier")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/Skin_AI_Dataset"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args.dataset_dir, args.splits_dir, args.out, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
