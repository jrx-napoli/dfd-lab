import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so `src.*` imports work even when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.shard_dataset import ShardClipDataset


class SimpleCNN(nn.Module):
    """A small CNN for binary classification on single RGB frames."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def collate_center_frame(samples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function: take center frame from each clip, to float32 [0,1], CHW.

    Each sample has keys: "data": Tensor[T,H,W,3] uint8, "label": int.
    """
    images: List[torch.Tensor] = []
    labels: List[int] = []
    for s in samples:
        frames: torch.Tensor = s["data"]  # [T,H,W,3], uint8 on CPU
        t = int(frames.shape[0] // 2)
        img = frames[t]  # [H,W,3]
        img = img.permute(2, 0, 1).contiguous()  # [3,H,W]
        img = img.to(torch.float32).div_(255.0)
        images.append(img)
        labels.append(int(s["label"]))
    batch_x = torch.stack(images, dim=0)
    batch_y = torch.tensor(labels, dtype=torch.long)
    return batch_x, batch_y


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    data_wait_s = 0.0
    compute_s = 0.0

    t_prev = time.perf_counter()
    for batch in loader:
        t_after_data = time.perf_counter()
        data_wait_s += t_after_data - t_prev

        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t_compute_start = time.perf_counter()
        logits = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_compute_end = time.perf_counter()
        compute_s += t_compute_end - t_compute_start

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        t_prev = t_compute_end

    acc = correct / max(total, 1)
    return {
        "accuracy": acc,
        "data_wait_s": data_wait_s,
        "compute_s": compute_s,
        "total_s": data_wait_s + compute_s,
        "samples": total,
        "throughput_samples_per_s": total / max(data_wait_s + compute_s, 1e-9),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int | None = None,
) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    seen = 0

    data_wait_s = 0.0
    compute_s = 0.0

    t_prev = time.perf_counter()
    steps = 0
    for batch in loader:
        t_after_data = time.perf_counter()
        data_wait_s += t_after_data - t_prev

        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t_compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_compute_end = time.perf_counter()
        compute_s += t_compute_end - t_compute_start

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            running_correct += (pred == y).sum().item()
            running_loss += float(loss.item())
            seen += y.numel()

        steps += 1
        t_prev = t_compute_end
        if max_steps is not None and steps >= max_steps:
            break

    avg_loss = running_loss / max(steps, 1)
    acc = running_correct / max(seen, 1)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "data_wait_s": data_wait_s,
        "compute_s": compute_s,
        "total_s": data_wait_s + compute_s,
        "samples": seen,
        "throughput_samples_per_s": seen / max(data_wait_s + compute_s, 1e-9),
        "steps": steps,
    }


def build_loaders(
    shards_dir: str,
    index_filename: str,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> DataLoader:
    dataset = ShardClipDataset(
        shards_dir=shards_dir,
        index_filename=index_filename,
        target_device="cpu",
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_center_frame,
    )
    return loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple CNN on shard dataset and measure IO vs compute")
    parser.add_argument("--shards-dir", type=str, required=True, help="Path to directory with shard .tar files and index CSV(s)")
    parser.add_argument("--train-index", type=str, default="index.csv", help="Train index CSV filename in shards dir")
    parser.add_argument("--val-index", type=str, default=None, help="Validation index CSV filename in shards dir; if missing, validation is skipped")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Maximum number of training samples to stream per epoch")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Maximum number of validation samples to stream")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build data loaders
    train_loader = build_loaders(
        shards_dir=args.shards_dir,
        index_filename=args.train_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_train_samples,
    )

    val_loader = None
    if args.val_index is not None:
        val_path = os.path.join(args.shards_dir, args.val_index)
        if os.path.exists(val_path):
            val_loader = build_loaders(
                shards_dir=args.shards_dir,
                index_filename=args.val_index,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_val_samples,
            )
        else:
            print(f"Validation index '{val_path}' not found; skipping validation.")

    # Model, optimizer
    model = SimpleCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        print(
            "TRAIN | loss={loss:.4f} acc={acc:.4f} data_wait={dw:.3f}s compute={comp:.3f}s total={tot:.3f}s thrpt={thrpt:.2f} samples/s steps={steps}".format(
                loss=train_stats["loss"],
                acc=train_stats["accuracy"],
                dw=train_stats["data_wait_s"],
                comp=train_stats["compute_s"],
                tot=train_stats["total_s"],
                thrpt=train_stats["throughput_samples_per_s"],
                steps=train_stats["steps"],
            )
        )

        if val_loader is not None:
            val_stats = evaluate(model, val_loader, device)
            print(
                "VAL   | acc={acc:.4f} data_wait={dw:.3f}s compute={comp:.3f}s total={tot:.3f}s thrpt={thrpt:.2f} samples/s samples={n}".format(
                    acc=val_stats["accuracy"],
                    dw=val_stats["data_wait_s"],
                    comp=val_stats["compute_s"],
                    tot=val_stats["total_s"],
                    thrpt=val_stats["throughput_samples_per_s"],
                    n=val_stats["samples"],
                )
            )


if __name__ == "__main__":
    main()


