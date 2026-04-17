"""Train Stage 1 (real vs forged) or Stage 2 (edited vs AI-generated)."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from dataset import ForensicDataset
from model import UnifiedForgeryModel
from config_utils import load_config, resolve_path


def _class_weights(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    # Computes inverse-frequency class weights so that underrepresented classes
    # contribute more to the loss during training.
    counter = Counter(labels)
    total = sum(counter.values())
    return torch.tensor(
        [total / (num_classes * counter.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )


def _data_args(p: argparse.ArgumentParser):
    # Adds common dataset and training arguments shared by both Stage 1 and Stage 2.
    p.add_argument("--rgb-root", type=Path, required=True)
    p.add_argument("--ela-root", type=Path, required=True)
    p.add_argument("--srm-root", type=Path, required=True)
    p.add_argument("--fft-root", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)


# ---------------------------------------------------------------------------
# Stage 1 — binary: real vs forged
# ---------------------------------------------------------------------------
def train_stage1(args):
    print("Building dataset...")
    # Loads the full dataset containing all three classes (real, edited, AI).
    ds = ForensicDataset(
        args.rgb_root, args.ela_root, args.srm_root, args.fft_root,
        image_size=args.image_size,
    )

    print(f"Total samples: {len(ds)}")

    # Count how many real vs forged samples exist.
    # Forged = edited + AI-generated.
    n_real = sum(1 for s in ds.samples if s.label == 0)
    n_forged = sum(1 for s in ds.samples if s.label != 0)
    print(f"real={n_real}, forged={n_forged}")

    # Ensure both classes are present, otherwise training is invalid.
    if n_real == 0 or n_forged == 0:
        raise SystemExit(
            f"Stage 1 needs both classes. real={n_real}, forged={n_forged}.\n"
        )

    print("Creating dataloader...")
    # Creates a dataloader that shuffles data each epoch.
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print("Dataloader ready")

    print(f"Using device: {args.device}")
    model = UnifiedForgeryModel().to(args.device)

    # If a checkpoint already exists, resume training from it.
    if Path(args.save_path).exists():
        model.load_state_dict(torch.load(args.save_path, map_location=args.device))
        print(f"Loaded checkpoint from {args.save_path}")

    # Convert labels into binary form: 0 = real, 1 = forged.
    labels = [int(s.label != 0) for s in ds.samples]

    # Use class weights to handle imbalance between real and forged samples.
    criterion = nn.CrossEntropyLoss(weight=_class_weights(labels, 2).to(args.device))

    # AdamW optimizer for stable training with weight decay regularization.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    print("Starting training...\n")

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs} started")
            running = 0.0

            for i, batch in enumerate(loader):
                if i == 0:
                    print("    First batch loaded")

                if i % 10 == 0:
                    print(f"    Batch {i+1}/{len(loader)}")

                rgb = batch["rgb"].to(args.device)
                fore = batch["forensic"].to(args.device)
                y = batch["label2"].to(args.device)

                # Forward pass through stage1 head only.
                loss = criterion(model(rgb, fore)["stage1_logits"], y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += loss.item()

            epoch_loss = running / len(loader)
            print(f" Epoch {epoch + 1} done | loss={epoch_loss:.4f}")

            # Save checkpoint after each epoch so training progress is preserved.
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f" Saved checkpoint → {args.save_path}\n")

    except KeyboardInterrupt:
        # If training is interrupted manually, save the current model state.
        print("\n Interrupted! Saving model before exit...")
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f" Saved checkpoint → {args.save_path}")

    print(" Training complete!")

# ---------------------------------------------------------------------------
# Stage 2 — conditional: edited vs AI-generated (forged samples only)
# ---------------------------------------------------------------------------

def train_stage2(args):
    print("Building dataset...")
    # Load the full dataset again (includes all classes).
    ds = ForensicDataset(
        args.rgb_root, args.ela_root, args.srm_root, args.fft_root,
        image_size=args.image_size,
    )
    print(f"Total samples: {len(ds)}")

    # Select only forged samples (edited and AI) for Stage 2 training.
    forged_idx = [i for i, s in enumerate(ds.samples) if s.label in (1, 2)]

    # Count distribution of edited vs AI samples.
    n_edited = sum(1 for s in ds.samples if s.label == 1)
    n_ai = sum(1 for s in ds.samples if s.label == 2)
    print(f"edited={n_edited}, ai_generated={n_ai}, forged_total={len(forged_idx)}")

    # If no forged samples exist, Stage 2 cannot be trained.
    if not forged_idx:
        raise SystemExit("No forged samples found. Cannot train Stage 2.")

    print("Creating dataloader...")
    # Create dataloader using only forged samples.
    loader = DataLoader(
        Subset(ds, forged_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print("Dataloader ready")
    print(f"Using device: {args.device}")

    model = UnifiedForgeryModel().to(args.device)

    # Load Stage 1 weights to reuse learned feature representations.
    if args.skip_stage1_ckpt:
        print("Skipping Stage 1 checkpoint — using ImageNet init.")
    elif args.stage1_ckpt and Path(args.stage1_ckpt).exists():
        model.load_state_dict(torch.load(args.stage1_ckpt, map_location=args.device))
        print(f"Loaded Stage 1 weights from {args.stage1_ckpt}")
    else:
        raise SystemExit(
            f"Stage 1 checkpoint not found at {args.stage1_ckpt}.\n"
            "Run train stage1 first, or pass --skip-stage1-ckpt."
        )

    print(" Freezing shared trunk; training only Stage 2 head")

    # Freeze backbone and fusion layers so only stage2 head is trained.
    for p in (
        list(model.rgb_branch.parameters())
        + list(model.forensic_branch.parameters())
        + list(model.fusion.parameters())
    ):
        p.requires_grad = False

    # Convert labels: edited → 0, ai → 1
    labels = [0 if ds.samples[i].label == 1 else 1 for i in forged_idx]

    # Use class-weighted loss again to handle imbalance.
    criterion = nn.CrossEntropyLoss(weight=_class_weights(labels, 2).to(args.device))

    # Optimizer updates only stage2_head parameters.
    optimizer = AdamW(model.stage2_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    print("Starting Stage 2 training...\n")

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs} started")
            running = 0.0

            for i, batch in enumerate(loader):
                if i == 0:
                    print("    First batch loaded")

                if i % 10 == 0:
                    print(f"    Batch {i+1}/{len(loader)}")

                rgb = batch["rgb"].to(args.device)
                fore = batch["forensic"].to(args.device)

                # Convert original 3-class label into 2-class (edited vs AI).
                y = (batch["label3"].to(args.device) - 1).clamp(0, 1)

                # Forward pass through stage2 head only.
                loss = criterion(model(rgb, fore)["stage2_logits"], y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += loss.item()

            epoch_loss = running / len(loader)
            print(f" Epoch {epoch + 1} done | stage2_loss={epoch_loss:.4f}")

            # Save checkpoint after each epoch.
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f" Saved checkpoint → {args.save_path}\n")

    except KeyboardInterrupt:
        print("\n Interrupted! Saving model before exit...")
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f" Saved checkpoint → {args.save_path}")

    print(" Stage 2 training complete!")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # CLI entry point that allows selecting Stage 1 or Stage 2 training.
    parser = argparse.ArgumentParser(description="Train Stage 1 or Stage 2")
    sub = parser.add_subparsers(dest="stage", required=True)

    # Stage 1 CLI setup
    p1 = sub.add_parser("stage1", help="Binary: real vs forged")
    _data_args(p1)
    p1.add_argument("--epochs", type=int, default=15)
    cfg = load_config()
    ckpts = cfg.get("checkpoints", {})
    p1.add_argument("--save-path", type=Path, default=resolve_path(ckpts.get("stage1", "checkpoints/stage1.pt")))
    p1.set_defaults(func=train_stage1)

    # Stage 2 CLI setup
    p2 = sub.add_parser("stage2", help="Conditional: edited vs AI-generated")
    _data_args(p2)
    p2.add_argument("--epochs", type=int, default=10)
    p2.add_argument("--stage1-ckpt", type=Path, default=resolve_path(ckpts.get("stage1", "checkpoints/stage1.pt")))
    p2.add_argument("--save-path", type=Path, default=resolve_path(ckpts.get("stage2", "checkpoints/stage2.pt")))
    p2.add_argument("--skip-stage1-ckpt", action="store_true",
                    help="Skip loading Stage 1 weights (train from ImageNet init).")
    p2.set_defaults(func=train_stage2)

    args = parser.parse_args()

    # Dispatch to the selected training function.
    args.func(args)


if __name__ == "__main__":
    main()