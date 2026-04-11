#!/usr/bin/env python3
"""Educational Conditional VAE (CVAE) on Fashion-MNIST.

This script is intentionally simple so it can be adapted into an assignment.
Key CVAE idea: both encoder and decoder are conditioned on class label y.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


NUM_CLASSES = 10
IMAGE_SHAPE = (1, 28, 28)
IMAGE_DIM = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]


class CVAE(nn.Module):
    """Convolutional CVAE for 28x28 grayscale images."""

    def __init__(
        self,
        hidden_dim: int = 512,
        latent_dim: int = 16,
        num_classes: int = 10,
        base_channels: int = 32,
        cond_channels: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.cond_channels = cond_channels

        # Encoder condition: broadcast a label embedding over spatial dimensions.
        self.label_to_map = nn.Embedding(num_classes, cond_channels)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1 + cond_channels, base_channels, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        enc_flat_dim = (base_channels * 4) * 7 * 7
        self.encoder_fc = nn.Sequential(
            nn.Linear(enc_flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Decoder condition: concatenate z and one-hot(y), then decode with transposed convs.
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, enc_flat_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1),
        )
        self._dec_channels = base_channels * 4

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_map = self.label_to_map(y).unsqueeze(-1).unsqueeze(-1)
        y_map = y_map.expand(-1, self.cond_channels, IMAGE_SHAPE[1], IMAGE_SHAPE[2])
        h = self.encoder_conv(torch.cat([x, y_map], dim=1))
        h = self.encoder_fc(h.flatten(start_dim=1))
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        h = self.decoder_fc(torch.cat([z, y_onehot], dim=1))
        h = h.view(z.size(0), self._dec_channels, 7, 7)
        logits = self.decoder_conv(h)
        return logits.view(z.size(0), -1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, y)
        return logits, mu, logvar


def cvae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_flat = x.view(x.size(0), -1)
    recon = F.binary_cross_entropy_with_logits(logits, x_flat, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kld
    return total, recon, kld


def train_one_epoch(
    model: CVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    log_every: int,
    limit_batches: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        if limit_batches > 0 and batch_idx >= limit_batches:
            break

        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits, mu, logvar = model(x, y)
        loss, recon, kld = cvae_loss(logits, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_samples += batch_size
        total_loss += loss.item()
        total_recon += recon.item()
        total_kld += kld.item()

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            avg = total_loss / total_samples
            print(f"    step={batch_idx + 1:4d}  loss/sample={avg:.4f}")

    return {
        "loss_per_sample": total_loss / total_samples,
        "recon_per_sample": total_recon / total_samples,
        "kld_per_sample": total_kld / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: CVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    limit_batches: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        if limit_batches > 0 and batch_idx >= limit_batches:
            break

        x = x.to(device)
        y = y.to(device)
        logits, mu, logvar = model(x, y)
        loss, recon, kld = cvae_loss(logits, x, mu, logvar, beta=beta)

        batch_size = x.size(0)
        total_samples += batch_size
        total_loss += loss.item()
        total_recon += recon.item()
        total_kld += kld.item()

    return {
        "loss_per_sample": total_loss / total_samples,
        "recon_per_sample": total_recon / total_samples,
        "kld_per_sample": total_kld / total_samples,
    }


@torch.no_grad()
def save_class_conditioned_samples(
    model: CVAE,
    device: torch.device,
    out_path: Path,
    samples_per_class: int,
) -> None:
    model.eval()
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(samples_per_class)
    z = torch.randn(NUM_CLASSES * samples_per_class, model.latent_dim, device=device)
    logits = model.decode(z, labels)
    images = torch.sigmoid(logits).view(-1, *IMAGE_SHAPE).cpu()

    grid = make_grid(images, nrow=samples_per_class, padding=2, pad_value=1.0)
    save_image(grid, out_path)


@torch.no_grad()
def save_reconstructions(
    model: CVAE,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    num_items: int,
) -> None:
    model.eval()
    x, y = next(iter(loader))
    x = x[:num_items].to(device)
    y = y[:num_items].to(device)

    logits, _, _ = model(x, y)
    recon = torch.sigmoid(logits).view(-1, *IMAGE_SHAPE)

    comparison = torch.cat([x.cpu(), recon.cpu()], dim=0)
    grid = make_grid(comparison, nrow=num_items, padding=2, pad_value=1.0)
    save_image(grid, out_path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational CVAE on Fashion-MNIST")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cvae_fashion_mnist"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512, help="FC bottleneck width inside conv CVAE.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32, help="Base conv width for encoder/decoder.")
    parser.add_argument("--cond-channels", type=int, default=8, help="Label embedding channels for encoder conditioning.")
    parser.add_argument("--beta", type=float, default=1.0, help="KL multiplier (beta-VAE style).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--sample-cols", type=int, default=8, help="Samples per class in output grid.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=0,
        help="If > 0, only use this many train batches per epoch (for quick smoke tests).",
    )
    parser.add_argument(
        "--limit-test-batches",
        type=int,
        default=0,
        help="If > 0, only use this many test batches during evaluation.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=args.data_dir, train=True, transform=transform, download=True)
    test_ds = datasets.FashionMNIST(root=args.data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = CVAE(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_classes=NUM_CLASSES,
        base_channels=args.base_channels,
        cond_channels=args.cond_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(
        f"Train set: {len(train_ds)} images | Test set: {len(test_ds)} images | "
        f"Params: {sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=args.beta,
            log_every=args.log_every,
            limit_batches=args.limit_train_batches,
        )
        test_stats = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            beta=args.beta,
            limit_batches=args.limit_test_batches,
        )

        print(
            "  train: "
            f"loss={train_stats['loss_per_sample']:.4f}, "
            f"recon={train_stats['recon_per_sample']:.4f}, "
            f"kld={train_stats['kld_per_sample']:.4f}"
        )
        print(
            "  test : "
            f"loss={test_stats['loss_per_sample']:.4f}, "
            f"recon={test_stats['recon_per_sample']:.4f}, "
            f"kld={test_stats['kld_per_sample']:.4f}"
        )

        sample_path = args.out_dir / "images" / f"samples_epoch_{epoch:02d}.png"
        save_class_conditioned_samples(
            model=model,
            device=device,
            out_path=sample_path,
            samples_per_class=args.sample_cols,
        )

    recon_path = args.out_dir / "images" / "reconstructions.png"
    save_reconstructions(
        model=model,
        loader=test_loader,
        device=device,
        out_path=recon_path,
        num_items=10,
    )

    ckpt_path = args.out_dir / "cvae_fashion_mnist.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Saved images in: {args.out_dir / 'images'}")


if __name__ == "__main__":
    main()
