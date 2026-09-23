"""Train TwoStreamDenoiser with a DDPM epsilon-prediction objective.

The dataset file may be a .pt/.pth file containing either:
    {"points": Tensor[N, 3] or Tensor[M, N, 3],
     "point_cloud": Tensor[N, 6] or Tensor[M, N, 6]}

or a .npz file with arrays named ``points`` and ``point_cloud``.
The diffusion target contains XYZ only; the conditioning point cloud contains
XYZ plus normals.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

from model import TwoStreamDenoiser


class GaussianDiffusion(nn.Module):
    """Fixed linear-beta DDPM forward process."""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def q_sample(self, clean_points, timesteps, noise=None):
        """Sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(clean_points)
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
        return sqrt_alpha_bar * clean_points + sqrt_one_minus_alpha_bar * noise


def _as_point_tensor(value, name, channels):
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[-1] != channels:
        raise ValueError(
            f"{name} must have shape [N, {channels}] or [B, N, {channels}], "
            f"got {tuple(value.shape)}"
        )
    return value


class PointCloudDataset(Dataset):
    """Load paired target and conditioning point clouds from one file."""

    def __init__(self, path):
        path = Path(path)
        if path.suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu")
        elif path.suffix == ".npz":
            payload = dict(np.load(path))
        else:
            raise ValueError("data must be a .pt, .pth, or .npz file")

        if isinstance(payload, dict):
            points = payload.get("points", payload.get("target"))
            condition = payload.get("point_cloud", payload.get("condition"))
        else:
            raise ValueError(
                "dataset must be a dictionary containing 'points' [B, N, 3] "
                "and 'point_cloud' [B, N, 6]"
            )

        if points is None:
            raise ValueError("dataset must contain a 'points' or 'target' tensor")
        if condition is None:
            raise ValueError(
                "dataset must contain 'point_cloud' [B, N, 6] for conditioning"
            )
        self.points = _as_point_tensor(points, "points", channels=3)
        self.condition = _as_point_tensor(condition, "point_cloud", channels=6)
        if len(self.points) != len(self.condition):
            raise ValueError("points and point_cloud must contain the same number of examples")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return {
            "points": self.points[index],
            "point_cloud": self.condition[index],
        }


def ddpm_loss(model, diffusion, batch, device):
    """Compute the epsilon-prediction loss for one batch."""
    clean_points = batch["points"].to(device)
    point_cloud = batch["point_cloud"].to(device)
    timesteps = torch.randint(
        0, diffusion.timesteps, (clean_points.shape[0],), device=device
    )
    noise = torch.randn_like(clean_points)
    noisy_points = diffusion.q_sample(clean_points, timesteps, noise)
    predicted_noise, _ = model(
        noisy_points.transpose(1, 2).contiguous(),
        timesteps,
        point_cloud=point_cloud,
    )
    return nn.functional.mse_loss(predicted_noise, noise.transpose(1, 2).contiguous())


def train_one_epoch(model, diffusion, loader, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0.0
    total_examples = 0
    progress = tqdm(loader, desc="training", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        loss = ddpm_loss(model, diffusion, batch, device)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = batch["points"].shape[0]
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{total_loss / max(total_examples, 1):.6f}")
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(model, diffusion, loader, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    progress = tqdm(loader, desc="validation", leave=False)
    for batch in progress:
        loss = ddpm_loss(model, diffusion, batch, device)
        batch_size = batch["points"].shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{total_loss / max(total_examples, 1):.6f}")
    return total_loss / max(total_examples, 1)


def save_checkpoint(path, model, optimizer, scheduler, epoch, loss, val_loss, diffusion):
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "diffusion": {"timesteps": diffusion.timesteps},
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a .pt/.pth/.npz dataset")
    parser.add_argument("--val-data", help="Optional separate validation dataset")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Run validation every N epochs and always on the final epoch",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb-project", default="PointInfinity")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("--val-split must be in [0, 1)")
    if args.val_every < 1:
        raise ValueError("--val-every must be at least 1")
    wandb = None
    if args.wandb:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking requested. Install it with 'pip install wandb'."
            ) from error
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = PointCloudDataset(args.data)
    if dataset.points.shape[1] != args.num_points:
        raise ValueError(
            f"dataset has {dataset.points.shape[1]} points, expected {args.num_points}; "
            "pass --num-points to match the dataset"
        )

    if args.val_data:
        train_dataset = dataset
        val_dataset = PointCloudDataset(args.val_data)
        if val_dataset.points.shape[1] != args.num_points:
            raise ValueError("validation dataset has a different number of points")
    else:
        val_size = max(1, int(len(dataset) * args.val_split))
        if val_size >= len(dataset):
            raise ValueError("dataset must contain more than one example for validation")
        train_dataset, val_dataset = random_split(
            dataset,
            [len(dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = TwoStreamDenoiser(num_points=args.num_points).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )
    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, diffusion, train_loader, optimizer, device, grad_clip=args.grad_clip
        )
        should_validate = epoch % args.val_every == 0 or epoch == args.epochs
        val_loss = evaluate(model, diffusion, val_loader, device) if should_validate else None
        checkpoint = os.path.join(args.output, f"epoch_{epoch:04d}.pt")
        save_checkpoint(
            checkpoint, model, optimizer, scheduler, epoch, loss, val_loss, diffusion
        )
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = os.path.join(args.output, "best.pt")
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                loss,
                val_loss,
                diffusion,
            )
        learning_rate = optimizer.param_groups[0]["lr"]
        val_display = f"{val_loss:.6f}" if val_loss is not None else "skipped"
        print(
            f"epoch {epoch:04d} | train {loss:.6f} | val {val_display} | "
            f"lr {learning_rate:.3e} | "
            f"saved {checkpoint}"
        )
        if wandb is not None:
            metrics = {
                "epoch": epoch,
                "train/loss": loss,
                "learning_rate": learning_rate,
            }
            if val_loss is not None:
                metrics["validation/loss"] = val_loss
            wandb.log(metrics, step=epoch)
        scheduler.step()

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
