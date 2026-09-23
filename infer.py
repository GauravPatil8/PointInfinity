"""Generate and visualize a point cloud from a trained PointInfinity checkpoint."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import TwoStreamDenoiser


def load_condition(path, index):
    path = Path(path)
    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".npz":
        payload = dict(np.load(path))
    else:
        raise ValueError("data must be a .pt, .pth, or .npz file")

    if not isinstance(payload, dict):
        raise ValueError("data must contain a 'point_cloud' or 'condition' entry")
    condition = payload.get("point_cloud", payload.get("condition"))
    if condition is None:
        raise ValueError("data must contain a 'point_cloud' or 'condition' entry")

    condition = torch.as_tensor(condition, dtype=torch.float32)
    if condition.ndim == 2:
        condition = condition.unsqueeze(0)
    if condition.ndim != 3 or condition.shape[-1] != 6:
        raise ValueError(
            "conditioning data must have shape [N, 6] or [B, N, 6], "
            f"got {tuple(condition.shape)}"
        )
    if not 0 <= index < len(condition):
        raise IndexError(f"--index must be between 0 and {len(condition) - 1}")
    return condition[index : index + 1]


def load_model(checkpoint_path, num_points, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TwoStreamDenoiser(num_points=num_points).to(device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    timesteps = checkpoint.get("diffusion", {}).get("timesteps", 1000)
    return model, int(timesteps)


@torch.no_grad()
def sample(model, condition, timesteps, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    betas = torch.linspace(1e-4, 2e-2, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    points = torch.randn(
        condition.shape[0], model.num_points, 3, device=device, generator=generator
    )

    # Cache the conditioning embeddings once instead of re-encoding every step.
    cached = model.cached_model_kwargs({"point_cloud": condition})
    embeddings = cached["embeddings"]

    prev_latent = None
    for timestep in range(timesteps - 1, -1, -1):
        timestep_tensor = torch.full(
            (condition.shape[0],), timestep, device=device, dtype=torch.long
        )
        predicted_noise, prev_latent = model(
            points.transpose(1, 2).contiguous(),
            timestep_tensor,
            embeddings=embeddings,
            prev_latent=prev_latent,
        )
        alpha = alphas[timestep]
        alpha_bar = alpha_bars[timestep]
        points = (points - (1.0 - alpha) * predicted_noise.transpose(1, 2) / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
        if timestep > 0:
            alpha_bar_prev = alpha_bars[timestep - 1]
            posterior_variance = betas[timestep] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            points = points + posterior_variance.sqrt() * torch.randn(
                points.shape, device=device, generator=generator
            )
    return points


def save_visualization(condition, generated, output_path):
    condition_xyz = condition[0, :, :3].cpu().numpy()
    generated_xyz = generated[0].cpu().numpy()
    figure = plt.figure(figsize=(12, 5))
    for position, points, title, color in (
        (1, condition_xyz, "Conditioning point cloud", "tab:blue"),
        (2, generated_xyz, "Generated point cloud", "tab:orange"),
    ):
        axis = figure.add_subplot(1, 2, position, projection="3d")
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=color)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.set_box_aspect((1, 1, 1))
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Dataset containing point_cloud [B, N, 6]")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--output", default="generated.pt")
    parser.add_argument("--visualization", default="generated.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    condition = load_condition(args.data, args.index).to(device)
    model, timesteps = load_model(args.checkpoint, args.num_points, device)
    generated = sample(model, condition, timesteps, device, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generated[0].cpu(), output_path)
    save_visualization(condition, generated, args.visualization)
    print(f"saved generated points to {output_path}")
    print(f"saved visualization to {args.visualization}")


if __name__ == "__main__":
    main()
