"""Generate primitive-shape point-cloud datasets for train.py.

Each saved file contains a dictionary with:
    points:      Tensor[num_examples, num_points, 3]
    point_cloud: Tensor[num_examples, num_points, 6]  # XYZ + normals
    shape:       list[str]

Example:
    .venv\\Scripts\\python.exe preprocess_primitives.py \\
        --output data --train-size 10000 --val-size 1000 --num-points 1024
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh


SHAPES = ("cube", "sphere", "torus", "cone", "plane")


def random_rotation(rng):
    """Create a uniformly distributed 3D rotation matrix."""
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def make_mesh(shape, rng):
    """Create a unit primitive with randomized shape dimensions."""
    if shape == "cube":
        extents = rng.uniform(0.75, 1.25, size=3)
        return trimesh.creation.box(extents=extents)
    if shape == "sphere":
        return trimesh.creation.icosphere(
            subdivisions=3, radius=float(rng.uniform(0.45, 0.65))
        )
    if shape == "torus":
        return trimesh.creation.torus(
            major_radius=float(rng.uniform(0.35, 0.55)),
            minor_radius=float(rng.uniform(0.12, 0.22)),
        )
    if shape == "cone":
        return trimesh.creation.cone(
            radius=float(rng.uniform(0.4, 0.65)),
            height=float(rng.uniform(0.75, 1.25)),
        )
    if shape == "plane":
        return trimesh.creation.box(
            extents=(float(rng.uniform(0.9, 1.3)), float(rng.uniform(0.9, 1.3)), 0.03)
        )
    raise ValueError(f"unknown shape: {shape}")


def normalize_to_unit_box(mesh):
    """Center a mesh and fit its longest bounding-box side to length one."""
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    center = (bounds[0] + bounds[1]) * 0.5
    extent = np.max(bounds[1] - bounds[0])
    if extent <= 0:
        raise ValueError("cannot normalize a degenerate mesh")
    mesh.vertices = (np.asarray(mesh.vertices) - center) / extent
    return mesh


def sample_surface(mesh, num_points):
    """Sample XYZ and face normals with trimesh area-weighted surface sampling."""
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float32)
    return np.asarray(points, dtype=np.float32), normals


def generate_example(shape, num_points, condition_points, rng, scale_min, scale_max):
    mesh = normalize_to_unit_box(make_mesh(shape, rng))
    points, _ = sample_surface(mesh, num_points)
    condition_xyz, condition_normals = sample_surface(mesh, condition_points)

    scale = rng.uniform(scale_min, scale_max, size=3).astype(np.float32)
    rotation = random_rotation(rng)
    points = (points * scale) @ rotation.T
    condition_xyz = (condition_xyz * scale) @ rotation.T
    condition_normals = (condition_normals / scale) @ rotation.T
    condition_normals /= np.linalg.norm(condition_normals, axis=1, keepdims=True).clip(min=1e-8)

    # Keep every example centered and at a comparable scale.
    center = points.mean(axis=0, keepdims=True)
    points -= center
    condition_xyz -= center
    return points.astype(np.float32), condition_xyz.astype(np.float32), condition_normals.astype(np.float32)


def generate_dataset(size, num_points, condition_points, seed, scale_min, scale_max):
    rng = np.random.default_rng(seed)
    points = np.empty((size, num_points, 3), dtype=np.float32)
    point_cloud = np.empty((size, condition_points, 6), dtype=np.float32)
    shape_names = []
    shape_order = list(SHAPES)

    for index in range(size):
        # Shuffle shape order while keeping the class distribution balanced.
        if index % len(SHAPES) == 0:
            rng.shuffle(shape_order)
        shape = shape_order[index % len(SHAPES)]
        xyz, condition_xyz, condition_normals = generate_example(
            shape, num_points, condition_points, rng, scale_min, scale_max
        )
        points[index] = xyz
        point_cloud[index] = np.concatenate((condition_xyz, condition_normals), axis=-1)
        shape_names.append(shape)

    return {
        "points": torch.from_numpy(points),
        "point_cloud": torch.from_numpy(point_cloud),
        "shape": shape_names,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument(
        "--condition-points",
        type=int,
        default=2048,
        help="Number of points in each conditioning cloud",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.25)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.train_size < 1
        or args.val_size < 1
        or args.num_points < 3
        or args.condition_points < 3
    ):
        raise ValueError("dataset sizes and point counts must be positive")
    if args.scale_min <= 0 or args.scale_max < args.scale_min:
        raise ValueError("scale range must satisfy 0 < scale-min <= scale-max")

    args.output.mkdir(parents=True, exist_ok=True)
    train = generate_dataset(
        args.train_size,
        args.num_points,
        args.condition_points,
        args.seed,
        args.scale_min,
        args.scale_max,
    )
    val = generate_dataset(
        args.val_size,
        args.num_points,
        args.condition_points,
        args.seed + 1,
        args.scale_min,
        args.scale_max,
    )
    torch.save(train, args.output / "train.pt")
    torch.save(val, args.output / "val.pt")
    print(f"saved {args.train_size} examples to {args.output / 'train.pt'}")
    print(f"saved {args.val_size} examples to {args.output / 'val.pt'}")


if __name__ == "__main__":
    main()
