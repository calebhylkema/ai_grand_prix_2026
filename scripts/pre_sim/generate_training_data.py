#!/usr/bin/env python3
"""Generate synthetic training data for the YOLOv8 keypoint gate detector.

Renders gates from randomised viewpoints with domain randomisation (random
backgrounds, colour jitter, blur, noise) and exports in YOLO keypoint format.

Output structure::

    output_dir/
      images/train/  *.png
      images/val/    *.png
      labels/train/  *.txt
      labels/val/    *.txt
      dataset.yaml          # YOLO training config

YOLO keypoint label format (one line per gate)::

    class_id cx cy w h  kp0_x kp0_y kp0_v  kp1_x kp1_y kp1_v  ...

All coordinates normalised to [0, 1].  kp_v: 0=missing, 2=visible.

Usage::

    uv run python scripts/generate_training_data.py --num-train 2000 --num-val 200
    uv run python scripts/generate_training_data.py --output data/gates --num-train 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drone_racing.utils.config import load_config
from drone_racing.utils.transforms import body_to_camera_rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate rendering (standalone, no MockSimAdapter dependency)
# ---------------------------------------------------------------------------


def _project_points(
    points_w: np.ndarray,
    r_cw: np.ndarray,
    t_cw: np.ndarray,
    k: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    """Project 3D world points to 2D pixel coordinates.

    Returns None if all points are behind the camera or outside the image.
    """
    pts_c = (r_cw @ points_w.T).T + t_cw[np.newaxis, :]
    if np.all(pts_c[:, 2] <= 0.1):
        return None

    valid = pts_c[:, 2] > 0.1
    if not np.any(valid):
        return None

    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]

    pixels = np.zeros((len(pts_c), 2), dtype=np.float64)
    z_safe = np.maximum(pts_c[:, 2], 0.1)
    pixels[:, 0] = fx * pts_c[:, 0] / z_safe + cx
    pixels[:, 1] = fy * pts_c[:, 1] / z_safe + cy

    margin = 20
    in_frame = (
        (pixels[:, 0] > -margin)
        & (pixels[:, 0] < img_w + margin)
        & (pixels[:, 1] > -margin)
        & (pixels[:, 1] < img_h + margin)
        & valid
    )
    if not np.any(in_frame):
        return None

    return pixels


def render_gate_image(
    rng: np.random.Generator,
    gate_pos_w: np.ndarray,
    gate_ori_quat: np.ndarray,
    outer_corners_local: np.ndarray,
    inner_corners_local: np.ndarray,
    drone_pos_w: np.ndarray,
    drone_ori_quat: np.ndarray,
    k: np.ndarray,
    img_w: int,
    img_h: int,
    tilt_deg: float,
    gate_colour_rgb: tuple[int, int, int] = (255, 200, 0),
    bg_colour_rgb: tuple[int, int, int] = (128, 128, 128),
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Render a single gate and return the image + keypoint annotations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
        (image, outer_pixels (4,2), inner_pixels (4,2))
        Pixel arrays are None if the gate is not visible.
    """
    image = np.full((img_h, img_w, 3), bg_colour_rgb, dtype=np.uint8)

    # Camera extrinsics
    r_cb = body_to_camera_rotation(tilt_deg)
    r_wb = Rotation.from_quat(drone_ori_quat).as_matrix()
    r_cw = r_cb @ r_wb.T
    t_cw = -r_cw @ drone_pos_w

    # Gate corners in world frame
    r_wg = Rotation.from_quat(gate_ori_quat).as_matrix()
    outer_w = (r_wg @ outer_corners_local.T).T + gate_pos_w
    inner_w = (r_wg @ inner_corners_local.T).T + gate_pos_w

    outer_px = _project_points(outer_w, r_cw, t_cw, k, img_w, img_h)
    if outer_px is None:
        return image, None, None

    inner_px = _project_points(inner_w, r_cw, t_cw, k, img_w, img_h)

    # Draw outer gate frame
    pts = outer_px.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(image, [pts], gate_colour_rgb)

    # Draw inner cutout
    if inner_px is not None:
        inner_pts = inner_px.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(image, [inner_pts], bg_colour_rgb)

    return image, outer_px, inner_px


# ---------------------------------------------------------------------------
# Domain randomisation
# ---------------------------------------------------------------------------


def apply_domain_randomisation(
    rng: np.random.Generator,
    image: np.ndarray,
) -> np.ndarray:
    """Apply domain randomisation to a rendered image.

    Augmentations (applied randomly):
    - HSV colour jitter
    - Gaussian noise
    - Gaussian blur
    - Brightness/contrast shift
    - Random background patches (simulates clutter)

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    image : np.ndarray
        (H, W, 3) uint8 RGB image.

    Returns
    -------
    np.ndarray
        Augmented image.
    """
    img = image.copy()

    # --- HSV jitter ---
    if rng.random() < 0.7:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.integers(-15, 16)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.integers(-40, 41), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.integers(-40, 41), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- Gaussian noise ---
    if rng.random() < 0.5:
        sigma = rng.uniform(5, 25)
        noise = rng.normal(0, sigma, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # --- Gaussian blur ---
    if rng.random() < 0.3:
        ksize = rng.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # --- Brightness/contrast ---
    if rng.random() < 0.5:
        alpha = rng.uniform(0.7, 1.3)  # contrast
        beta = rng.uniform(-30, 30)  # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # --- Random background rectangles (clutter) ---
    if rng.random() < 0.4:
        h, w = img.shape[:2]
        n_rects = rng.integers(1, 6)
        for _ in range(n_rects):
            x1, y1 = rng.integers(0, w), rng.integers(0, h)
            x2 = min(x1 + rng.integers(20, 150), w)
            y2 = min(y1 + rng.integers(20, 150), h)
            colour = tuple(int(c) for c in rng.integers(50, 200, 3))
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)

    return img


# ---------------------------------------------------------------------------
# YOLO keypoint annotation
# ---------------------------------------------------------------------------


def make_yolo_label(
    outer_px: np.ndarray,
    inner_px: np.ndarray | None,
    img_w: int,
    img_h: int,
) -> str | None:
    """Create a YOLO keypoint annotation line.

    Parameters
    ----------
    outer_px : np.ndarray
        (4, 2) outer corner pixel coordinates.
    inner_px : np.ndarray | None
        (4, 2) inner corner pixel coordinates, or None.
    img_w, img_h : int
        Image dimensions for normalisation.

    Returns
    -------
    str | None
        YOLO format label line, or None if gate is mostly off-screen.
    """
    # Combine all visible keypoints
    all_kp = outer_px.copy()
    if inner_px is not None:
        all_kp = np.vstack([all_kp, inner_px])

    # Bounding box from all keypoints
    x_min = float(np.clip(all_kp[:, 0].min(), 0, img_w))
    x_max = float(np.clip(all_kp[:, 0].max(), 0, img_w))
    y_min = float(np.clip(all_kp[:, 1].min(), 0, img_h))
    y_max = float(np.clip(all_kp[:, 1].max(), 0, img_h))

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    if bbox_w < 10 or bbox_h < 10:
        return None  # too small

    # Normalised bbox centre + size
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    bw = bbox_w / img_w
    bh = bbox_h / img_h

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]

    # 8 keypoints: 4 outer + 4 inner
    for i in range(4):
        kx = float(outer_px[i, 0]) / img_w
        ky = float(outer_px[i, 1]) / img_h
        visible = 0.0 <= outer_px[i, 0] < img_w and 0.0 <= outer_px[i, 1] < img_h
        parts.append(f"{kx:.6f} {ky:.6f} {2 if visible else 0}")

    for i in range(4):
        if inner_px is not None:
            kx = float(inner_px[i, 0]) / img_w
            ky = float(inner_px[i, 1]) / img_h
            visible = 0.0 <= inner_px[i, 0] < img_w and 0.0 <= inner_px[i, 1] < img_h
            parts.append(f"{kx:.6f} {ky:.6f} {2 if visible else 0}")
        else:
            parts.append("0.0 0.0 0")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


def generate_sample(
    rng: np.random.Generator,
    gate_pos_w: np.ndarray,
    gate_ori_quat: np.ndarray,
    outer_corners_local: np.ndarray,
    inner_corners_local: np.ndarray,
    k: np.ndarray,
    img_w: int,
    img_h: int,
    tilt_deg: float,
) -> tuple[np.ndarray, str] | None:
    """Generate one training sample (image + label).

    Randomises drone position, orientation, gate colour, and applies
    domain randomisation.

    Returns None if the gate is not visible from the sampled viewpoint.
    """
    # Random drone position: 3-15m from gate, random direction
    distance = rng.uniform(3.0, 15.0)

    # Gate normal (local Z in world)
    r_wg = Rotation.from_quat(gate_ori_quat)
    gate_normal = r_wg.apply(np.array([0.0, 0.0, 1.0]))

    # Position the drone roughly along the gate normal (in front of gate)
    # with random lateral offset
    lateral_offset = rng.uniform(-3.0, 3.0)
    vertical_offset = rng.uniform(-1.5, 1.5)

    # Base direction: along gate normal with some deviation
    gate_right = r_wg.apply(np.array([1.0, 0.0, 0.0]))
    gate_up = r_wg.apply(np.array([0.0, 1.0, 0.0]))

    drone_pos_w = (
        gate_pos_w
        + gate_normal * distance
        + gate_right * lateral_offset
        + gate_up * vertical_offset
    )

    # Drone orientation: roughly facing the gate
    to_gate = gate_pos_w - drone_pos_w
    to_gate_norm = to_gate / (np.linalg.norm(to_gate) + 1e-12)
    yaw = float(np.arctan2(to_gate_norm[1], to_gate_norm[0]))
    pitch = float(np.arcsin(np.clip(-to_gate_norm[2], -1, 1)))
    # Add random perturbation
    yaw += rng.uniform(-0.3, 0.3)
    pitch += rng.uniform(-0.2, 0.2)
    roll = rng.uniform(-0.15, 0.15)
    drone_ori_quat = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()

    # Random gate colour
    r = rng.integers(150, 256)
    g = rng.integers(100, 256)
    b = rng.integers(0, 100)
    gate_colour = (int(r), int(g), int(b))

    # Random background colour
    bg_val = rng.integers(80, 180)
    bg_colour = (int(bg_val), int(bg_val), int(bg_val))

    # Render
    image, outer_px, inner_px = render_gate_image(
        rng,
        gate_pos_w,
        gate_ori_quat,
        outer_corners_local,
        inner_corners_local,
        drone_pos_w,
        drone_ori_quat,
        k,
        img_w,
        img_h,
        tilt_deg,
        gate_colour_rgb=gate_colour,
        bg_colour_rgb=bg_colour,
    )

    if outer_px is None:
        return None

    # Make YOLO label
    label = make_yolo_label(outer_px, inner_px, img_w, img_h)
    if label is None:
        return None

    # Apply domain randomisation
    image = apply_domain_randomisation(rng, image)

    return image, label


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(
    output_dir: Path,
    num_train: int,
    num_val: int,
    seed: int = 42,
) -> None:
    """Generate a full YOLO keypoint training dataset.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    num_train : int
        Number of training images.
    num_val : int
        Number of validation images.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    config = load_config()

    gate_cfg = config.gate
    outer_local = gate_cfg.outer_corners_3d_array
    inner_local = gate_cfg.corners_3d_array[4:]  # inner corners
    k = config.camera.intrinsic_matrix
    img_w, img_h = config.camera.width, config.camera.height
    tilt_deg = config.camera.tilt_deg

    # Gate poses: spread around a variety of positions and orientations
    gate_poses = _generate_gate_poses(rng, n=20)

    for split, n_samples in [("train", num_train), ("val", num_val)]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        attempts = 0
        max_attempts = n_samples * 5

        while count < n_samples and attempts < max_attempts:
            attempts += 1

            # Pick a random gate pose
            idx = rng.integers(0, len(gate_poses))
            gate_pos, gate_ori = gate_poses[idx]

            result = generate_sample(
                rng,
                gate_pos,
                gate_ori,
                outer_local,
                inner_local,
                k,
                img_w,
                img_h,
                tilt_deg,
            )
            if result is None:
                continue

            image, label = result

            # Save
            fname = f"{split}_{count:05d}"
            # Convert RGB to BGR for OpenCV imwrite
            cv2.imwrite(str(img_dir / f"{fname}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            (lbl_dir / f"{fname}.txt").write_text(label + "\n")

            count += 1
            if count % 100 == 0:
                logger.info("%s: %d/%d samples generated", split, count, n_samples)

        logger.info("%s: %d samples generated (%d attempts)", split, count, attempts)

    # Write dataset.yaml
    yaml_content = f"""# YOLOv8 Keypoint Dataset — AI Grand Prix Gate Detection
# Auto-generated by generate_training_data.py

path: {output_dir.resolve()}
train: images/train
val: images/val

# Class names
names:
  0: gate

# Keypoint definition: 8 keypoints (4 outer + 4 inner corners)
kpt_shape: [8, 3]  # [num_keypoints, dims_per_kp (x, y, visibility)]
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)
    logger.info("Dataset config written to %s", output_dir / "dataset.yaml")


def _generate_gate_poses(
    rng: np.random.Generator,
    n: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate diverse gate poses for training data variety.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n : int
        Number of gate poses to generate.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (position_w, orientation_quat) pairs.
    """
    poses = []
    for _ in range(n):
        # Random position in a 40m × 40m × 6m volume
        pos = np.array(
            [
                rng.uniform(-20, 20),
                rng.uniform(-20, 20),
                rng.uniform(1, 7),
            ]
        )
        # Random orientation (yaw + slight pitch/roll)
        yaw = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-0.3, 0.3)
        roll = rng.uniform(-0.1, 0.1)
        ori = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_quat()
        poses.append((pos, ori))
    return poses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate synthetic YOLO keypoint training data for gate detection."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gate_keypoints",
        help="Output directory (default: data/gate_keypoints)",
    )
    parser.add_argument("--num-train", type=int, default=2000, help="Training samples")
    parser.add_argument("--num-val", type=int, default=200, help="Validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        output_dir=Path(args.output),
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
    )
    logger.info("Done! Dataset at %s", args.output)


if __name__ == "__main__":
    main()
