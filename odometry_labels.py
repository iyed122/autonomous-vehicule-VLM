"""
odometry_labels.py
==================
Self-supervised trajectory labels: the car's real future path, recovered from the
LiDAR itself, for every local key-frame. No ego poses or annotations needed.

For each pair of consecutive scans (0.5 s apart) the later scan is registered onto the
earlier one in the ego frame: a search over yaw (±8°, 0.5° steps, on a coarse 0.4 m
grid), then the translation at full 0.2 m resolution from an FFT cross-correlation of
occupancy grids built from returns 0.3-3 m above the road. That gives the car's motion
(forward, left, yaw) between the scans. Chaining 12 steps gives the positions 0.5 s to
6.0 s ahead in the current frame, resampled to the same 20 waypoints the planner was
trained on.

The script also runs the trained planner's encoder once per frame (and once on the
left-right mirrored raster) and stores the 256-d scene embedding, so motion_planner.py
can train new heads in seconds on a CPU.

    python odometry_labels.py     # -> inference_outputs/motion_dataset.npz
"""

from pathlib import Path

import numpy as np

from collision_risk import sensor_to_ego
from data_preparation_v2 import load_lidar_points, create_bev_representation
from scene_dashboard import group_scenes

ROOT = Path(__file__).resolve().parent
LIDAR_DIR = ROOT / "LIDAR_TOP"
OUT = ROOT / "inference_outputs"
DT, STEPS, WAYPOINTS = 0.5, 12, 20
SENSOR_HEIGHT = 1.84


def ego_points(points):
    """Returns 0.3-3 m above the road (mostly static structure), ego frame (x fwd, y left)."""
    h = points[:, 2] + SENSOR_HEIGHT
    keep = (h > 0.3) & (h < 3.0)
    x, y = sensor_to_ego(points[keep, 0], points[keep, 1])
    return np.stack([x, y], axis=1)


def grid(xy, res, half=50.0):
    n = int(2 * half / res)
    g = np.zeros((n, n), np.float32)
    ix = ((xy[:, 0] + half) / res).astype(np.int64)
    iy = ((xy[:, 1] + half) / res).astype(np.int64)
    m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    np.add.at(g, (ix[m], iy[m]), 1.0)
    return np.minimum(g, 3.0)


def shift(ga, gb):
    """Translation (cells) with p_a = p_b + shift, and the normalised peak height."""
    c = np.fft.irfft2(np.fft.rfft2(ga) * np.conj(np.fft.rfft2(gb)), s=ga.shape)
    k0, k1 = np.unravel_index(int(np.argmax(c)), c.shape)
    n0, n1 = c.shape
    q = float(c.max() / (np.sqrt((ga * ga).sum() * (gb * gb).sum()) + 1e-9))
    return (k0 + n0 // 2) % n0 - n0 // 2, (k1 + n1 // 2) % n1 - n1 // 2, q


def rotate(xy, psi):
    c, s = np.cos(psi), np.sin(psi)
    return xy @ np.array([[c, s], [-s, c]])


def register(a_xy, b_xy):
    """Motion of the car from scan a to scan b: (forward m, left m, yaw rad, quality)."""
    ga = grid(a_xy, 0.4)
    best = (-1.0, 0.0)
    for psi in np.radians(np.arange(-8.0, 8.01, 0.5)):
        q = shift(ga, grid(rotate(b_xy, psi), 0.4))[2]
        if q > best[0]:
            best = (q, psi)
    psi = best[1]
    d0, d1, q = shift(grid(a_xy, 0.2), grid(rotate(b_xy, psi), 0.2))
    return d0 * 0.2, d1 * 0.2, float(psi), q


def future_path(steps):
    """Chain per-step motions into positions in the first frame: (len(steps), 2)."""
    pos, heading, out = np.zeros(2), 0.0, []
    for fx, fy, psi, _ in steps:
        c, s = np.cos(heading), np.sin(heading)
        pos = pos + np.array([c * fx - s * fy, s * fx + c * fy])
        heading += psi
        out.append(pos.copy())
    return np.array(out)


def resample(path, n=WAYPOINTS):
    t0 = np.linspace(0, 1, len(path))
    t1 = np.linspace(0, 1, n)
    return np.stack([np.interp(t1, t0, path[:, 0]), np.interp(t1, t0, path[:, 1])], axis=1)


def main():
    import torch
    from safety_report import load_planner

    frames = sorted(LIDAR_DIR.glob("*.bin"))
    model, device = load_planner()
    emb, emb_flip, motion, target, valid, scene_id, quality = [], [], [], [], [], [], []
    for k, (start, end) in enumerate(group_scenes(frames)):
        pts = [load_lidar_points(str(f)) for f in frames[start:end]]
        xy = [ego_points(p) for p in pts]
        steps = [register(xy[j], xy[j + 1]) for j in range(len(xy) - 1)]
        with torch.no_grad():
            for j, p in enumerate(pts):
                t = torch.from_numpy(create_bev_representation(p)).permute(2, 0, 1).unsqueeze(0).float()
                for tensor, store in ((t, emb), (torch.flip(t, dims=[2]), emb_flip)):
                    fmap = model.bev_encoder(tensor.to(device))
                    pooled = torch.nn.functional.adaptive_avg_pool2d(fmap, 1).flatten(1)
                    store.append(model.fc_features(pooled)[0].cpu().numpy())
                # motion over the last 0.5 s (unknown for a scene's first frame)
                prev = steps[j - 1] if j > 0 else None
                motion.append([prev[0] / DT, prev[1] / DT, prev[2] / DT] if prev else [np.nan] * 3)
                fut = steps[j:j + STEPS]
                ok = prev is not None and len(fut) == STEPS
                q = min([prev[3]] + [s[3] for s in fut]) if ok else 0.0
                target.append(resample(future_path(fut)) if ok else np.full((WAYPOINTS, 2), np.nan))
                valid.append(ok)
                scene_id.append(k)
                quality.append(q)
        print(f"scene {k}: {end - start} frames, median step quality "
              f"{np.median([s[3] for s in steps]) if steps else 0:.2f}", flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "motion_dataset.npz", emb=np.array(emb), emb_flip=np.array(emb_flip),
                        motion=np.array(motion), target=np.array(target), valid=np.array(valid),
                        scene=np.array(scene_id), quality=np.array(quality),
                        files=np.array([f.name for f in frames]))
    print("saved", OUT / "motion_dataset.npz")


if __name__ == "__main__":
    main()
