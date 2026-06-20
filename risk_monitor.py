"""
risk_monitor.py
===============
The "business value" demo: a continuous collision-risk monitor over a real
driving sequence, not one frame at a time.

For every LiDAR frame in a scene it:
  1. builds the BEV raster,
  2. runs the trained trajectory planner to predict the ego path,
  3. fuses that path with the BEV obstacle map (collision_risk.py) into a
     CLEAR / CAUTION / BRAKE decision with distance-to-obstacle and TTC,
  4. renders an annotated frame with a live risk banner,
and stitches the frames into an MP4 (and a GIF) — the kind of output a fleet
operator or an ADAS validation engineer would actually watch.

Usage:
    python risk_monitor.py                 # first 100 frames -> inference_outputs/risk_monitor.mp4
    python risk_monitor.py --frames 200 --stride 1 --fps 10
    python risk_monitor.py --straight      # use a constant-velocity baseline path
                                           # (skips the neural net; needs no torch)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collision_risk import (RiskConfig, assess_risk, obstacle_map,
                            straight_fallback_path, LEVEL_COLORS)
from data_preparation_v2 import load_lidar_points, create_bev_representation

ROOT = Path(__file__).resolve().parent
LIDAR_DIR = ROOT / "LIDAR_TOP"
OUT = ROOT / "inference_outputs"
BEV_WEIGHTS = ROOT / "checkpoint_v3" / "best_trajectory_model.pth"
KEYFRAME_DT = 0.5  # nuScenes key-frame spacing (s) -> horizon = n_waypoints * dt


def load_planner():
    import torch
    from trajectory_model import TrajectoryPlannerModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryPlannerModel().to(device)
    ckpt = torch.load(str(BEV_WEIGHTS), map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model, device


def predict_path(model, device, bev):
    import torch
    bev_t = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        best, _ = model.predict_best_trajectory(bev_t)
    return best[0].cpu().numpy()


def render_frame(bev, path, occ, risk, idx, n):
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=110)
    ax.imshow(bev[:, :, 2], cmap="gray", origin="lower",
              extent=[-50, 50, -50, 50], alpha=0.85)
    # obstacle cells (downsampled scatter for speed)
    ys, xs = np.where(occ)
    if len(xs):
        mx = xs * 0.2 - 50.0   # NOTE: col index -> y(left); row index -> x(fwd)
        my = ys * 0.2 - 50.0
        ax.scatter(my, mx, s=1, c="#c0392b", alpha=0.25)
    # predicted path
    ax.plot(path[:, 0], path[:, 1], color="#3498db", lw=3, label="Predicted path")
    ax.scatter([0], [0], c="cyan", marker="*", s=240, edgecolors="k",
               zorder=6, label="Ego")
    if risk.hit_point_m is not None:
        ax.scatter([risk.hit_point_m[0]], [risk.hit_point_m[1]], s=220,
                   facecolors="none", edgecolors=LEVEL_COLORS[risk.level],
                   linewidths=3, zorder=7, label="Nearest obstacle")
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_aspect("equal")
    ax.set_xlabel("X (m) forward"); ax.set_ylabel("Y (m) left")
    ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8)

    # risk banner
    color = LEVEL_COLORS[risk.level]
    dist = "clear" if not np.isfinite(risk.nearest_dist_m) else f"{risk.nearest_dist_m:.1f} m"
    ttc = f"{risk.ttc_s:.1f} s" if risk.ttc_s else "—"
    spd = f"{risk.speed_mps*3.6:.0f} km/h" if risk.speed_mps else "—"
    ax.text(0.5, 1.14, f"{risk.level}", transform=ax.transAxes, ha="center",
            va="center", fontsize=22, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="none"))
    ax.text(0.5, 1.05, f"obstacle in path: {dist}   ·   TTC: {ttc}   ·   ego: {spd}",
            transform=ax.transAxes, ha="center", va="center", fontsize=10)
    ax.text(0.01, 1.05, f"frame {idx+1}/{n}", transform=ax.transAxes, ha="left",
            va="center", fontsize=9, color="#888")
    fig.suptitle("Collision-risk monitor", fontsize=9, color="#888", y=0.995)
    fig.subplots_adjust(top=0.86)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return img


def write_video(frames_rgb, path_mp4, path_gif, fps):
    import cv2
    h, w = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(str(path_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    # GIF (downscaled) via Pillow
    try:
        from PIL import Image
        ims = [Image.fromarray(f).resize((w // 2, h // 2)) for f in frames_rgb[::2]]
        ims[0].save(path_gif, save_all=True, append_images=ims[1:],
                    duration=int(1000 / fps), loop=0, optimize=True)
    except Exception as e:
        print("GIF skipped:", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--straight", action="store_true",
                    help="use a constant-velocity baseline path (no torch needed)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    cfg = RiskConfig()
    frames = sorted(LIDAR_DIR.glob("*.bin"))[:args.frames * args.stride:args.stride]
    if not frames:
        raise FileNotFoundError(f"No .bin frames in {LIDAR_DIR}")
    horizon = 20 * KEYFRAME_DT

    model = device = None
    if not args.straight:
        model, device = load_planner()
        print(f"Loaded planner on {device}")

    rendered, counts = [], {"CLEAR": 0, "CAUTION": 0, "BRAKE": 0}
    for i, f in enumerate(frames):
        bev = create_bev_representation(load_lidar_points(str(f)))
        path = (straight_fallback_path(30, 20) if args.straight
                else predict_path(model, device, bev))
        risk = assess_risk(path, bev, cfg, horizon_s=horizon)
        counts[risk.level] += 1
        rendered.append(render_frame(bev, path, obstacle_map(bev, cfg), risk, i, len(frames)))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(frames)} frames…")

    mp4 = OUT / "risk_monitor.mp4"
    gif = OUT / "risk_monitor.gif"
    write_video(rendered, mp4, gif, args.fps)
    print(f"\nRisk distribution: {counts}")
    print(f"Saved {mp4}")
    print(f"Saved {gif}")


if __name__ == "__main__":
    main()
