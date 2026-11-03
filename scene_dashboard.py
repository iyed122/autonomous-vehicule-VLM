"""
scene_dashboard.py
==================
A synchronized perception+planning dashboard frame, and a sequence -> GIF builder.

Why a side-by-side dashboard rather than LiDAR-points-painted-on-the-image?
True LiDAR->camera projection needs the sensor calibration (camera intrinsics +
the LiDAR->camera extrinsic transform). Those live in the nuScenes metadata and
are not bundled with this package. The field-standard way to present perception
results without calibration — and what the nuScenes devkit itself renders — is a
multi-panel view: the camera with detections next to the bird's-eye-view with the
planned path and collision risk. If you later add the calibration matrices,
`project_lidar_to_image()` below is the hook to switch to a true overlay.

Each dashboard frame is:
    ┌───────────────────────────┬───────────────────────────┐
    │  front camera + detections │  BEV LiDAR + path + risk   │
    └───────────────────────────┴───────────────────────────┘
with a risk banner on top. A sequence of frames is encoded to an animated GIF
(and optionally MP4).

This module's rendering is import-light (numpy + matplotlib + PIL) so the layout
and GIF encoding can be exercised without torch.
"""

import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collision_risk import (RiskConfig, obstacle_map, LEVEL_COLORS, bev_for_display,
                            obstacle_points_ego)

ROOT = Path(__file__).resolve().parent


def render_dashboard_frame(bev, path, risk, idx, n,
                           camera_rgb=None, camera_caption="Front camera"):
    """Compose one dashboard frame -> RGB uint8 array.

    Args:
        bev        : (H, W, 3) BEV raster.
        path       : (N, 2) predicted ego waypoints (m).
        risk       : RiskResult from collision_risk.assess_risk.
        camera_rgb : optional (h, w, 3) RGB image (ideally already annotated
                     with detections). If None, only the BEV panel is drawn.
    """
    cfg = RiskConfig()
    has_cam = camera_rgb is not None
    fig, axes = plt.subplots(1, 2 if has_cam else 1,
                             figsize=(13.5 if has_cam else 7.5, 6.8), dpi=100)
    axes = np.atleast_1d(axes)
    bev_ax = axes[1] if has_cam else axes[0]

    if has_cam:
        cam_ax = axes[0]
        cam_ax.imshow(camera_rgb)
        cam_ax.set_title(camera_caption, fontsize=11)
        cam_ax.axis("off")

    # --- BEV panel ---
    img, extent = bev_for_display(bev[:, :, 2], cfg)
    bev_ax.imshow(img, cmap="gray", origin="lower", extent=extent, alpha=0.85)
    ox, oy = obstacle_points_ego(obstacle_map(bev, cfg), cfg)
    if len(ox):
        bev_ax.scatter(ox, oy, s=1, c="#c0392b", alpha=0.22)
    bev_ax.plot(path[:, 0], path[:, 1], color="#3498db", lw=3, label="Predicted path")
    bev_ax.scatter([0], [0], c="cyan", marker="*", s=240, edgecolors="k",
                   zorder=6, label="Ego")
    if risk.hit_point_m is not None:
        bev_ax.scatter([risk.hit_point_m[0]], [risk.hit_point_m[1]], s=220,
                       facecolors="none", edgecolors=LEVEL_COLORS[risk.level],
                       linewidths=3, zorder=7, label="Nearest obstacle")
    bev_ax.set_xlim(-50, 50); bev_ax.set_ylim(-50, 50); bev_ax.set_aspect("equal")
    bev_ax.set_xlabel("X (m) forward"); bev_ax.set_ylabel("Y (m) left")
    bev_ax.grid(alpha=0.25); bev_ax.legend(loc="lower right", fontsize=8)
    bev_ax.set_title("BEV LiDAR · planned path · obstacles", fontsize=11)

    # --- risk banner + readout ---
    dist = "clear" if not np.isfinite(risk.nearest_dist_m) else f"{risk.nearest_dist_m:.1f} m"
    ttc = f"{risk.ttc_s:.1f} s" if risk.ttc_s else "—"
    spd = f"{risk.speed_mps*3.6:.0f} km/h" if risk.speed_mps else "—"
    fig.suptitle(f"{risk.level}", fontsize=20, fontweight="bold", color="white",
                 x=0.5, y=0.985,
                 bbox=dict(boxstyle="round,pad=0.35", fc=LEVEL_COLORS[risk.level], ec="none"))
    fig.text(0.5, 0.92, f"obstacle in path: {dist}   ·   TTC: {ttc}   ·   ego: {spd}"
             f"      (frame {idx+1}/{n})", ha="center", va="center", fontsize=10)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.05, right=0.97, wspace=0.12)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return img


def frames_to_gif_bytes(frames_rgb, fps=8, scale=0.6):
    """Encode a list of RGB frames into GIF bytes (in-memory)."""
    from PIL import Image
    ims = []
    for f in frames_rgb:
        im = Image.fromarray(f)
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)))
        ims.append(im)
    buf = io.BytesIO()
    ims[0].save(buf, format="GIF", save_all=True, append_images=ims[1:],
                duration=int(1000 / fps), loop=0, optimize=True)
    return buf.getvalue()


def frames_to_mp4(frames_rgb, path_mp4, fps=8):
    import cv2
    h, w = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(str(path_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def project_lidar_to_image(points, image, cam_intrinsic, lidar_to_cam):
    """Hook for a TRUE LiDAR->camera overlay once calibration is available.

    Args:
        points       : (N, 3+) LiDAR points in the LiDAR frame.
        image        : (h, w, 3) camera image.
        cam_intrinsic: (3, 3) camera intrinsic matrix K.
        lidar_to_cam : (4, 4) homogeneous transform LiDAR -> camera.
    Returns the image with depth-coloured points drawn on it.

    Not used by the default dashboard because this package ships without the
    nuScenes calibration metadata; included so the upgrade path is explicit.
    """
    pts = np.asarray(points)[:, :3]
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    cam = (lidar_to_cam @ hom.T).T[:, :3]
    cam = cam[cam[:, 2] > 0.1]                      # keep points in front of camera
    proj = (cam_intrinsic @ cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    depth = cam[:, 2]
    h, w = image.shape[:2]
    m = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    out = image.copy()
    import matplotlib.cm as cm
    if m.any():
        d = depth[m]; norm = (d - d.min()) / (d.ptp() + 1e-6)
        colors = (cm.jet(1 - norm)[:, :3] * 255).astype(np.uint8)
        for (u, v), c in zip(uv[m].astype(int), colors):
            out[max(0, v-1):v+2, max(0, u-1):u+2] = c
    return out


# --------------------------------------------------------------------------- #
# Frame selection & camera<->LiDAR synchronisation helpers
# --------------------------------------------------------------------------- #
def parse_timestamp(path):
    """Trailing microsecond timestamp from a nuScenes filename, or None."""
    import re
    m = re.findall(r"(\d{10,})", Path(path).stem)
    return int(m[-1]) if m else None


def group_scenes(files, gap_s=2.0):
    """Split a time-sorted list of frames into continuous scenes wherever the
    timestamp gap exceeds `gap_s`. Returns [(start_idx, end_idx_exclusive), ...]."""
    ts = [parse_timestamp(f) for f in files]
    if not files or any(t is None for t in ts):
        return [(0, len(files))]
    scenes, start = [], 0
    for i in range(1, len(files)):
        if abs(ts[i] - ts[i - 1]) / 1e6 > gap_s:   # abs: a new log can start earlier in time
            scenes.append((start, i)); start = i
    scenes.append((start, len(files)))
    return scenes


def filter_front_camera(cam_files):
    """Keep only CAM_FRONT frames if a folder mixes the six nuScenes cameras.
    nuScenes filenames are token-delimited by '__' (…__CAM_FRONT__<ts>.jpg), so
    we match the exact 'CAM_FRONT' token and thus exclude CAM_FRONT_LEFT/_RIGHT.
    Falls back to all files if none match (e.g. a non-nuScenes naming scheme)."""
    def is_front(f):
        return any(tok == "CAM_FRONT" for tok in Path(f).name.split("__"))
    front = [f for f in cam_files if is_front(f)]
    return front or cam_files


def match_cameras_by_timestamp(lidar_files, cam_files, max_dt_s=0.5):
    """For each LiDAR frame, pick the camera frame with the nearest timestamp
    (within max_dt_s). Returns a list aligned to lidar_files of paths or None."""
    import bisect
    cam_ts = sorted((parse_timestamp(c), c) for c in cam_files
                    if parse_timestamp(c) is not None)
    if not cam_ts:
        return [None] * len(lidar_files)
    times = [t for t, _ in cam_ts]
    out = []
    for lf in lidar_files:
        lt = parse_timestamp(lf)
        if lt is None:
            out.append(None); continue
        j = bisect.bisect_left(times, lt)
        cands = []
        if j < len(times): cands.append(cam_ts[j])
        if j > 0: cands.append(cam_ts[j - 1])
        best = min(cands, key=lambda tc: abs(tc[0] - lt))
        out.append(best[1] if abs(best[0] - lt) / 1e6 <= max_dt_s else None)
    return out


# --------------------------------------------------------------------------- #
# 360° surround-view assembly (6 nuScenes cameras) for the VLM reasoning layer
# --------------------------------------------------------------------------- #
NUSCENES_CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def camera_channel(path):
    """Return the CAM_* channel token from a nuScenes filename, or None."""
    for tok in Path(path).name.split("__"):
        if tok.startswith("CAM_"):
            return tok
    return None


def group_cameras_by_channel(cam_files):
    """Group a flat list of camera frames into {channel: [time-sorted files]}."""
    groups = {}
    for f in cam_files:
        ch = camera_channel(f)
        if ch:
            groups.setdefault(ch, []).append(f)
    for ch in groups:
        groups[ch].sort()
    return groups


def match_surround(lidar_file, cam_files, max_dt_s=0.5):
    """For one LiDAR frame, return {channel: nearest-timestamp camera path or None}
    across all six nuScenes cameras."""
    groups = group_cameras_by_channel(cam_files)
    out = {}
    for ch in NUSCENES_CAMERAS:
        files = groups.get(ch, [])
        out[ch] = match_cameras_by_timestamp([lidar_file], files, max_dt_s)[0] if files else None
    return out


def assemble_surround(cam_paths, tile=(480, 270), bev_rgb=None):
    """Compose a 2x3 surround montage (+ optional BEV tile) into one RGB array.

    Layout:
        FRONT_LEFT | FRONT | FRONT_RIGHT
        BACK_LEFT  | BACK  | BACK_RIGHT
    `cam_paths` is {channel: path or None}. Missing views render as a labelled blank.
    """
    from PIL import Image, ImageDraw
    tw, th = tile
    order = NUSCENES_CAMERAS

    def load_tile(ch):
        p = cam_paths.get(ch)
        if p:
            im = Image.open(p).convert("RGB").resize((tw, th))
        else:
            im = Image.new("RGB", (tw, th), (32, 32, 32))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, tw - 1, th - 1], outline=(90, 90, 90))
        d.text((6, 6), ch + ("" if p else "  (missing)"), fill=(0, 255, 200))
        return np.array(im)

    rows = [np.hstack([load_tile(order[r * 3 + c]) for c in range(3)]) for r in range(2)]
    grid = np.vstack(rows)

    if bev_rgb is not None:
        bev_im = np.array(Image.fromarray(bev_rgb).convert("RGB").resize((grid.shape[1], grid.shape[1] // 3)))
        grid = np.vstack([grid, bev_im])
    return grid


def image_to_png_bytes(rgb):
    """Encode an RGB numpy array to PNG bytes (for sending to the VLM)."""
    import io as _io
    from PIL import Image
    buf = _io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
