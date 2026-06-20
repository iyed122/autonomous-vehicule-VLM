"""
collision_risk.py
=================
The piece that turns two model outputs into a *decision*.

Perception (where are the obstacles, from the LiDAR BEV) and planning (where is
the ego vehicle about to go, from the trajectory model) are only valuable to an
ADAS when they are fused into a single answer: *is my planned path about to hit
something, and how urgently should I react?*

This module is pure NumPy (no torch), so the risk logic can be unit-tested and
run anywhere. It is consumed by:
  * app.py          (live single-frame risk in the demo)
  * risk_monitor.py (annotated video over a driving sequence)
  * safety_report.py(batch safety report)

Method
------
1. Obstacle map: a BEV cell counts as an obstacle if it is both *dense* (enough
   LiDAR returns) and *elevated* (max height clearly above the road surface).
   The height test removes flat ground returns, which a raw density map keeps.
2. Ego deadzone: returns within `EGO_DEADZONE_M` of the origin are the vehicle's
   own body / immediate ground and are ignored (standard in real sensor stacks).
3. Corridor scan: walk along the predicted path; at each waypoint test a
   half-vehicle-width corridor against the obstacle map. The first hit is the
   nearest in-path obstacle.
4. Decision: map nearest-obstacle distance to BRAKE / CAUTION / CLEAR, and (if a
   speed is available) to a time-to-obstacle.

All thresholds live in `RiskConfig` so they are explicit and tunable.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RiskConfig:
    x_range: Tuple[float, float] = (-50.0, 50.0)
    y_range: Tuple[float, float] = (-50.0, 50.0)
    resolution: float = 0.2          # metres / cell  (matches data_preparation_v2)
    corridor_half_width_m: float = 1.0   # half the ego vehicle width
    ego_deadzone_m: float = 3.0          # ignore the vehicle's own returns
    density_thresh: float = 0.2          # normalised density (>= ~2 returns/cell)
    height_thresh: float = 0.55          # normalised height; >road surface
    brake_dist_m: float = 8.0            # nearest obstacle below this -> BRAKE
    caution_dist_m: float = 20.0         # below this -> CAUTION, else CLEAR
    scan_samples: int = 90               # densify the path before scanning


LEVELS = {"CLEAR": 0, "CAUTION": 1, "BRAKE": 2}
LEVEL_COLORS = {"CLEAR": "#2ecc71", "CAUTION": "#f39c12", "BRAKE": "#e74c3c"}


@dataclass
class RiskResult:
    level: str                       # CLEAR / CAUTION / BRAKE
    nearest_dist_m: float            # inf if path is clear
    ttc_s: Optional[float]           # time-to-obstacle, if speed known
    hit_point_m: Optional[Tuple[float, float]]  # obstacle location (ego frame)
    speed_mps: Optional[float]


def obstacle_map(bev: np.ndarray, cfg: RiskConfig = RiskConfig()) -> np.ndarray:
    """Boolean obstacle grid from a (H, W, 3) BEV raster (height/intensity/density)."""
    height = bev[:, :, 0]
    density = bev[:, :, 2]
    return (density > cfg.density_thresh) & (height > cfg.height_thresh)


def _m_to_grid(px: float, py: float, cfg: RiskConfig):
    gx = int((px - cfg.x_range[0]) / cfg.resolution)
    gy = int((py - cfg.y_range[0]) / cfg.resolution)
    return gx, gy


def _densify(path: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline path to ~n points by arc-length for a continuous scan."""
    if len(path) < 2:
        return path
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if seg[-1] == 0:
        return path
    u = np.linspace(0, seg[-1], n)
    x = np.interp(u, seg, path[:, 0])
    y = np.interp(u, seg, path[:, 1])
    return np.stack([x, y], axis=1)


def estimate_speed_mps(path: np.ndarray, horizon_s: float) -> Optional[float]:
    """Crude ego speed = path length / horizon. `horizon_s` = seconds the
    predicted waypoints span (nuScenes key-frames are 0.5 s apart)."""
    if len(path) < 2 or horizon_s <= 0:
        return None
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return length / horizon_s


def assess_risk(path_xy: np.ndarray,
                bev: np.ndarray,
                cfg: RiskConfig = RiskConfig(),
                horizon_s: float = 4.0) -> RiskResult:
    """Fuse a predicted path with the BEV obstacle map into a risk decision.

    Args:
        path_xy : (N, 2) predicted ego waypoints in metres (x forward, y left).
        bev     : (H, W, 3) BEV raster from create_bev_representation.
        horizon_s: time the waypoints span, for the speed/TTC estimate.
    """
    occ = obstacle_map(bev, cfg)
    H, W = occ.shape
    hw = max(1, int(cfg.corridor_half_width_m / cfg.resolution))

    path = _densify(np.asarray(path_xy, dtype=float), cfg.scan_samples)
    speed = estimate_speed_mps(path, horizon_s)

    nearest = np.inf
    hit_point = None
    for px, py in path:
        r = float(np.hypot(px, py))
        if r < cfg.ego_deadzone_m:
            continue
        gx, gy = _m_to_grid(px, py, cfg)
        if not (0 <= gx < H and 0 <= gy < W):
            continue
        win = occ[max(0, gx - hw):gx + hw + 1, max(0, gy - hw):gy + hw + 1]
        if win.size and win.any():
            nearest = r
            hit_point = (px, py)
            break

    if nearest < cfg.brake_dist_m:
        level = "BRAKE"
    elif nearest < cfg.caution_dist_m:
        level = "CAUTION"
    else:
        level = "CLEAR"

    ttc = (nearest / speed) if (speed and np.isfinite(nearest) and speed > 0.1) else None
    return RiskResult(level=level, nearest_dist_m=nearest, ttc_s=ttc,
                      hit_point_m=hit_point, speed_mps=speed)


def straight_fallback_path(length_m: float = 30.0, n: int = 20) -> np.ndarray:
    """A simple constant-velocity forward path, useful as a baseline / fallback."""
    return np.stack([np.linspace(0, length_m, n), np.zeros(n)], axis=1)


if __name__ == "__main__":
    # Self-test on a real bundled LiDAR frame (no torch needed).
    import glob
    from data_preparation_v2 import load_lidar_points, create_bev_representation

    cfg = RiskConfig()
    frames = sorted(glob.glob("LIDAR_TOP/*.bin"))[:5]
    for f in frames:
        bev = create_bev_representation(load_lidar_points(f))
        res = assess_risk(straight_fallback_path(45, 30), bev, cfg)
        ttc = f"{res.ttc_s:.1f}s" if res.ttc_s else "n/a"
        d = f"{res.nearest_dist_m:.1f}m" if np.isfinite(res.nearest_dist_m) else "clear"
        print(f"{f.split('/')[-1][:40]:40s}  {res.level:8s} nearest={d:8s} ttc={ttc}")
