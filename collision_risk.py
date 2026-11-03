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
  * app.py           (live single-frame risk in the demo)
  * risk_monitor.py  (annotated video over a driving sequence)
  * safety_report.py (per-scene safety report)
  * fleet_review.py  (every scene, review queue)

Two coordinate frames
---------------------
* The planner predicts waypoints in the **ego frame**: x forward, y left, origin at
  the rear axle (how nuScenes defines ego poses).
* The BEV raster is built from LIDAR_TOP points in the **sensor frame**: x to the
  right, y forward, origin at the roof LiDAR. Raster axis 0 is sensor x, axis 1 is
  sensor y.
`ego_to_sensor` / `sensor_to_ego` convert between them; every lookup into the raster
and every plot goes through them. (Until November 2026 the corridor scan indexed the
raster with ego coordinates directly, i.e. it searched to the right of the car.)

Method
------
1. Obstacle map: a BEV cell counts as an obstacle if it is *dense* (enough LiDAR
   returns) and *tall*: its highest return is at least `min_obstacle_height_m`
   above the road (the road sits `sensor_height_m` below the LiDAR).
2. Corridor scan: walk along the predicted path from the front bumper outward; at
   each point test a vehicle-wide corridor against the obstacle map. The first hit
   is the nearest in-path obstacle; its distance is the gap to the front bumper.
3. Decision: gap below `brake_dist_m` -> BRAKE, below `caution_dist_m` -> CAUTION,
   else CLEAR, plus a time-to-obstacle when a speed is available.

All thresholds live in `RiskConfig` so they are explicit and tunable.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RiskConfig:
    x_range: Tuple[float, float] = (-50.0, 50.0)   # raster extent, sensor x (right)
    y_range: Tuple[float, float] = (-50.0, 50.0)   # raster extent, sensor y (forward)
    resolution: float = 0.2              # metres / cell  (matches data_preparation_v2)
    corridor_half_width_m: float = 1.0   # half the ego vehicle width, plus a margin
    front_bumper_m: float = 3.4          # front bumper, metres ahead of the rear axle
    rear_bumper_m: float = -0.9          # rear bumper (negative: behind the rear axle)
    body_half_width_m: float = 1.0       # half the car's width, plus a margin
    density_thresh: float = 0.2          # normalised density (more than 2 returns/cell)
    sensor_height_m: float = 1.84        # LIDAR_TOP sits ~1.84 m above the road
    min_obstacle_height_m: float = 0.5   # highest return at least this far above the road
    brake_dist_m: float = 5.0            # gap from the front bumper below this -> BRAKE
    caution_dist_m: float = 15.0         # below this -> CAUTION, else CLEAR
    scan_samples: int = 120              # densify the path before scanning


BEV_Z_RANGE = (-3.0, 3.0)   # z window of create_bev_representation (sensor frame, metres)
LIDAR_X_OFFSET_M = 0.94     # LIDAR_TOP is ~0.94 m ahead of the ego origin (rear axle)

LEVELS = {"CLEAR": 0, "CAUTION": 1, "BRAKE": 2}
LEVEL_COLORS = {"CLEAR": "#2ecc71", "CAUTION": "#f39c12", "BRAKE": "#e74c3c"}

# Time between the planner's first and last waypoint. Training targets are the ego
# positions of the next 12 nuScenes key-frames (2 Hz, i.e. 0.5 s to 6.0 s ahead),
# resampled to 20 points, so the 20 waypoints span 5.5 s. Speed and time-to-obstacle
# are derived from it, so every caller should pass this value.
PRED_HORIZON_S = 5.5


@dataclass
class RiskResult:
    level: str                       # CLEAR / CAUTION / BRAKE
    nearest_dist_m: float            # gap from the front bumper; inf if path is clear
    ttc_s: Optional[float]           # time-to-obstacle, if speed known
    hit_point_m: Optional[Tuple[float, float]]  # obstacle location (ego frame)
    speed_mps: Optional[float]


def ego_to_sensor(px, py):
    """Ego frame (x forward, y left, rear axle) -> LIDAR_TOP frame (x right, y forward)."""
    return -np.asarray(py), np.asarray(px) - LIDAR_X_OFFSET_M


def sensor_to_ego(sx, sy):
    """LIDAR_TOP frame (x right, y forward) -> ego frame (x forward, y left, rear axle)."""
    return np.asarray(sy) + LIDAR_X_OFFSET_M, -np.asarray(sx)


def bev_for_display(channel: np.ndarray, cfg: RiskConfig = RiskConfig()):
    """(image, extent) to `imshow(image, origin="lower", extent=extent)` one BEV channel
    in ego-frame axes: x forward to the right of the plot, y left towards the top."""
    extent = [cfg.y_range[0] + LIDAR_X_OFFSET_M, cfg.y_range[1] + LIDAR_X_OFFSET_M,
              -cfg.x_range[1], -cfg.x_range[0]]
    return channel[::-1, :], extent


def obstacle_points_ego(occ: np.ndarray, cfg: RiskConfig = RiskConfig()):
    """Cell centres of an obstacle grid, in ego-frame metres (x forward, y left)."""
    gx, gy = np.where(occ)
    sx = cfg.x_range[0] + (gx + 0.5) * cfg.resolution
    sy = cfg.y_range[0] + (gy + 0.5) * cfg.resolution
    return sensor_to_ego(sx, sy)


def obstacle_map(bev: np.ndarray, cfg: RiskConfig = RiskConfig()) -> np.ndarray:
    """Boolean obstacle grid from a (H, W, 3) BEV raster (height/intensity/density)."""
    height = bev[:, :, 0]
    density = bev[:, :, 2]
    # The height channel is (z - z_min) / (z_max - z_min) in the sensor frame, where the
    # road sits at z = -sensor_height. Convert "0.5 m above the road" into that scale.
    z_min, z_max = BEV_Z_RANGE
    thresh = (cfg.min_obstacle_height_m - cfg.sensor_height_m - z_min) / (z_max - z_min)
    return (density > cfg.density_thresh) & (height > thresh)


def _m_to_grid(px: float, py: float, cfg: RiskConfig):
    """Ego-frame metres -> raster indices (axis 0 = sensor x, axis 1 = sensor y)."""
    sx, sy = ego_to_sensor(px, py)
    gx = int(np.floor((sx - cfg.x_range[0]) / cfg.resolution))
    gy = int(np.floor((sy - cfg.y_range[0]) / cfg.resolution))
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
    """Crude ego speed = path length / horizon. `horizon_s` = seconds between the
    first and last waypoint (PRED_HORIZON_S for the trained planner)."""
    if len(path) < 2 or horizon_s <= 0:
        return None
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return length / horizon_s


def assess_risk(path_xy: np.ndarray,
                bev: np.ndarray,
                cfg: RiskConfig = RiskConfig(),
                horizon_s: float = PRED_HORIZON_S,
                speed_mps: Optional[float] = None) -> RiskResult:
    """Fuse a path with the BEV obstacle map into a risk decision.

    Args:
        path_xy : (N, 2) ego waypoints in metres (x forward, y left): the learned path,
                  or heading_corridor().
        bev     : (H, W, 3) BEV raster from create_bev_representation.
        horizon_s: time the waypoints span, for the path-based speed estimate.
        speed_mps: measured speed (odometry_labels.register); if given, it replaces the
                  path-based estimate for TTC.
    """
    occ = obstacle_map(bev, cfg)
    H, W = occ.shape
    # The roof LiDAR also sees the car itself: drop returns inside its own footprint.
    gx0, gy0 = _m_to_grid(cfg.front_bumper_m, cfg.body_half_width_m, cfg)
    gx1, gy1 = _m_to_grid(cfg.rear_bumper_m, -cfg.body_half_width_m, cfg)
    occ = occ.copy()
    occ[min(gx0, gx1):max(gx0, gx1) + 1, min(gy0, gy1):max(gy0, gy1) + 1] = False
    hw = max(1, int(cfg.corridor_half_width_m / cfg.resolution))

    path = np.asarray(path_xy, dtype=float)
    speed = speed_mps if speed_mps is not None else estimate_speed_mps(path, horizon_s)
    # The first waypoint is already 0.5 s ahead: scan from the car itself.
    path = _densify(np.vstack([[0.0, 0.0], path]), cfg.scan_samples)

    nearest = np.inf
    hit_point = None
    for px, py in path:
        if px < cfg.front_bumper_m:          # still inside the car's own footprint
            continue
        gx, gy = _m_to_grid(px, py, cfg)
        if not (0 <= gx < H and 0 <= gy < W):
            continue
        x0, y0 = max(0, gx - hw), max(0, gy - hw)
        win = occ[x0:gx + hw + 1, y0:gy + hw + 1]
        if win.size and win.any():
            # Measure to the obstacle cells themselves, not to the path point whose
            # window touched them (that would read up to one window short).
            wx, wy = np.where(win)
            cx, cy = sensor_to_ego(cfg.x_range[0] + (x0 + wx + 0.5) * cfg.resolution,
                                   cfg.y_range[0] + (y0 + wy + 0.5) * cfg.resolution)
            k = int(np.argmin(np.hypot(cx, cy)))
            nearest = max(0.0, float(np.hypot(cx[k], cy[k])) - cfg.front_bumper_m)
            hit_point = (float(cx[k]), float(cy[k]))
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


MOVING_MPS = 2.0   # ~7 km/h: below this a close gap is a queue or a stop, not a close call


def effective_level(level: str, speed_mps: Optional[float]) -> str:
    """Risk level that counts for scoring and review: a close gap while stopped or crawling
    (traffic light, queue) is not a close call. Unknown speed keeps the level."""
    if speed_mps is not None and speed_mps < MOVING_MPS:
        return "CLEAR"
    return level


def extend_path(path_xy: np.ndarray, cfg: RiskConfig = RiskConfig()) -> np.ndarray:
    """Extend a predicted path along its last direction until it covers at least the
    CAUTION range ahead of the bumper (a slow car's 5.5 s path can be shorter)."""
    path = np.asarray(path_xy, dtype=float)
    reach = cfg.front_bumper_m + cfg.caution_dist_m + 1.0
    end = path[-1]
    if np.hypot(*end) >= reach:
        return path
    d = path[-1] - path[-4] if len(path) >= 4 else np.array([1.0, 0.0])
    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.05 else np.array([1.0, 0.0])
    extra = [end + d * s for s in np.arange(1.0, reach + 1.0, 1.0)]
    extra = [p for p in extra if np.hypot(*p) <= reach + 1.0]
    return np.vstack([path] + ([np.array(extra)] if extra else []))


def heading_corridor(speed_mps: Optional[float], cfg: RiskConfig = RiskConfig(),
                     horizon_s: float = PRED_HORIZON_S) -> np.ndarray:
    """The path along the car's current heading that the risk check scans: as far as the
    car travels in `horizon_s` at `speed_mps`, and never shorter than the CAUTION range.
    Used when no planner is loaded, or for frames whose motion could not be measured."""
    reach = cfg.front_bumper_m + cfg.caution_dist_m + 1.0
    length = max(reach, (speed_mps or 0.0) * horizon_s + cfg.front_bumper_m)
    return straight_fallback_path(length, 20)


if __name__ == "__main__":
    # Self-test on real bundled LiDAR frames (no torch needed).
    import glob
    from data_preparation_v2 import load_lidar_points, create_bev_representation

    # Frame convention: ego (10 m ahead, 2 m left) is sensor (-2 m, 9.06 m) and back.
    sx, sy = ego_to_sensor(10.0, 2.0)
    assert np.allclose([sx, sy], [-2.0, 10.0 - LIDAR_X_OFFSET_M])
    assert np.allclose(sensor_to_ego(sx, sy), [10.0, 2.0])

    cfg = RiskConfig()
    frames = sorted(glob.glob("LIDAR_TOP/*.bin"))[:5]
    for f in frames:
        bev = create_bev_representation(load_lidar_points(f))
        res = assess_risk(straight_fallback_path(45, 30), bev, cfg)
        ttc = f"{res.ttc_s:.1f}s" if res.ttc_s else "n/a"
        d = f"{res.nearest_dist_m:.1f}m" if np.isfinite(res.nearest_dist_m) else "clear"
        print(f"{f.split('/')[-1][:40]:40s}  {res.level:8s} gap={d:8s} ttc={ttc}")
