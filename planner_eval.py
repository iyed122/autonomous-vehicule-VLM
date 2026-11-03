"""
planner_eval.py
===============
Is the learned path better than the simplest baseline, "the car keeps its heading"?

The local data has no ego poses, so the car's real motion is recovered from the LiDAR
itself: consecutive scans 1 s apart are aligned (collision_risk.scan_motion) and the
shifts are chained up to 3 s ahead. Frames where the alignment is weak or the car is
nearly stopped are skipped. The planner's position at 3 s is interpolated from its 20
waypoints (0.5 s to 6.0 s).

    python planner_eval.py            # writes inference_outputs/planner_eval.md
    python planner_eval.py --from-csv # re-report from inference_outputs/planner_eval.csv

Translation-only alignment drifts in sharp turns, so treat the numbers as a sanity check
of the checkpoint, not a benchmark.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from collision_risk import scan_motion
from data_preparation_v2 import load_lidar_points, create_bev_representation
from safety_report import load_planner, predict_path
from scene_dashboard import group_scenes

ROOT = Path(__file__).resolve().parent
LIDAR_DIR = ROOT / "LIDAR_TOP"
OUT = ROOT / "inference_outputs"
AHEAD_S, STEP = 3.0, 2          # 3 s ahead, chained in 1 s (2 key-frame) steps


def position_at(path, t_s):
    """Interpolate a 20-waypoint path spanning 0.5-6.0 s at time t_s."""
    j = (t_s - 0.5) / 5.5 * (len(path) - 1)
    lo = int(np.floor(j))
    w = j - lo
    return path[lo] * (1 - w) + path[lo + 1] * w


def measure(every=9):
    frames = sorted(LIDAR_DIR.glob("*.bin"))
    model, device = load_planner()
    bev = lambda f: create_bev_representation(load_lidar_points(str(f)))
    n_ahead = int(AHEAD_S / 0.5)
    rows = []
    for start, end in group_scenes(frames):
        for i in range(start, end - n_ahead, every):
            fwd = left = 0.0
            weak = False
            for k in range(0, n_ahead, STEP):
                df, dl, q = scan_motion(bev(frames[i + k]), bev(frames[i + k + STEP]))
                weak |= q < 0.2
                fwd += df
                left += dl
            if weak or fwd < 3.0:
                continue
            p = position_at(predict_path(model, device, bev(frames[i])), AHEAD_S)
            rows.append((fwd, left, float(p[0]), float(p[1])))
    r = np.array(rows)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "planner_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["real_fwd_m", "real_left_m", "planner_fwd_m", "planner_left_m"])
        w.writerows(np.round(r, 2).tolist())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()
    if args.from_csv:
        with open(OUT / "planner_eval.csv", encoding="utf-8") as f:
            r = np.array([[float(x) for x in row] for row in list(csv.reader(f))[1:]])
    else:
        r = measure()
    # Means, not medians: the alignment moves in 0.2 m steps, so most real sideways
    # shifts round to exactly 0 and a median would flatter the baseline.
    lat_model = np.mean(np.abs(r[:, 3] - r[:, 1]))
    lat_heading = np.mean(np.abs(r[:, 1]))
    fwd_model = np.mean(np.abs(r[:, 2] - r[:, 0]))
    corr = np.corrcoef(r[:, 3], r[:, 1])[0, 1]
    corr_fwd = np.corrcoef(r[:, 2], r[:, 0])[0, 1]
    text = "\n".join([
        "# Planner vs the car's real motion (3 s ahead)",
        "",
        f"{len(r)} frames where the car was moving and consecutive scans aligned well.",
        "",
        "| 3 s ahead | mean absolute error |",
        "|---|---|",
        f"| Sideways, learned planner (epoch 5) | {lat_model:.2f} m |",
        f"| Sideways, 'keeps its heading' baseline | {lat_heading:.2f} m |",
        f"| Forward distance, learned planner | {fwd_model:.1f} m (real median {np.median(r[:, 0]):.1f} m, "
        f"planner median {np.median(r[:, 2]):.1f} m) |",
        "",
        f"Correlation between predicted and real sideways motion: {corr:.2f}; forward distance: "
        f"{corr_fwd:.2f} (it tracks speed, but overshoots). The alignment moves in 0.2 m steps.",
        "",
        "Verdict: at this checkpoint the risk check should not follow the learned path. The cause "
        "found in the training code: the 'mirror' augmentation flipped the raster's forward axis "
        "while mirroring the target path sideways (see the technical report).",
    ])
    (OUT / "planner_eval.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
