"""
safety_report.py
================
The fleet / insurance-telematics deliverable: run a whole driving sequence
through perception + planning + collision-risk fusion, then produce a single
PDF safety report — the artefact a fleet-safety or underwriting team would read.

It quantifies, for a scene:
  * how much of the drive was spent at each risk level (CLEAR/CAUTION/BRAKE),
  * a risk timeline (level + obstacle distance + estimated speed over time),
  * discrete "risk events" (contiguous stretches of elevated risk) with the
    closest approach and time-to-obstacle for each,
  * a one-line scene safety score.

Usage:
    python safety_report.py                      # first scene in LIDAR_TOP/
    python safety_report.py --scene 12           # any scene (0-based, see fleet_review.py)
    python safety_report.py --path model         # scan along the learned path instead

Output: inference_outputs/safety_report.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from collision_risk import (RiskConfig, assess_risk, straight_fallback_path,
                            LEVELS, LEVEL_COLORS, PRED_HORIZON_S)
from data_preparation_v2 import load_lidar_points, create_bev_representation

ROOT = Path(__file__).resolve().parent
LIDAR_DIR = ROOT / "LIDAR_TOP"
OUT = ROOT / "inference_outputs"
BEV_WEIGHTS = ROOT / "checkpoint_v3" / "best_trajectory_model.pth"
KEYFRAME_DT = 0.5


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
    t = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        best, _ = model.predict_best_trajectory(t)
    return best[0].cpu().numpy()


def find_events(levels, dists, ttcs, dt):
    """Group contiguous CAUTION/BRAKE frames into discrete risk events."""
    events, i, n = [], 0, len(levels)
    while i < n:
        if levels[i] == "CLEAR":
            i += 1
            continue
        j = i
        while j < n and levels[j] != "CLEAR":
            j += 1
        seg_d = dists[i:j]
        seg_t = [t for t in ttcs[i:j] if t is not None]
        peak = "BRAKE" if "BRAKE" in levels[i:j] else "CAUTION"
        events.append({
            "start": i, "end": j - 1, "frames": j - i,
            "duration_s": (j - i) * dt,
            "peak": peak,
            "min_dist": float(np.nanmin(seg_d)) if len(seg_d) else float("nan"),
            "min_ttc": float(np.nanmin(seg_t)) if seg_t else None,
        })
        i = j
    return events


def safety_score(pct, n_brake_events):
    """Simple 0-100 safety score: penalise time-at-risk and proximity."""
    return max(0.0, 100.0 - pct["CAUTION"] * 0.5 - pct["BRAKE"] * 1.5
               - n_brake_events * 2.0)


def build_report(levels, dists, ttcs, speeds, dt, scene_name, n_frames, mode):
    counts = {k: levels.count(k) for k in LEVELS}
    pct = {k: 100.0 * counts[k] / max(1, len(levels)) for k in LEVELS}
    events = find_events(levels, dists, ttcs, dt)
    brake_events = [e for e in events if e["peak"] == "BRAKE"]
    finite_d = np.array([d for d in dists if np.isfinite(d)])
    min_dist = finite_d.min() if len(finite_d) else float("nan")
    score = safety_score(pct, len(brake_events))

    pp = PdfPages(str(OUT / "safety_report.pdf"))

    # ---- Page 1: summary ----
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.suptitle("Drive Safety Report", fontsize=20, fontweight="bold", y=0.97)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    band = LEVEL_COLORS["BRAKE"] if score < 60 else (
        LEVEL_COLORS["CAUTION"] if score < 85 else LEVEL_COLORS["CLEAR"])
    ax.text(0.5, 0.90, f"Scene safety score: {score:.0f}/100",
            ha="center", fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=band, ec="none", alpha=0.85),
            color="white")
    lines = [
        f"Scene             : {scene_name}",
        f"Frames analysed   : {n_frames}  (~{n_frames*dt:.0f} s at {1/dt:.0f} Hz key-frames)",
        f"Path checked      : {'learned planner path' if mode == 'model' else 'current heading, LiDAR-measured speed'}",
        "",
        f"Time CLEAR        : {pct['CLEAR']:.0f}%   ({counts['CLEAR']} frames)",
        f"Time CAUTION      : {pct['CAUTION']:.0f}%   ({counts['CAUTION']} frames)",
        f"Time BRAKE        : {pct['BRAKE']:.0f}%   ({counts['BRAKE']} frames)",
        "",
        f"Risk events       : {len(events)}   (of which {len(brake_events)} reached BRAKE)",
        f"Closest approach  : {min_dist:.1f} m" if np.isfinite(min_dist) else "Closest approach  : n/a",
    ]
    ax.text(0.12, 0.82, "\n".join(lines), va="top", ha="left",
            fontsize=11, family="monospace")

    # risk composition bar
    ax2 = fig.add_axes([0.12, 0.40, 0.76, 0.18])
    left = 0
    for lvl in ["CLEAR", "CAUTION", "BRAKE"]:
        ax2.barh(0, pct[lvl], left=left, color=LEVEL_COLORS[lvl],
                 edgecolor="white", label=f"{lvl} {pct[lvl]:.0f}%")
        left += pct[lvl]
    ax2.set_xlim(0, 100); ax2.set_yticks([]); ax2.set_xlabel("% of drive")
    ax2.set_title("Risk composition", fontsize=11)
    ax2.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.35), fontsize=9)

    ax.text(0.12, 0.30, "Top risk events", fontsize=12, fontweight="bold")
    rows = ["frame range     dur    peak     min dist   min TTC"]
    for e in sorted(events, key=lambda x: x["min_dist"])[:8]:
        ttc = f"{e['min_ttc']:.1f}s" if e["min_ttc"] else "  —"
        rows.append(f"{e['start']:>4}–{e['end']:<4}   {e['duration_s']:>4.1f}s  "
                    f"{e['peak']:<7} {e['min_dist']:>6.1f} m   {ttc:>6}")
    ax.text(0.12, 0.27, "\n".join(rows) if len(rows) > 1 else "No risk events.",
            va="top", ha="left", fontsize=9.5, family="monospace")
    ax.text(0.5, 0.04, "Generated by safety_report.py · perception + planning + "
            "collision-risk fusion on nuScenes LiDAR", ha="center",
            fontsize=8, color="#888")
    pp.savefig(fig); plt.close(fig)

    # ---- Page 2: timelines ----
    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), sharex=True)
    fig.suptitle("Risk timeline", fontsize=16, fontweight="bold", y=0.96)
    t = np.arange(len(levels)) * dt
    lvl_num = [LEVELS[l] for l in levels]
    axes[0].step(t, lvl_num, where="mid", color="#34495e")
    axes[0].fill_between(t, lvl_num, step="mid", alpha=0.25, color="#34495e")
    axes[0].set_yticks([0, 1, 2]); axes[0].set_yticklabels(["CLEAR", "CAUTION", "BRAKE"])
    axes[0].set_title("Risk level"); axes[0].grid(alpha=0.3)

    d_plot = [min(d, 50) if np.isfinite(d) else 50 for d in dists]
    axes[1].plot(t, d_plot, color="#c0392b")
    axes[1].axhline(RiskConfig().brake_dist_m, ls="--", c="#e74c3c", label="BRAKE thresh")
    axes[1].axhline(RiskConfig().caution_dist_m, ls="--", c="#f39c12", label="CAUTION thresh")
    axes[1].set_ylabel("gap to obstacle (m)"); axes[1].set_title("Obstacle in path: gap from the front bumper")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    sp = [s * 3.6 if s else np.nan for s in speeds]
    axes[2].plot(t, sp, color="#2980b9")
    axes[2].set_ylabel("speed (km/h)"); axes[2].set_xlabel("time (s)")
    axes[2].set_title("Speed measured from the LiDAR (scan alignment)"); axes[2].grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pp.savefig(fig); plt.close(fig)
    pp.close()
    return score, counts, len(events)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=int, default=0,
                    help="which continuous scene to report on (0-based)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--path", choices=["heading", "model"], default="heading",
                    help="scan along the car's heading (default) or the learned path")
    args = ap.parse_args()

    from scene_dashboard import group_scenes
    from fleet_review import assess_scene
    from collision_risk import effective_level
    OUT.mkdir(exist_ok=True)
    all_frames = sorted(LIDAR_DIR.glob("*.bin"))
    if not all_frames:
        raise FileNotFoundError(f"No .bin frames in {LIDAR_DIR}")
    scenes = group_scenes(all_frames)
    start, end = scenes[args.scene]
    frames = all_frames[start:end:args.stride]
    scene_name = f"{frames[0].name.split('__')[0]}  (scene {args.scene} of {len(scenes)})"
    dt = KEYFRAME_DT * args.stride

    model = device = None
    if args.path == "model":
        model, device = load_planner()
    results = assess_scene(frames, RiskConfig(), model, device, dt)
    levels = [effective_level(r.level, sp) for _, _, r, sp in results]
    dists = [r.nearest_dist_m for _, _, r, _ in results]
    ttcs = [r.ttc_s for _, _, r, _ in results]
    speeds = [sp for _, _, _, sp in results]

    score, counts, n_ev = build_report(levels, dists, ttcs, speeds, dt,
                                        scene_name, len(frames), args.path)
    print(f"\nSafety score {score:.0f}/100 · {counts} · {n_ev} risk events")
    print(f"Saved {OUT / 'safety_report.pdf'}")


if __name__ == "__main__":
    main()
