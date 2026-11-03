"""
fleet_review.py
===============
The business question this project answers:

    "We record hours of driving. Which moments should a safety manager actually
     watch, and how risky was each drive?"

Every LiDAR key-frame goes through the collision-risk layer. The car's motion (speed,
sideways drift, yaw rate) is measured from the LiDAR itself by registering each scan on
the previous one (odometry_labels.register). The path checked is planner v4's most likely
path for the next 5.5 s (motion_planner.py) when its cross-validated error beats the CTRV
baseline, otherwise the lane along the current heading (--path heading forces that).

The recording is split into continuous scenes (drives); each gets the same 0-100 safety
score as safety_report.py. Every stretch where an obstacle sat in the path less than the
BRAKE gap ahead of the bumper *while the car was moving* becomes a review event: gaps
while stopped or crawling under 7 km/h (lights, queues) are not counted.

Outputs (inference_outputs/):
    fleet_frames.csv     one row per key-frame: scene, time, level, distance, TTC, speed
    fleet_scenes.csv     one row per scene, worst safety score first
    fleet_events.csv     every BRAKE event, closest approach first
    fleet_review.md      the summary a fleet-safety manager reads
    fleet_overview.png   chart: scenes ranked by score, and time at each risk level

Usage:
    python fleet_review.py                 # all frames in LIDAR_TOP/
    python fleet_review.py --path heading  # scan along the current heading instead
    python fleet_review.py --from-csv      # rebuild the summary from fleet_frames.csv
    python fleet_review.py --from-csv --notes qwen3:8b
                                           # + a plain-language review note per top event,
                                           #   written by a local Ollama model
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collision_risk import (RiskConfig, assess_risk, heading_corridor, extend_path,
                            effective_level, MOVING_MPS, LEVELS, LEVEL_COLORS)
from data_preparation_v2 import load_lidar_points, create_bev_representation
from safety_report import find_events, safety_score
from scene_dashboard import group_scenes, parse_timestamp

ROOT = Path(__file__).resolve().parent
LIDAR_DIR = ROOT / "LIDAR_TOP"
OUT = ROOT / "inference_outputs"
DT = 0.5            # nuScenes key-frames are 2 Hz
LEAD_IN_S = 2.0     # footage a reviewer watches before each event starts


def city(log_name):
    """nuScenes logs carry the local UTC offset: -0400 is Boston, +0800 Singapore."""
    return "Boston" if "-0400" in log_name else ("Singapore" if "+0800" in log_name else "?")


def default_planner(path_mode="auto"):
    """Planner v4 if it exists and (in auto mode) beat the CTRV baseline in cross-validation."""
    from motion_planner import CKPT, MotionPlanner
    if path_mode == "heading" or not CKPT.exists():
        return None
    planner = MotionPlanner()
    return planner if (path_mode == "learned" or planner.beats_baseline) else None


def assess_scene(files, cfg=RiskConfig(), planner=None, dt=DT):
    """Risk for every key-frame of one continuous scene.

    Motion: the scan registered on the previous one (the next pair for the first frame),
    kept only when the registration is clear. Path: planner v4's most likely path,
    extended to the CAUTION range, or the lane along the current heading.
    Returns [(bev, path, RiskResult, speed_mps), ...]."""
    from odometry_labels import ego_points, register
    from motion_planner import median_steps, motion_features
    pts = [load_lidar_points(str(f)) for f in files]
    bevs = [create_bev_representation(p) for p in pts]
    xy = [ego_points(p) for p in pts]
    steps = [register(xy[j], xy[j + 1]) for j in range(len(xy) - 1)]
    past = median_steps(steps, centered=False) if steps else []   # same filtering as training
    out = []
    for j, bev in enumerate(bevs):
        i = j - 1 if j > 0 else 0                      # the step that ends at this scan
        motion = speed = None
        if steps and steps[i][3] >= 0.15:
            speed = float(np.hypot(past[i][0], past[i][1])) / dt
            motion = motion_features(past, j)        # needs two past steps
        if planner is not None and motion is not None:
            path = extend_path(planner.predict(bev, motion)[0], cfg)
        else:
            path = heading_corridor(speed, cfg)
        out.append((bev, path, assess_risk(path, bev, cfg, speed_mps=speed), speed))
    return out


def run_frames(frames, path_mode):
    planner = default_planner(path_mode)
    print("path checked:", "planner v4" if planner else "current heading")
    cfg, rows = RiskConfig(), []
    for k, (start, end) in enumerate(group_scenes(frames)):
        t0 = parse_timestamp(frames[start])
        results = assess_scene(frames[start:end], cfg, planner)
        for f, (_, _, r, speed) in zip(frames[start:end], results):
            rows.append({
                "scene": k, "log": f.name.split("__")[0], "file": f.name,
                "t_s": round((parse_timestamp(f) - t0) / 1e6, 2), "level": r.level,
                "dist_m": round(r.nearest_dist_m, 2) if np.isfinite(r.nearest_dist_m) else "",
                "ttc_s": round(r.ttc_s, 2) if r.ttc_s else "",
                "speed_kmh": round(speed * 3.6, 1) if speed is not None else "",
            })
        print(f"  scene {k}: {end - start} frames")
    return rows


def counted_level(row):
    """The level used for scores and events (see collision_risk.effective_level)."""
    speed = float(row["speed_kmh"]) / 3.6 if row["speed_kmh"] != "" else None
    return effective_level(row["level"], speed)


def summarise(rows):
    scenes, events = [], []
    for k in sorted({r["scene"] for r in rows}):
        sr = [r for r in rows if r["scene"] == k]
        levels = [counted_level(r) for r in sr]
        dists = [float(r["dist_m"]) if r["dist_m"] != "" else np.inf for r in sr]
        ttcs = [float(r["ttc_s"]) if r["ttc_s"] != "" else None for r in sr]
        pct = {lvl: 100.0 * levels.count(lvl) / len(levels) for lvl in LEVELS}
        ev = find_events(levels, dists, ttcs, DT)
        brake = [e for e in ev if e["peak"] == "BRAKE"]
        for e in brake:
            sp = [float(r["speed_kmh"]) for r in sr[e["start"]:e["end"] + 1] if r["speed_kmh"] != ""]
            events.append({
                "scene": k, "city": city(sr[0]["log"]),
                "start_s": float(sr[e["start"]]["t_s"]), "duration_s": e["duration_s"],
                "min_dist_m": round(e["min_dist"], 1),
                "min_ttc_s": round(e["min_ttc"], 1) if e["min_ttc"] else "",
                "max_speed_kmh": round(max(sp), 1) if sp else "",
                "review_s": e["duration_s"] + min(LEAD_IN_S, float(sr[e["start"]]["t_s"])),
            })
        scenes.append({
            "scene": k, "city": city(sr[0]["log"]), "log": sr[0]["log"],
            "duration_s": len(sr) * DT,
            "clear_pct": round(pct["CLEAR"]), "caution_pct": round(pct["CAUTION"]),
            "brake_pct": round(pct["BRAKE"]), "brake_events": len(brake),
            "score": round(safety_score(pct, len(brake))),
        })
    scenes.sort(key=lambda s: s["score"])
    events.sort(key=lambda e: e["min_dist_m"])
    return scenes, events


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def overview_chart(scenes, rows, path):
    total = {lvl: sum(counted_level(r) == lvl for r in rows) * DT / 60 for lvl in LEVELS}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 2]})
    worst = scenes[:15]
    colors = [LEVEL_COLORS["BRAKE"] if s["score"] < 60 else
              LEVEL_COLORS["CAUTION"] if s["score"] < 85 else LEVEL_COLORS["CLEAR"] for s in worst]
    a1.barh([f"scene {s['scene']} · {s['city']}" for s in worst], [s["score"] for s in worst],
            color=colors)
    a1.invert_yaxis(); a1.set_xlim(0, 100); a1.set_xlabel("safety score (0-100)")
    a1.set_title(f"The 15 riskiest of {len(scenes)} scenes", loc="left", fontweight="bold")
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    a2.bar(list(total), list(total.values()), color=[LEVEL_COLORS[l] for l in total])
    for i, v in enumerate(total.values()):
        a2.text(i, v, f"{v:.1f} min", ha="center", va="bottom")
    a2.set_ylabel("minutes of driving"); a2.set_title("Time at each risk level", loc="left",
                                                      fontweight="bold")
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


NOTE_PROMPT = (
    "You write review notes for a fleet-safety manager. A collision-risk system flagged "
    "one moment in a recorded drive. Use ONLY the measurements below; you cannot see the "
    "video, so do not name or guess any object, person or road feature.\n\n{facts}\n\n"
    'Respond with ONLY this JSON: {{"note": "<one sentence: why this moment was flagged, '
    'in plain words>", "check": "<one sentence: what to look for when watching the footage>"}}'
)


def add_notes(events, rows, model, top=5):
    """Ask a local language model (Ollama) for a plain-language note per top event."""
    from vlm_reasoner import OllamaVLM, _parse
    backend = OllamaVLM(model=model)
    for e in events[:top]:
        sr = [r for r in rows if r["scene"] == e["scene"]]
        speeds = [float(r["speed_kmh"]) for r in sr
                  if r["speed_kmh"] != "" and e["start_s"] <= float(r["t_s"]) < e["start_s"] + e["duration_s"]]
        facts = "\n".join([
            f"- city: {e['city']}",
            f"- the moment starts {e['start_s']:.1f} s into a {len(sr) * DT:.0f} s clip "
            f"and lasts {e['duration_s']:.1f} s",
            f"- gap between the front bumper and the closest obstacle inside the predicted path: {e['min_dist_m']} m "
            f"(the BRAKE threshold is {RiskConfig().brake_dist_m:.0f} m from the front bumper)",
            f"- shortest time-to-obstacle: {e['min_ttc_s']} s" if e["min_ttc_s"] != "" else
            "- time-to-obstacle: not available",
            f"- estimated speed during the moment: {min(speeds):.0f}-{max(speeds):.0f} km/h"
            if speeds else "- estimated speed: not available",
        ])
        d = _parse(backend.generate(NOTE_PROMPT.format(facts=facts)))
        e["note"] = str(d.get("note", "")).strip()
        e["check"] = str(d.get("check", "")).strip()
    for e in events[top:]:
        e["note"] = e["check"] = ""
    return model


def summary_md(scenes, events, rows, path, notes_model=None):
    total_min = len(rows) * DT / 60
    review_min = sum(e["review_s"] for e in events) / 60
    brake_scenes = sum(1 for s in scenes if s["brake_events"])
    lines = [
        "# Fleet review",
        "",
        f"- Driving analysed: **{total_min:.0f} minutes** ({len(rows)} LiDAR key-frames, "
        f"{len(scenes)} scenes, {' and '.join(sorted({s['city'] for s in scenes}))}).",
        f"- Moments to review: **{len(events)} BRAKE events** in {brake_scenes} scenes "
        f"(an obstacle in the car's path, less than {RiskConfig().brake_dist_m:.0f} m ahead of the front bumper, "
        f"while moving above {MOVING_MPS * 3.6:.0f} km/h; speed measured from the LiDAR).",
        f"- Footage to watch, with {LEAD_IN_S:.0f} s before each event: **{review_min:.1f} minutes**, "
        f"{100 * review_min / total_min:.0f}% of the recording.",
        "",
        "## Worst scenes",
        "",
        "| scene | city | length | safety score | BRAKE events | time in BRAKE |",
        "|---|---|---|---|---|---|",
    ]
    for s in scenes[:5]:
        lines.append(f"| {s['scene']} | {s['city']} | {s['duration_s']:.0f} s | {s['score']}/100 | "
                     f"{s['brake_events']} | {s['brake_pct']}% |")
    lines += ["", "## Closest approaches", "",
              "| scene | starts at | lasts | smallest gap to an obstacle in path |", "|---|---|---|---|"]
    for e in events[:10]:
        lines.append(f"| {e['scene']} | {e['start_s']:.1f} s | {e['duration_s']:.1f} s | "
                     f"{e['min_dist_m']} m |")
    if notes_model:
        lines += ["", f"## Review notes (written by {notes_model} from the numbers above, "
                  "no video access)", ""]
        for e in events[:5]:
            lines.append(f"- **Scene {e['scene']} at {e['start_s']:.1f} s:** {e['note']} "
                         f"*Check:* {e['check']}")
    lines += ["", "Thresholds are fixed distances, not tuned on crash or near-miss labels "
              "(nuScenes has none), and the lane checked is planner v4's prediction from one scan plus "
              "the measured motion, so it can be wrong in a sharp turn. Treat the ranking as a review queue, not a verdict."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", choices=["auto", "learned", "heading"], default="auto")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--notes", metavar="OLLAMA_MODEL",
                    help="write review notes for the top events with this local model")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.from_csv:
        with open(OUT / "fleet_frames.csv", encoding="utf-8") as f:
            rows = [dict(r, scene=int(r["scene"])) for r in csv.DictReader(f)]
    else:
        frames = sorted(LIDAR_DIR.glob("*.bin"))
        if not frames:
            raise FileNotFoundError(f"No .bin frames in {LIDAR_DIR}")
        rows = run_frames(frames, args.path)
        write_csv(OUT / "fleet_frames.csv", rows)

    scenes, events = summarise(rows)
    notes_model = add_notes(events, rows, args.notes) if args.notes and events else None
    write_csv(OUT / "fleet_scenes.csv", scenes)
    if events:
        write_csv(OUT / "fleet_events.csv", events)
    overview_chart(scenes, rows, OUT / "fleet_overview.png")
    summary_md(scenes, events, rows, OUT / "fleet_review.md", notes_model)
    print((OUT / "fleet_review.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
