"""
vlm_analyze.py
==============
End-to-end VLM reasoning over one driving moment: take a LiDAR frame, assemble the
6-camera surround view + the LiDAR BEV (with planned path & collision risk), send them to
a local Vision-Language model, and render a "reasoning card" — the geometric decision next
to the VLM's scene caption, open-vocabulary hazards, agreement, and recommended action.

Examples
--------
    # offline smoke test (no GPU / no Ollama) — uses bundled images + mock VLM
    python vlm_analyze.py --backend mock --straight

    # real run: your nuScenes camera folder(s) + a local vision model via Ollama
    #   ollama pull qwen2.5vl:7b   (qwen3:8b is TEXT-ONLY and will not see the images)
    python vlm_analyze.py --cam-dir /path/to/nuscenes/samples \
                          --frame 552 --backend ollama --model qwen2.5vl:7b

Output: inference_outputs/vlm_analysis.png  (+ .json)
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scene_dashboard as sd
import vlm_reasoner as vlm
from collision_risk import RiskConfig, assess_risk, straight_fallback_path, LEVEL_COLORS
from data_preparation_v2 import load_lidar_points, create_bev_representation

ROOT = Path(__file__).resolve().parent
LIDAR_DIR = ROOT / "LIDAR_TOP"
OUT = ROOT / "inference_outputs"
KEYFRAME_DT = 0.5


def predict_path(bev, straight):
    if straight:
        return straight_fallback_path(30, 20)
    import torch
    from trajectory_model import TrajectoryPlannerModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryPlannerModel().to(device)
    ck = torch.load(str(ROOT / "checkpoint_v3" / "best_trajectory_model.pth"), map_location=device)
    model.load_state_dict(ck.get("model_state_dict", ck)); model.eval()
    t = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        best, _ = model.predict_best_trajectory(t)
    return best[0].cpu().numpy()


def render_card(surround_rgb, risk, analysis, out_path):
    """Compose the surround montage (top) with a VLM reasoning panel (bottom)."""
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.06)

    ax_img = fig.add_subplot(gs[0]); ax_img.imshow(surround_rgb); ax_img.axis("off")
    ax_img.set_title("360° surround + LiDAR BEV  →  VLM reasoning", fontsize=13)

    ax = fig.add_subplot(gs[1]); ax.axis("off")
    color = LEVEL_COLORS.get(risk.level, "#555")
    ax.text(0.0, 1.0, f"GEOMETRIC DECISION: {risk.level}", transform=ax.transAxes,
            fontsize=14, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"))
    dist = "clear" if not np.isfinite(risk.nearest_dist_m) else f"{risk.nearest_dist_m:.1f} m"
    ttc = f"{risk.ttc_s:.1f} s" if risk.ttc_s else "—"
    ax.text(0.42, 1.0, f"nearest {dist} · TTC {ttc}", transform=ax.transAxes,
            fontsize=10, va="top", color="#333")

    agree = analysis.agrees_with_decision
    agree_txt = ("✓ VLM corroborates" if agree else
                 "✗ VLM disagrees" if agree is False else "VLM: —")
    agree_col = "#2ecc71" if agree else "#e74c3c" if agree is False else "#777"

    blocks = [
        ("VLM scene", analysis.caption or "—"),
        ("Open-vocabulary hazards", "• " + "\n• ".join(analysis.hazards) if analysis.hazards else "—"),
        (agree_txt, analysis.rationale or "—"),
        ("Recommended action", analysis.recommended_action or "—"),
    ]
    y = 0.80
    for title, body in blocks:
        tc = agree_col if title.startswith(("✓", "✗", "VLM:")) else "#1a5276"
        ax.text(0.0, y, title, transform=ax.transAxes, fontsize=11, fontweight="bold",
                color=tc, va="top")
        y -= 0.085
        ax.text(0.02, y, body, transform=ax.transAxes, fontsize=10, va="top",
                color="#222", wrap=True)
        y -= 0.085 * (1 + body.count("\n"))
    ax.text(1.0, 1.0, f"model: {analysis.model}"
            + ("  (saw images)" if analysis.used_images else "  (text-only)"),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#999")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=552, help="LiDAR frame index")
    ap.add_argument("--cam-dir", default="", help="folder with nuScenes camera frames "
                    "(all 6 channels). Blank = bundled illustrative images.")
    ap.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    ap.add_argument("--model", default=vlm.DEFAULT_MODEL)
    ap.add_argument("--host", default=vlm.OLLAMA_DEFAULT_HOST)
    ap.add_argument("--straight", action="store_true",
                    help="use constant-velocity path (no torch needed)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    lidar = sorted(LIDAR_DIR.glob("*.bin"))
    lf = str(lidar[min(args.frame, len(lidar) - 1)])

    bev = create_bev_representation(load_lidar_points(lf))
    path = predict_path(bev, args.straight)
    risk = assess_risk(path, bev, RiskConfig(), horizon_s=20 * KEYFRAME_DT)
    bev_render = sd.render_dashboard_frame(bev, path, risk, 0, 1, camera_rgb=None)

    # --- assemble the 6-camera surround ---
    if args.cam_dir and Path(args.cam_dir).is_dir():
        cams = [str(p) for p in Path(args.cam_dir).rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        cam_paths = sd.match_surround(lf, cams)
        matched = sum(v is not None for v in cam_paths.values())
        print(f"Matched {matched}/6 cameras to the LiDAR frame by timestamp.")
    else:
        bundled = sorted(glob.glob(str(ROOT / "yolo_dataset" / "images" / "train" / "*.jpg")))[:6]
        cam_paths = {ch: (bundled[i] if i < len(bundled) else None)
                     for i, ch in enumerate(sd.NUSCENES_CAMERAS)}
        print("Using bundled illustrative images (point --cam-dir at your nuScenes "
              "camera folder for a real, time-synced surround view).")

    surround = sd.assemble_surround(cam_paths, bev_rgb=bev_render)

    # --- build context + run the VLM ---
    ctx = {
        "risk_level": risk.level,
        "nearest_obstacle_m": ("clear" if not np.isfinite(risk.nearest_dist_m)
                               else f"{risk.nearest_dist_m:.1f} m"),
        "ttc_s": f"{risk.ttc_s:.1f} s" if risk.ttc_s else None,
        "ego_speed_kmh": round(risk.speed_mps * 3.6) if risk.speed_mps else None,
        "notes": "BEV is the LiDAR top-down; obstacles are dense+elevated cells in path.",
    }
    labels = [c for c in sd.NUSCENES_CAMERAS] + ["BEV_LiDAR"]
    backend = vlm.make_backend(args.backend, model=args.model, host=args.host) \
        if args.backend == "ollama" else vlm.make_backend("mock")

    print(f"Querying VLM backend='{args.backend}' model='{getattr(backend,'model','?')}' …")
    try:
        analysis = vlm.analyze(backend, ctx, images=[sd.image_to_png_bytes(surround)],
                               image_labels=labels)
    except Exception as e:
        print(f"VLM call failed ({e}). Is `ollama serve` running and the model pulled? "
              f"Falling back to mock.")
        analysis = vlm.analyze(vlm.make_backend("mock"), ctx,
                               images=[b"x"], image_labels=labels)

    out_png = OUT / "vlm_analysis.png"
    out_json = OUT / "vlm_analysis.json"
    render_card(surround, risk, analysis, out_png)
    out_json.write_text(json.dumps({"context": ctx, "analysis": analysis.to_dict()}, indent=2))
    print(f"\nVLM: {analysis.caption}")
    print(f"Hazards: {', '.join(analysis.hazards) or '—'}")
    print(f"Action: {analysis.recommended_action}  |  agrees={analysis.agrees_with_decision}")
    print(f"Saved {out_png} and {out_json}")


if __name__ == "__main__":
    main()
