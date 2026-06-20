"""
scene_narrator.py
=================
The narrated-scene demo: pick a few driving scenes and have a local Vision-Language model
*narrate each scene as it unfolds*, reasoning over the full 6-camera surround view AND the
LiDAR bird's-eye-view across the scene's frames.

For each selected scene it:
  1. splits the LiDAR stream into scenes by timestamp and selects 4-5 with variety,
  2. for every frame builds the 6-camera surround montage + the LiDAR BEV (depth/obstacles),
  3. asks the VLM for a short live "co-pilot" comment per segment of the scene,
  4. asks the VLM for a holistic scene summary over several keyframes (multi-image),
  5. renders a narrated clip (GIF + MP4) with the commentary as a subtitle bar, plus a
     text transcript — and an index page across all scenes.

Cameras and LiDAR are paired by timestamp (see scene_dashboard.match_surround); point
`--cam-dir` at your nuScenes camera folder for a real surround view. With no camera folder
it falls back to bundled illustrative frames. With `--backend mock` the whole thing runs
with no GPU and no Ollama server (used to verify the pipeline).

Examples
--------
    # offline pipeline check (no GPU / no server)
    python scene_narrator.py --backend mock --scenes 2 --straight

    # real demo: your nuScenes cameras + a local vision model
    ollama pull qwen2.5vl:7b
    python scene_narrator.py --cam-dir /path/to/nuscenes/samples \
                             --scenes 5 --model qwen2.5vl:7b
"""

import argparse
import glob
import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

import scene_dashboard as sd
import vlm_reasoner as vlm
from collision_risk import RiskConfig, assess_risk, straight_fallback_path

ROOT = Path(__file__).resolve().parent
LIDAR_DIR = ROOT / "LIDAR_TOP"
OUT = ROOT / "demo" / "scenes"
KEYFRAME_DT = 0.5


# --------------------------------------------------------------------------- #
# Fast BEV (vectorised; no per-point Python loop) — height + density channels.
# --------------------------------------------------------------------------- #
def fast_bev(points, res=0.2, xr=(-50, 50), yr=(-50, 50), zr=(-3, 3)):
    g = int((xr[1] - xr[0]) / res)
    m = ((points[:, 0] >= xr[0]) & (points[:, 0] < xr[1]) &
         (points[:, 1] >= yr[0]) & (points[:, 1] < yr[1]) &
         (points[:, 2] >= zr[0]) & (points[:, 2] < zr[1]))
    p = points[m]
    bev = np.zeros((g, g, 3), np.float32)
    if len(p):
        xi = np.clip(((p[:, 0] - xr[0]) / res).astype(int), 0, g - 1)
        yi = np.clip(((p[:, 1] - yr[0]) / res).astype(int), 0, g - 1)
        np.maximum.at(bev[:, :, 0], (xi, yi), (p[:, 2] - zr[0]) / (zr[1] - zr[0]))
        d = np.zeros((g, g), np.float32)
        np.add.at(d, (xi, yi), 1)
        bev[:, :, 2] = np.clip(d / 10, 0, 1)
    return bev


def load_lidar(f):
    return np.fromfile(f, dtype=np.float32).reshape(-1, 5)[:, :4]


# --------------------------------------------------------------------------- #
# Scene selection
# --------------------------------------------------------------------------- #
def select_demo_scenes(lidar_files, n=5, min_len=8, sample=4):
    """Group LiDAR into scenes by timestamp, score each by its dominant risk, and pick
    `n` scenes that are both long enough and varied (mix of risk profiles)."""
    cfg = RiskConfig()
    path = straight_fallback_path(30, 20)
    scenes = sd.group_scenes([str(p) for p in lidar_files])
    scored = []
    for si, (s, e) in enumerate(scenes):
        if e - s < min_len:
            continue
        idxs = np.linspace(s, e - 1, min(sample, e - s)).astype(int)
        levels = []
        for i in idxs:
            bev = fast_bev(load_lidar(str(lidar_files[i])))
            levels.append(assess_risk(path, bev, cfg, horizon_s=10.0).level)
        from collections import Counter
        dom = Counter(levels).most_common(1)[0][0]
        scored.append({"idx": si, "start": s, "end": e, "len": e - s, "dominant": dom})
    # pick variety: take the longest scene of each risk type first, then fill by length
    picked, seen = [], set()
    for lvl in ["CLEAR", "CAUTION", "BRAKE"]:
        cands = [c for c in scored if c["dominant"] == lvl]
        if cands:
            best = max(cands, key=lambda c: c["len"])
            picked.append(best); seen.add(best["idx"])
    for c in sorted(scored, key=lambda c: -c["len"]):
        if len(picked) >= n:
            break
        if c["idx"] not in seen:
            picked.append(c); seen.add(c["idx"])
    return picked[:n]


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def overlay_caption(rgb, header, comment):
    """Append a subtitle strip (header + wrapped comment) BELOW the frame so it never
    covers the camera/BEV content."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb).convert("RGB")
    W, H = im.size
    strip_h = 110  # fixed so all GIF/MP4 frames share one size
    lines = textwrap.wrap(comment, width=max(50, W // 13))[:3]  # cap at 3 lines
    canvas = Image.new("RGB", (W, H + strip_h), (15, 15, 20))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((16, H + 8), header, fill=(0, 220, 180))
    d.multiline_text((16, H + 30), "\n".join(lines), fill=(245, 245, 245), spacing=4)
    return np.array(canvas)


def build_frame_composite(lidar_path, cam_paths, cfg):
    """Surround montage (6 cams) + BEV tile for one frame; returns (composite_rgb, risk)."""
    bev = fast_bev(load_lidar(lidar_path))
    path = straight_fallback_path(30, 20)
    risk = assess_risk(path, bev, cfg, horizon_s=20 * KEYFRAME_DT)
    bev_render = sd.render_dashboard_frame(bev, path, risk, 0, 1, camera_rgb=None)
    composite = sd.assemble_surround(cam_paths, bev_rgb=bev_render)
    return composite, risk


def resolve_cameras_for_frame(lidar_path, cam_files, bundled, frame_i):
    """Return {channel: image path} for a frame: by timestamp if a real camera folder is
    given, else bundled illustrative images cycled across the six channels."""
    if cam_files:
        return sd.match_surround(lidar_path, cam_files)
    return {ch: (bundled[(frame_i + k) % len(bundled)] if bundled else None)
            for k, ch in enumerate(sd.NUSCENES_CAMERAS)}


# --------------------------------------------------------------------------- #
# Narrate one scene
# --------------------------------------------------------------------------- #
def narrate_scene(scene, lidar_files, cam_files, bundled, backend,
                  n_segments=5, max_frames=16):
    cfg = RiskConfig()
    s, e = scene["start"], scene["end"]
    idxs = np.linspace(s, e - 1, min(max_frames, e - s)).astype(int)

    composites, risks = [], []
    for fi, i in enumerate(idxs):
        cams = resolve_cameras_for_frame(str(lidar_files[i]), cam_files, bundled, fi)
        comp, risk = build_frame_composite(str(lidar_files[i]), cams, cfg)
        composites.append(comp); risks.append(risk)

    # split frames into segments; one live comment per segment
    seg_bounds = np.linspace(0, len(idxs), n_segments + 1).astype(int)
    seg_for_frame = np.zeros(len(idxs), int)
    comments = []
    for sgi in range(n_segments):
        a, b = seg_bounds[sgi], seg_bounds[sgi + 1]
        if b <= a:
            comments.append("")
            continue
        seg_for_frame[a:b] = sgi
        mid = (a + b) // 2
        ctx = {"risk_level": risks[mid].level,
               "nearest_obstacle_m": ("clear" if not np.isfinite(risks[mid].nearest_dist_m)
                                      else f"{risks[mid].nearest_dist_m:.1f} m"),
               "ttc_s": f"{risks[mid].ttc_s:.1f} s" if risks[mid].ttc_s else None}
        png = sd.image_to_png_bytes(composites[mid])
        comments.append(vlm.comment_segment(backend, ctx, images=[png]))

    # holistic scene summary over keyframe composites (multi-image)
    key = seg_bounds[:-1] + np.diff(seg_bounds) // 2
    key = [k for k in key if k < len(composites)]
    summary_ctx = {"risk_profile": ", ".join(r.level for r in risks)}
    summary = vlm.summarize_scene(backend, summary_ctx,
                                  images=[sd.image_to_png_bytes(composites[k]) for k in key])

    # render frames with subtitle bar
    frames = []
    for fi in range(len(idxs)):
        header = (f"{summary.get('scene_type','scene') or 'scene'}  ·  "
                  f"risk {risks[fi].level}  ·  frame {fi+1}/{len(idxs)}")
        frames.append(overlay_caption(composites[fi], header, comments[seg_for_frame[fi]]))

    transcript = {
        "scene_index": scene["idx"], "frames": int(len(idxs)),
        "dominant_risk": scene["dominant"],
        "scene_type": summary.get("scene_type", ""),
        "summary": summary.get("summary", ""),
        "hazards": summary.get("hazards", []),
        "commentary": [{"segment": k + 1, "text": c} for k, c in enumerate(comments) if c],
    }
    return frames, transcript


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--cam-dir", default="", help="nuScenes camera folder (all 6 channels)")
    ap.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    ap.add_argument("--model", default=vlm.DEFAULT_MODEL)
    ap.add_argument("--host", default=vlm.OLLAMA_DEFAULT_HOST)
    ap.add_argument("--segments", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--straight", action="store_true", help="(default path mode)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lidar = sorted(LIDAR_DIR.glob("*.bin"))
    if not lidar:
        raise FileNotFoundError(f"No LiDAR frames in {LIDAR_DIR}")

    cam_files = []
    if args.cam_dir and Path(args.cam_dir).is_dir():
        cam_files = [str(p) for p in Path(args.cam_dir).rglob("*")
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(f"Camera folder: {len(cam_files)} images (paired to LiDAR by timestamp).")
    bundled = sorted(glob.glob(str(ROOT / "yolo_dataset" / "images" / "train" / "*.jpg")))[:6]
    if not cam_files:
        print("No --cam-dir: using bundled illustrative camera frames.")

    if args.backend == "mock":
        backend = vlm.MockNarrator()
    else:
        backend = vlm.make_backend("ollama", model=args.model, host=args.host)
    print(f"VLM backend={args.backend} model={getattr(backend,'model','?')}")

    chosen = select_demo_scenes(lidar, n=args.scenes)
    print(f"Selected {len(chosen)} scenes: "
          + ", ".join(f"#{c['idx']+1}({c['dominant']},{c['len']}f)" for c in chosen))

    index = []
    for rank, scene in enumerate(chosen, 1):
        print(f"\n[{rank}/{len(chosen)}] narrating scene #{scene['idx']+1} "
              f"({scene['len']} frames, dominant {scene['dominant']}) …")
        try:
            frames, transcript = narrate_scene(
                scene, lidar, cam_files, bundled, backend,
                n_segments=args.segments, max_frames=args.max_frames)
        except Exception as ex:
            print(f"  scene failed ({ex}); falling back to mock narration.")
            frames, transcript = narrate_scene(
                scene, lidar, cam_files, bundled, vlm.MockNarrator(),
                n_segments=args.segments, max_frames=args.max_frames)

        stem = f"scene_{rank:02d}_{scene['dominant'].lower()}"
        gif = sd.frames_to_gif_bytes(frames, fps=args.fps, scale=0.5)
        (OUT / f"{stem}.gif").write_bytes(gif)
        try:
            sd.frames_to_mp4(frames, OUT / f"{stem}.mp4", fps=args.fps)
        except Exception as ex:
            print(f"  mp4 skipped ({ex})")
        (OUT / f"{stem}.json").write_text(json.dumps(transcript, indent=2))
        print(f"  scene: {transcript['scene_type']} — {transcript['summary'][:80]}…")
        index.append((stem, transcript))

    # index transcript across all scenes
    lines = ["# Narrated scene demo\n",
             "Generated by `scene_narrator.py` — a local VLM narrating each scene over the "
             "6-camera surround view + LiDAR BEV.\n"]
    for stem, t in index:
        lines.append(f"\n## {stem}  ({t['dominant_risk']} · {t['scene_type']})\n")
        lines.append(f"![{stem}]({stem}.gif)\n")
        lines.append(f"**Summary.** {t['summary']}\n")
        if t["hazards"]:
            lines.append(f"**Hazards:** {', '.join(t['hazards'])}\n")
        lines.append("**Live commentary:**\n")
        for c in t["commentary"]:
            lines.append(f"- {c['text']}")
    (OUT / "README.md").write_text("\n".join(lines))
    print(f"\nDone. {len(index)} scenes → {OUT}")


if __name__ == "__main__":
    main()
