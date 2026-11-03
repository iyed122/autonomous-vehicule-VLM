"""
demo/make_demo.py
=================
Rebuilds every demo asset from real LiDAR, after `python fleet_review.py` has written
inference_outputs/fleet_*.csv:

    assets/demo_risk_{clear,caution,brake}.gif   one real stretch of each risk level
    assets/risk_monitor_frame.png                a BRAKE frame
    assets/safety_report_p1.png, _p2.png         the safety report of the demo scene
    assets/fleet_overview.png                    the fleet review chart
    demo/demo.mp4, demo/demo_poster.jpg          the ~45 s walkthrough (H.264)

The video is written for someone deciding in a minute whether the work matters:
question, answer, one real example, the evidence, how it is built.

Usage (from the repo root):
    python demo/make_demo.py                 # picks the closest moving BRAKE event
    python demo/make_demo.py --scene 12
"""

import argparse
import csv
import dataclasses
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_risk import RiskConfig, obstacle_map, effective_level, LEVEL_COLORS  # noqa: E402
from fleet_review import assess_scene, counted_level, default_planner  # noqa: E402
from risk_monitor import render_frame  # noqa: E402
from safety_report import build_report  # noqa: E402

OUT = ROOT / "inference_outputs"
ASSETS = ROOT / "assets"
DEMO = ROOT / "demo"
LIDAR_DIR = ROOT / "LIDAR_TOP"
W, H, FPS = 1280, 720, 12
VIEW = (-12, 48, -22, 22)                      # metres: behind, ahead, right, left
FONT_DIR = Path("C:/Windows/Fonts")
INK, MUTED, ACCENT, BG = (28, 32, 38), (95, 103, 115), (41, 128, 185), (250, 250, 248)
CFG = RiskConfig()
PLANNER = None                                 # planner v4, loaded in main()


def font(size, bold=False):
    for name in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        if (FONT_DIR / name).exists():
            return ImageFont.truetype(str(FONT_DIR / name), size)
    return ImageFont.load_default()


def read_csv(name):
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if draw.textlength(trial, font=fnt) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line] if line else lines


def card(title, body=(), kicker=""):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 140
    if kicker:
        d.text((110, y), kicker.upper(), font=font(24, True), fill=ACCENT)
        y += 54
    for line in wrap(d, title, font(50, True), W - 220):
        d.text((110, y), line, font=font(50, True), fill=INK)
        y += 66
    y += 26
    for para in body:
        for line in wrap(d, para, font(29), W - 220):
            d.text((110, y), line, font=font(29), fill=MUTED)
            y += 42
        y += 14
    return np.array(img)


def numbers_card(kicker, title, figures, foot):
    """Big-number slide: [(value, label), ...]."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((110, 110), kicker.upper(), font=font(24, True), fill=ACCENT)
    d.text((110, 160), title, font=font(44, True), fill=INK)
    col = (W - 220) // len(figures)
    for i, (value, label) in enumerate(figures):
        x = 110 + i * col
        d.text((x, 290), value, font=font(78, True), fill=INK if i < len(figures) - 1 else ACCENT)
        for j, line in enumerate(wrap(d, label, font(26), col - 40)):
            d.text((x, 400 + 36 * j), line, font=font(26), fill=MUTED)
    for j, line in enumerate(wrap(d, foot, font(22), W - 220)):
        d.text((110, 600 + 30 * j), line, font=font(22), fill=MUTED)
    return np.array(img)


def picture_slide(title, picture, caption=""):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 30), title, font=font(36, True), fill=INK)
    pic = Image.open(picture).convert("RGB")
    pic.thumbnail((W - 120, H - 170))
    img.paste(pic, ((W - pic.width) // 2, 96))
    for j, line in enumerate(wrap(d, caption, font(22), W - 120)):
        d.text((60, H - 62 + 28 * j), line, font=font(22), fill=MUTED)
    return np.array(img)


def replay_frame(risk_img, levels, k, scene_label):
    img = Image.new("RGB", (W, H), BG)
    pic = Image.fromarray(risk_img)
    pic.thumbnail((H - 30, H - 30))
    img.paste(pic, (20, (H - pic.height) // 2))
    d = ImageDraw.Draw(img)
    x0 = 20 + pic.width + 36
    d.text((x0, 60), "How a moment gets flagged", font=font(32, True), fill=INK)
    for i, line in enumerate(wrap(d, scene_label, font(22), W - x0 - 40)):
        d.text((x0, 110 + 30 * i), line, font=font(22), fill=MUTED)
    d.text((x0, 200), "Risk over this drive", font=font(22, True), fill=INK)
    bar_w = W - x0 - 50
    step = bar_w / len(levels)
    for i, lvl in enumerate(levels):
        d.rectangle([x0 + i * step, 236, x0 + (i + 1) * step, 272], fill=LEVEL_COLORS[lvl])
    cx = x0 + (k + 0.5) * step
    d.polygon([(cx - 9, 286), (cx + 9, 286), (cx, 272)], fill=INK)
    notes = [
        "Top-down LiDAR view, the car at the star, driving to the right.",
        "Blue: where planner v4 predicts the car will be in the next 5.5 s. That lane is checked.",
        "Red: anything at least 0.5 m tall. Circle: the first obstacle in the lane.",
        "Under 5 m from the bumper while moving: BRAKE, a moment to review. Under 15 m: CAUTION.",
    ]
    y = 320
    for n in notes:
        for line in wrap(d, n, font(23), W - x0 - 40):
            d.text((x0, y), line, font=font(23), fill=MUTED)
            y += 32
        y += 12
    return np.array(img)


def write_mp4(frames, path):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    if not vw.isOpened():
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def save_gif(frames, path, ms=450):
    ims = [Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)


def render(item, idx, n):
    bev, path, risk, speed = item
    # show the level that counts (a close gap while stopped is CLEAR), as in the timeline
    risk = dataclasses.replace(risk, level=effective_level(risk.level, speed))
    return render_frame(bev, path, obstacle_map(bev, CFG), risk, idx, n,
                        path_label="Predicted path, planner v4" if PLANNER else "Lane checked (next 5.5 s)",
                        corridor_m=CFG.corridor_half_width_m, view=VIEW)


def scene_files(rows, scene):
    return [LIDAR_DIR / r["file"] for r in rows if int(r["scene"]) == scene]


def longest_run(rows, level, max_len=8):
    """Longest stretch of one counted level inside a single scene: (scene, a, b) row indices."""
    best, i = (None, 0, 0), 0
    while i < len(rows):
        if counted_level(rows[i]) != level:
            i += 1
            continue
        j = i
        while j < len(rows) and counted_level(rows[j]) == level and rows[j]["scene"] == rows[i]["scene"]:
            j += 1
        if j - i > best[2] - best[1]:
            best = (int(rows[i]["scene"]), i, j)
        i = j
    s, a, b = best
    return s, a, min(b, a + max_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int)
    args = ap.parse_args()

    global PLANNER
    PLANNER = default_planner()
    rows = read_csv("fleet_frames.csv")
    scenes = {int(s["scene"]): s for s in read_csv("fleet_scenes.csv")}
    events = read_csv("fleet_events.csv") if (OUT / "fleet_events.csv").exists() else []
    first_row = {}
    for i, r in enumerate(rows):
        first_row.setdefault(int(r["scene"]), i)

    # 1. one real stretch of each level -> README GIFs
    for lvl in ("CLEAR", "CAUTION", "BRAKE"):
        s, a, b = longest_run(rows, lvl)
        results = assess_scene(scene_files(rows, s), CFG, PLANNER)
        off = first_row[s]
        frames = [render(results[i - off], i - a, b - a) for i in range(a, b)]
        save_gif(frames, ASSETS / f"demo_risk_{lvl.lower()}.gif")
        if lvl == "BRAKE":
            Image.fromarray(frames[len(frames) // 2]).save(ASSETS / "risk_monitor_frame.png")
        print(f"{lvl}: scene {s}, rows {a}-{b}")

    # 2. the demo scene: the closest moving BRAKE event that starts at least 4 s into a
    #    scene with CLEAR and CAUTION too, so the viewer sees the situation build up
    if args.scene is None:
        varied = {k for k, s in scenes.items()
                  if int(s["clear_pct"]) > 0 and int(s["caution_pct"]) > 0}
        ranked = sorted(events, key=lambda e: float(e["min_dist_m"]))
        args.scene = next((int(e["scene"]) for e in ranked
                           if int(e["scene"]) in varied and float(e["start_s"]) >= 4.0),
                          int(ranked[0]["scene"]) if ranked else 0)
    results = assess_scene(scene_files(rows, args.scene), CFG, PLANNER)
    levels = [effective_level(r.level, sp) for _, _, r, sp in results]
    s = scenes[args.scene]
    label = (f"A real {len(results) * 0.5:.0f}-second drive in {s['city']} from the nuScenes "
             f"dataset. Safety score {s['score']}/100. Shown at 2x speed.")
    build_report(levels, [r.nearest_dist_m for _, _, r, _ in results],
                 [r.ttc_s for _, _, r, _ in results], [sp for _, _, _, sp in results], 0.5,
                 f"{rows[first_row[args.scene]]['log']}  (scene {args.scene})", len(results),
                 "learned" if PLANNER else "heading")
    import fitz
    pdf = fitz.open(OUT / "safety_report.pdf")
    for n in range(2):
        pdf[n].get_pixmap(dpi=110).save(ASSETS / f"safety_report_p{n + 1}.png")
    page = pdf[0]
    top = fitz.Rect(page.rect.width * 0.08, page.rect.height * 0.02,
                    page.rect.width * 0.92, page.rect.height * 0.62)
    page.get_pixmap(dpi=150, clip=top).save(OUT / "safety_report_top.png")
    shutil.copy(OUT / "fleet_overview.png", ASSETS / "fleet_overview.png")

    # 3. numbers for the slides
    total_min = len(rows) * 0.5 / 60
    review_min = sum(float(e["review_s"]) for e in events) / 60
    pct = 100 * review_min / total_min

    # 4. the walkthrough video
    video = []
    hold = lambda img, sec: video.extend([img] * int(sec * FPS))
    hold(card("Which moments of a drive should a safety manager actually watch?",
              ["Fleets and insurers record far more driving than anyone can review.",
               "This project scores every drive from its LiDAR data and keeps only the "
               "moments worth a human look."], "The business question"), 5)
    hold(numbers_card("The answer, on real data", "From hours of footage to a short review list",
                      [(f"{total_min:.0f} min", "of real driving, Boston and Singapore"),
                       (f"{len(events)}", "close calls while moving, ranked worst first"),
                       (f"{review_min:.1f} min", f"left to watch: {pct:.0f}% of the footage")],
                      "nuScenes autonomous-driving data. Every drive also gets a 0-100 safety "
                      "score and a one-page report."), 6)
    for k, item in enumerate(results):
        video.extend([replay_frame(render(item, k, len(results)), levels, k, label)] * (FPS // 4))
    hold(picture_slide("Each drive gets a one-page safety report", OUT / "safety_report_top.png",
                       "This drive's report: score, time at each risk level, risk events by closest gap."), 5)
    hold(picture_slide("Across the whole recording", ASSETS / "fleet_overview.png",
                       "The riskiest drives come first, so a reviewer starts where it matters."), 5)
    import torch
    from motion_planner import CKPT
    cv = torch.load(CKPT, map_location="cpu", weights_only=False)
    v4 = cv["cv"]["Planner v4 (this work)"]
    best_base = min((n for n in cv["cv"] if n not in ("Planner v4 (this work)", "Epoch-5 planner (one scan)")),
                    key=lambda n: cv["cv"][n][0])
    base = cv["cv"][best_base]
    multi = list(cv["multi"].values())
    hold(numbers_card("The model", "Trained at scale, tested on drives it never saw",
                      [("200 GB", "of nuScenes driving data, trained on AWS SageMaker"),
                       (f"{100 * (base[1] - v4[1]) / base[1]:.0f}%", "lower final error than the best physics baseline"),
                       (f"{100 * (multi[0] - multi[1]) / multi[0]:.0f}%",
                        "lower error for its 3 candidate paths than 3 physics guesses")],
                      "5-fold cross-validation split by drive, on 1,945 key-frames with labels recovered "
                      "from the LiDAR itself."), 6)
    hold(card("How it's built",
              ["A bird's-eye-view network (PyTorch, ~12 M parameters) trained on 200 GB of nuScenes on "
               "AWS SageMaker reads each LiDAR scan; planner v4 adds the car's motion, measured by "
               "registering consecutive scans, and predicts 3 candidate paths for the next 5.5 s.",
               "A NumPy risk layer checks the predicted lane; close calls while moving go into the "
               "review queue. A YOLOv8 camera detector runs alongside.",
               "A local language model writes a plain-language note for each flagged moment."],
              "Under the hood"), 8)
    hold(card("Iyed Mediouni", ["AI & software engineer, Tunisia, open to remote work.",
                                "github.com/iyed122/autonomous-vehicule-VLM",
                                "iyed122.github.io"], "Thanks for watching"), 4)
    write_mp4(video, DEMO / "demo.mp4")
    Image.fromarray(video[0]).save(DEMO / "demo_poster.jpg", quality=88)
    print(f"demo scene {args.scene}: {len(video) / FPS:.0f} s video -> {DEMO / 'demo.mp4'}")


if __name__ == "__main__":
    main()
