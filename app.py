"""
app.py — Autonomous Perception & Planning demo (YOLO + BEV)
===========================================================
Interactive Streamlit demo that actually RUNS both trained models:

  * YOLO  : object detection on a camera image (nuScenes-style frame)
  * BEV   : multi-modal ego-trajectory prediction from a LiDAR .pcd.bin frame

Run:
    streamlit run app.py

All paths are resolved relative to this file, so the app is portable.
"""

import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = ROOT / "runs" / "train" / "nuscenes_yolo" / "weights" / "best.pt"
BEV_WEIGHTS = ROOT / "checkpoint_v3" / "best_trajectory_model.pth"
LIDAR_DIR = ROOT / "LIDAR_TOP"

st.set_page_config(layout="wide", page_title="Autonomous Perception (YOLO + BEV)")
st.title("Autonomous Perception & Planning — YOLO + BEV (nuScenes)")
st.caption("Real-time obstacle detection and multi-modal ego-trajectory prediction. "
           "Trained on the nuScenes autonomous-driving dataset.")

task = st.sidebar.radio("Task", ["YOLO object detection",
                                 "BEV trajectory prediction",
                                 "Scene dashboard (multi-frame → GIF)",
                                 "VLM scene reasoning (Ollama)"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models**\n\n"
    "- YOLOv8n detector\n- BEV trajectory planner\n- Collision-risk fusion\n\n"
    "Weights load from this repo; no external data needed."
)


# --------------------------------------------------------------------------- #
# Cached model loaders
# --------------------------------------------------------------------------- #
# COCO class ids for road objects -> gives real NAMES with zero retraining.
COCO_ROAD_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck


@st.cache_resource
def load_yolo(which="finetuned"):
    from ultralytics import YOLO
    if which == "named":
        # Stock COCO-pretrained YOLOv8n: detects car/truck/bus/person/... by name.
        # Auto-downloads (~6 MB) on first use; no training needed.
        return YOLO("yolov8n.pt")
    return YOLO(str(YOLO_WEIGHTS))


@st.cache_resource
def load_bev():
    import torch
    from trajectory_model import TrajectoryPlannerModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryPlannerModel().to(device)
    ckpt = torch.load(str(BEV_WEIGHTS), map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, device, ckpt


# --------------------------------------------------------------------------- #
# YOLO
# --------------------------------------------------------------------------- #
if task == "YOLO object detection":
    st.header("YOLO object detection")
    if not YOLO_WEIGHTS.exists():
        st.error(f"Weights not found: {YOLO_WEIGHTS}")
        st.stop()

    detector = st.radio(
        "Detector",
        ["Named objects (COCO pretrained)", "Fine-tuned nuScenes (single class)"],
        help="Your fine-tuned model was trained on one class ('object'), so it "
             "can't name vehicles. The COCO-pretrained model names car/truck/bus/"
             "person/bicycle/motorcycle out of the box — no retraining.")
    which = "named" if detector.startswith("Named") else "finetuned"

    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    uploaded = st.file_uploader("Upload a camera frame", type=["jpg", "jpeg", "png"])

    # Offer bundled images if the user has none handy.
    sample = None
    val_dir = ROOT / "yolo_dataset" / "images" / "train"
    if uploaded is None and val_dir.exists():
        samples = sorted(p for p in val_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".png"})[:50]
        if samples:
            pick = st.selectbox("…or pick a bundled sample frame",
                                ["(none)"] + [p.name for p in samples])
            if pick != "(none)":
                sample = val_dir / pick

    src = None
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read()); tmp.close()
        src = tmp.name
    elif sample is not None:
        src = str(sample)

    if src and st.button("Run detection"):
        with st.spinner("Loading model and running inference…"):
            model = load_yolo(which)
            kw = dict(source=src, conf=conf, verbose=False)
            if which == "named":
                kw["classes"] = COCO_ROAD_CLASSES  # only road objects
            res = model.predict(**kw)[0]
            annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
        c1, c2 = st.columns([3, 1])
        c1.image(annotated, caption="Detections", use_container_width=True)
        c2.metric("Objects detected", len(res.boxes))
        if len(res.boxes):
            confs = res.boxes.conf.cpu().numpy()
            c2.metric("Mean confidence", f"{confs.mean():.2f}")
            names = model.names
            cls = res.boxes.cls.cpu().numpy().astype(int)
            tally = {}
            for c in cls:
                tally[names[c]] = tally.get(names[c], 0) + 1
            c2.markdown("**By class**")
            for k, v in sorted(tally.items(), key=lambda x: -x[1]):
                c2.write(f"{k}: {v}")


# --------------------------------------------------------------------------- #
# BEV trajectory prediction
# --------------------------------------------------------------------------- #
elif task == "BEV trajectory prediction":
    st.header("BEV multi-modal trajectory prediction")
    if not BEV_WEIGHTS.exists():
        st.error(f"Checkpoint not found: {BEV_WEIGHTS}")
        st.stop()

    from data_preparation_v2 import load_lidar_points, create_bev_representation

    uploaded = st.file_uploader("Upload a LiDAR frame (.pcd.bin)", type=["bin"])
    frame_path = None
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        tmp.write(uploaded.read()); tmp.close()
        frame_path = tmp.name
    elif LIDAR_DIR.exists():
        frames = sorted(LIDAR_DIR.glob("*.bin"))
        if frames:
            names = [p.name for p in frames[:100]]
            pick = st.selectbox("…or pick a bundled LiDAR frame", names)
            frame_path = str(LIDAR_DIR / pick)

    if frame_path and st.button("Predict trajectory"):
        import torch
        with st.spinner("Building BEV and running the planner…"):
            points = load_lidar_points(frame_path)
            bev = create_bev_representation(points)
            model, device, ckpt = load_bev()
            bev_t = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = model(bev_t)
                trajs = out["trajectories"][0].cpu().numpy()
                mode_probs = out["mode_probs"][0].cpu().numpy()
                conf = out["confidence"][0].cpu().numpy()
                best, best_p = model.predict_best_trajectory(bev_t)
                best = best[0].cpu().numpy()

        ade = ckpt.get("val_ade", float("nan"))
        fde = ckpt.get("val_fde", float("nan"))
        st.write(f"Loaded **{points.shape[0]:,}** LiDAR points · "
                 f"checkpoint epoch {ckpt.get('epoch','?')} "
                 f"(val ADE {ade:.2f} m, FDE {fde:.2f} m)")

        col1, col2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(bev[:, :, 2], cmap="gray", origin="lower",
                  extent=[-50, 50, -50, 50], alpha=0.85)
        colors = ["#2ecc71", "#3498db", "#f39c12"]
        for m, (t, p) in enumerate(zip(trajs, mode_probs)):
            ax.plot(t[:, 0], t[:, 1], color=colors[m], lw=2,
                    alpha=0.3 + 0.6 * float(p), label=f"Mode {m+1} ({p:.2f})")
        ax.plot(best[:, 0], best[:, 1], color="#e74c3c", lw=3, label="Selected path")
        ax.scatter([0], [0], c="cyan", marker="*", s=200, edgecolors="k",
                   zorder=5, label="Ego")
        ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_aspect("equal")
        ax.set_xlabel("X (m) forward"); ax.set_ylabel("Y (m) left")
        ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
        ax.set_title("Predicted ego trajectories (BEV)")
        col1.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.bar(np.arange(len(conf)), conf, color="#3498db", edgecolor="k", alpha=0.8)
        ax2.set_title("Per-waypoint confidence")
        ax2.set_xlabel("Waypoint index (0.5 s steps)"); ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1); ax2.grid(alpha=0.3, axis="y")
        col2.pyplot(fig2)

        st.success(f"Selected mode {int(mode_probs.argmax())+1} "
                   f"with probability {best_p.item():.2f}")


# --------------------------------------------------------------------------- #
# Scene dashboard (multi-frame -> GIF)
# --------------------------------------------------------------------------- #
elif task == "Scene dashboard (multi-frame → GIF)":
    st.header("Scene dashboard — perception + planning + risk over a sequence")
    st.caption(
        "Builds a camera + BEV dashboard GIF over a chosen, time-ordered LiDAR "
        "scene. The front camera shows what's ahead; the BEV LiDAR shows the full "
        "surroundings, the planned path and the collision risk.")

    if not BEV_WEIGHTS.exists() or not LIDAR_DIR.exists():
        st.error("Need the BEV checkpoint and the LIDAR_TOP frames.")
        st.stop()

    import torch
    from data_preparation_v2 import load_lidar_points, create_bev_representation
    from collision_risk import RiskConfig, assess_risk, straight_fallback_path
    import scene_dashboard as sd

    all_lidar = sorted(LIDAR_DIR.glob("*.bin"))
    scenes = sd.group_scenes([str(p) for p in all_lidar])

    # ---- 1. pick the LiDAR scene + exact frame range ----
    st.subheader("1 · Select LiDAR frames")
    labels = [f"Scene {i+1}  ·  frames {s}–{e-1}  ({e-s} frames)"
              for i, (s, e) in enumerate(scenes)]
    si = st.selectbox("Scene (one continuous segment)", range(len(scenes)),
                      format_func=lambda i: labels[i])
    s_start, s_end = scenes[si]
    n_in_scene = s_end - s_start
    c1, c2, c3 = st.columns(3)
    offset = c1.slider("Start offset in scene", 0, max(0, n_in_scene - 2), 0)
    max_count = max(2, min(60, n_in_scene - offset))
    count = c2.slider("Number of frames", 2, max_count, min(16, max_count))
    stride = c3.slider("Frame stride", 1, 5, 1)
    sel_idx = list(range(s_start + offset, s_end, stride))[:count]
    lidar_sel = [str(all_lidar[i]) for i in sel_idx]
    st.caption(f"Selected {len(lidar_sel)} LiDAR frames, starting "
               f"`{all_lidar[sel_idx[0]].name}`")

    # ---- 2. front-camera source ----
    st.subheader("2 · Front-camera source")
    cam_dir_str = st.text_input(
        "Folder of front-camera images (point to your nuScenes CAM_FRONT folder "
        "for a coherent, time-synced GIF). Leave blank to use bundled samples.",
        value="")
    fps = st.slider("GIF fps", 2, 15, 6)
    det_label = st.radio("Camera detector",
                         ["Named objects (COCO)", "Fine-tuned nuScenes", "No camera panel"],
                         horizontal=True)
    use_planner = st.checkbox(
        "Use trajectory planner for the path (uncheck = constant-velocity baseline)",
        value=True)
    detector = "named" if det_label.startswith("Named") else "finetuned"
    show_cam = not det_label.startswith("No camera")

    # Resolve camera frames now so the user sees the match count before building.
    cam_sel = [None] * len(lidar_sel)
    if show_cam:
        if cam_dir_str.strip() and Path(cam_dir_str.strip()).is_dir():
            cams = sorted(str(p) for p in Path(cam_dir_str.strip()).iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            cams = sd.filter_front_camera(cams)
            cam_sel = sd.match_cameras_by_timestamp(lidar_sel, cams)
            matched = sum(c is not None for c in cam_sel)
            st.success(f"{len(cams)} front-camera frames found · matched "
                       f"{matched}/{len(lidar_sel)} to LiDAR by nearest timestamp.")
            if matched == 0:
                st.warning("No timestamp matches — is this the same scene as the "
                           "selected LiDAR frames?")
        else:
            if cam_dir_str.strip():
                st.warning("Folder not found; using bundled sample frames.")
            cam_dir = ROOT / "yolo_dataset" / "images" / "train"
            bundled = sorted(str(p) for p in cam_dir.iterdir()
                             if p.suffix.lower() in {".jpg", ".png"})[:len(lidar_sel)]
            cam_sel = (bundled + [None] * len(lidar_sel))[:len(lidar_sel)]
            st.info("Using bundled sample frames (mixed cameras/scenes — they will "
                    "look incoherent; point to your CAM_FRONT folder above).")

    # ---- 3. build ----
    if st.button("Build dashboard GIF"):
        cfg = RiskConfig()
        model = device = None
        if use_planner:
            model, device, _ = load_bev()
        yolo = load_yolo(detector) if show_cam else None

        frames, counts = [], {"CLEAR": 0, "CAUTION": 0, "BRAKE": 0}
        prog = st.progress(0.0, text="Rendering frames…")
        for i, lf in enumerate(lidar_sel):
            bev = create_bev_representation(load_lidar_points(lf))
            if use_planner:
                t = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    best, _ = model.predict_best_trajectory(t)
                path = best[0].cpu().numpy()
            else:
                path = straight_fallback_path(30, 20)
            risk = assess_risk(path, bev, cfg, horizon_s=20 * 0.5)
            counts[risk.level] += 1

            cam_rgb = None
            if show_cam and cam_sel[i] is not None:
                kw = dict(source=cam_sel[i], conf=0.25, verbose=False)
                if detector == "named":
                    kw["classes"] = COCO_ROAD_CLASSES
                cam_rgb = yolo.predict(**kw)[0].plot()[:, :, ::-1]

            frames.append(sd.render_dashboard_frame(
                bev, path, risk, i, len(lidar_sel), camera_rgb=cam_rgb,
                camera_caption="Front camera + detections"))
            prog.progress((i + 1) / len(lidar_sel), text=f"Frame {i+1}/{len(lidar_sel)}")

        gif = sd.frames_to_gif_bytes(frames, fps=fps, scale=0.7)
        prog.empty()
        st.image(gif, caption="Scene dashboard", use_container_width=True)
        st.download_button("Download GIF", gif, file_name="scene_dashboard.gif",
                           mime="image/gif")
        cc = st.columns(3)
        cc[0].metric("CLEAR frames", counts["CLEAR"])
        cc[1].metric("CAUTION frames", counts["CAUTION"])
        cc[2].metric("BRAKE frames", counts["BRAKE"])


# --------------------------------------------------------------------------- #
# VLM scene reasoning (local Ollama)
# --------------------------------------------------------------------------- #
else:
    st.header("VLM scene reasoning — local Ollama")
    st.caption("A local Vision-Language model reasons over the surround view + LiDAR BEV "
               "and the geometric decision, returning a scene caption, open-vocabulary "
               "hazards, agreement with CLEAR/CAUTION/BRAKE, and a recommended action.")

    import vlm_reasoner as vlm
    import scene_dashboard as sd
    from data_preparation_v2 import load_lidar_points, create_bev_representation
    from collision_risk import RiskConfig, assess_risk, straight_fallback_path

    host = st.text_input("Ollama host", vlm.OLLAMA_DEFAULT_HOST)
    models = vlm.list_ollama_models(host)
    if models:
        usable = [m for m in models if not vlm.is_embedding_model(m)]
        local = [m for m in usable if not vlm.is_cloud_model(m)]
        local_vision = [m for m in local if vlm.is_vision_model(m)]
        local_text = [m for m in local if not vlm.is_vision_model(m)]
        default = (local_vision or local_text or usable or models)[0]
        model = st.selectbox("Model", models, index=models.index(default))
        if not local_vision:
            st.info("No **local** vision model found. Easiest (no big download): "
                    "`ollama pull qwen3.5:cloud` — a cloud vision model that runs "
                    "remotely — then select it here. Fully-local option: "
                    "`ollama pull qwen2.5vl:7b` (~6 GB). Text models (`qwen3:8b`, "
                    "`phi3.5`) work now but reason over the perception summary only.")
    else:
        st.warning("Ollama not reachable — start it with `ollama serve`. "
                   "You can still type a model name to try.")
        model = st.text_input("Model name", "qwen3:8b")

    is_vision = vlm.is_vision_model(model) if model else False
    (st.success if is_vision else st.info)(
        f"**{model}** → " + ("VISION: the model will SEE the surround + BEV images."
                             if is_vision else
                             "TEXT-ONLY: the model reasons over the perception summary "
                             "(risk, distance, TTC) — it will not see the images."))

    cam_dir = st.text_input("Surround camera folder (optional nuScenes path). "
                            "Blank = bundled illustrative images.", "")
    frames = sorted(LIDAR_DIR.glob("*.bin")) if LIDAR_DIR.exists() else []
    if not frames:
        st.error("No LiDAR frames found in LIDAR_TOP/.")
        st.stop()
    fi = st.slider("LiDAR frame", 0, len(frames) - 1, min(552, len(frames) - 1))

    if st.button("Run VLM reasoning"):
        lf = str(frames[fi])
        with st.spinner("Building surround + LiDAR BEV…"):
            bev = create_bev_representation(load_lidar_points(lf))
            path = straight_fallback_path(30, 20)
            risk = assess_risk(path, bev, RiskConfig(), horizon_s=20 * 0.5)
            bev_render = sd.render_dashboard_frame(bev, path, risk, 0, 1, camera_rgb=None)
            if cam_dir and Path(cam_dir).is_dir():
                cams = [str(p) for p in Path(cam_dir).rglob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                cam_paths = sd.match_surround(lf, cams)
                st.caption(f"Matched {sum(v is not None for v in cam_paths.values())}/6 "
                           "cameras by timestamp.")
            else:
                bundled = sorted(str(p) for p in
                                 (ROOT / "yolo_dataset" / "images" / "train").glob("*.jpg"))[:6]
                cam_paths = {ch: (bundled[i] if i < len(bundled) else None)
                             for i, ch in enumerate(sd.NUSCENES_CAMERAS)}
            composite = sd.assemble_surround(cam_paths, bev_rgb=bev_render)

        st.image(composite, caption="Surround + LiDAR BEV — the VLM's input",
                 use_container_width=True)

        ctx = {"risk_level": risk.level,
               "nearest_obstacle_m": ("clear" if not np.isfinite(risk.nearest_dist_m)
                                      else f"{risk.nearest_dist_m:.1f} m"),
               "ttc_s": f"{risk.ttc_s:.1f} s" if risk.ttc_s else None,
               "ego_speed_kmh": round(risk.speed_mps * 3.6) if risk.speed_mps else None}
        labels = list(sd.NUSCENES_CAMERAS) + ["BEV_LiDAR"]
        backend = vlm.OllamaVLM(model=model, host=host)
        try:
            with st.spinner(f"Querying {model} via Ollama… (first call may load the model)"):
                analysis = vlm.analyze(backend, ctx,
                                       images=[sd.image_to_png_bytes(composite)],
                                       image_labels=labels)
        except Exception as e:
            if vlm.is_cloud_model(model):
                st.error(f"Ollama call failed: {e}\n\n`{model}` is a **cloud** model. "
                         "A 410 Gone usually means the model was retired — use a current "
                         "one: `ollama pull qwen3.5:cloud` (vision), then select it. "
                         "Ensure you're signed in (`ollama signin`) with Cloud enabled.")
            else:
                st.error(f"Ollama call failed: {e}\n\nIs `ollama serve` running and "
                         f"`{model}` available (`ollama list`)?")
            st.stop()

        c1, c2, c3 = st.columns(3)
        c1.metric("Geometric decision", risk.level)
        c2.metric("Nearest obstacle", ctx["nearest_obstacle_m"])
        c3.metric("VLM action", analysis.recommended_action or "—")

        st.subheader("VLM scene caption")
        st.write(analysis.caption or "—")
        st.subheader("Open-vocabulary hazards")
        if analysis.hazards:
            for h in analysis.hazards:
                st.write(f"- {h}")
        else:
            st.write("—")
        agree = analysis.agrees_with_decision
        st.subheader("✓ VLM corroborates the decision" if agree else
                     "✗ VLM disagrees with the decision" if agree is False else
                     "VLM rationale")
        st.write(analysis.rationale or "—")

        # --- spoken driver instruction (constructed instruction, then TTS) ---
        spoken = (getattr(analysis, "driver_instruction", "")
                  or (f"Action: {analysis.recommended_action}" if analysis.recommended_action else "")
                  or "Proceed with caution.")
        st.subheader("🔊 Spoken driver instruction")
        st.markdown(f"### \u201c{spoken}\u201d")
        import json as _json
        import streamlit.components.v1 as _components
        _t = _json.dumps(spoken)
        _components.html(
            '<button id="say" style="font-size:16px;padding:8px 16px;border-radius:8px;'
            'border:none;background:#2ecc71;color:white;cursor:pointer;">🔊 Read aloud</button>'
            '<script>'
            'const txt=' + _t + ';'
            'function pickEN(){var vs=window.speechSynthesis.getVoices();'
            'return vs.find(function(v){return /^en[-_]?US/i.test(v.lang);})||'
            'vs.find(function(v){return /^en/i.test(v.lang);})||null;}'
            'function say(){var u=new SpeechSynthesisUtterance(txt);u.lang="en-US";u.rate=1.0;'
            'var v=pickEN();if(v){u.voice=v;}'
            'window.speechSynthesis.cancel();window.speechSynthesis.speak(u);}'
            'document.getElementById("say").onclick=say;'
            'if(window.speechSynthesis.getVoices().length===0){'
            'window.speechSynthesis.onvoiceschanged=function(){};}'
            'setTimeout(function(){try{say();}catch(e){}},200);'
            '</script>',
            height=64)

        st.caption(f"model: {analysis.model} · "
                   + ("saw images" if analysis.used_images else "text-only (no image input)"))
