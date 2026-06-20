"""
run_inference.py
================
One-command inference for the two trained models in this project, using only the
files that ship in this repo (no nuScenes download, no SageMaker needed).

    # YOLO object detection on the bundled validation images
    python run_inference.py yolo

    # BEV multi-modal trajectory prediction on the bundled LiDAR frames
    python run_inference.py bev

    # both
    python run_inference.py all

Outputs are written to ./inference_outputs/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "inference_outputs"
YOLO_WEIGHTS = ROOT / "runs" / "train" / "nuscenes_yolo" / "weights" / "best.pt"
BEV_WEIGHTS = ROOT / "checkpoint_v3" / "best_trajectory_model.pth"
LIDAR_DIR = ROOT / "LIDAR_TOP"


# --------------------------------------------------------------------------- #
# YOLO object detection
# --------------------------------------------------------------------------- #
# COCO road-object class ids -> named detection with no retraining.
COCO_ROAD_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck


def run_yolo(num_images=6, conf=0.25, named=False):
    from ultralytics import YOLO

    out_dir = OUT / ("yolo_named" if named else "yolo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather sample images: prefer a held-out images folder, else reuse training imgs.
    img_dir = None
    for cand in [ROOT / "yolo_dataset" / "images" / "val",
                 ROOT / "yolo_dataset" / "images" / "train"]:
        if cand.exists():
            img_dir = cand
            break
    if img_dir is None:
        print("No image folder found under yolo_dataset/images/.")
        return

    images = sorted([p for p in img_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:num_images]
    if named:
        print("Loading COCO-pretrained YOLOv8n (named road objects) ...")
        model = YOLO("yolov8n.pt")
        kw = dict(classes=COCO_ROAD_CLASSES)
    else:
        print(f"Loading fine-tuned weights: {YOLO_WEIGHTS}")
        model = YOLO(str(YOLO_WEIGHTS))
        kw = {}

    print(f"Running detection on {len(images)} images (conf>={conf}) ...")
    results = model.predict(source=[str(p) for p in images], conf=conf,
                            save=True, project=str(out_dir), name="pred",
                            exist_ok=True, verbose=False, **kw)
    total = sum(len(r.boxes) for r in results)
    tally = {}
    for r in results:
        for c in r.boxes.cls.cpu().numpy().astype(int):
            tally[model.names[c]] = tally.get(model.names[c], 0) + 1
    print(f"Detected {total} objects across {len(images)} frames.")
    if tally:
        print("By class: " + ", ".join(f"{k}={v}" for k, v in
              sorted(tally.items(), key=lambda x: -x[1])))
    print(f"Annotated images saved to: {out_dir / 'pred'}")


# --------------------------------------------------------------------------- #
# BEV trajectory prediction
# --------------------------------------------------------------------------- #
def _load_bev_inputs():
    """Build BEV tensors from the bundled .pcd.bin LiDAR frames."""
    from data_preparation_v2 import load_lidar_points, create_bev_representation

    frames = sorted(LIDAR_DIR.glob("*.bin"))
    if not frames:
        raise FileNotFoundError(f"No .bin LiDAR frames in {LIDAR_DIR}")
    return frames, load_lidar_points, create_bev_representation


def run_bev(num_frames=6):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trajectory_model import TrajectoryPlannerModel, NUSCENES_CATEGORIES

    out_dir = OUT / "bev"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading BEV weights: {BEV_WEIGHTS}")

    model = TrajectoryPlannerModel().to(device)
    ckpt = torch.load(str(BEV_WEIGHTS), map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)  # strict=True: verified to match exactly
    model.eval()
    print(f"Loaded epoch={ckpt.get('epoch','?')} "
          f"val_ade={ckpt.get('val_ade','?')} val_fde={ckpt.get('val_fde','?')}")

    frames, load_lidar_points, create_bev_representation = _load_bev_inputs()
    frames = frames[:num_frames]

    for i, frame in enumerate(frames):
        points = load_lidar_points(str(frame))
        bev = create_bev_representation(points)                       # (500,500,3)
        bev_t = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(bev_t)
            trajs = out["trajectories"][0].cpu().numpy()              # (3,20,2)
            mode_probs = out["mode_probs"][0].cpu().numpy()           # (3,)
            best_traj, conf = model.predict_best_trajectory(bev_t)
            best_traj = best_traj[0].cpu().numpy()

        # --- plot ---
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(bev[:, :, 2], cmap="gray", origin="lower",
                  extent=[-50, 50, -50, 50], alpha=0.85)
        colors = ["#2ecc71", "#3498db", "#f39c12"]
        for m, (t, p) in enumerate(zip(trajs, mode_probs)):
            ax.plot(t[:, 0], t[:, 1], color=colors[m], lw=2,
                    alpha=0.3 + 0.6 * float(p), label=f"Mode {m+1} ({p:.2f})")
        ax.plot(best_traj[:, 0], best_traj[:, 1], color="#e74c3c", lw=3,
                label="Selected path")
        ax.scatter([0], [0], c="cyan", marker="*", s=220, edgecolors="k",
                   zorder=5, label="Ego")
        ax.set_title(f"BEV trajectory prediction — frame {i}")
        ax.set_xlabel("X (m) forward"); ax.set_ylabel("Y (m) left")
        ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
        ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        save_path = out_dir / f"trajectory_{i:03d}.png"
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i+1}/{len(frames)}] {frame.name} -> {save_path.name} "
              f"(best mode {int(mode_probs.argmax())+1}, p={mode_probs.max():.2f})")

    print(f"Trajectory visualizations saved to: {out_dir}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["yolo", "named", "bev", "all"],
                    help="yolo=fine-tuned detector, named=COCO named objects, "
                         "bev=trajectory, all=everything")
    ap.add_argument("-n", "--num", type=int, default=6, help="number of frames")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.task in ("yolo", "all"):
        run_yolo(num_images=args.num, named=False)
    if args.task in ("named", "all"):
        run_yolo(num_images=args.num, named=True)
    if args.task in ("bev", "all"):
        run_bev(num_frames=args.num)


if __name__ == "__main__":
    main()
