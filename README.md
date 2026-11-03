# Driving-risk review on nuScenes: a BEV trajectory planner trained at scale, collision risk, fleet safety scoring

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-BEV%20planner%20%2B%20YOLOv8-ee4c2c)
![Dataset](https://img.shields.io/badge/Dataset-nuScenes%20~200%20GB-success)
![Cloud](https://img.shields.io/badge/Trained%20on-AWS%20SageMaker-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**The business question.** Fleets and insurers record hours of driving and can't watch it all.
*Which moments should a safety manager actually look at, and how risky was each drive?*

**What this repo gives back.** A bird's-eye-view network trained on ~200 GB of nuScenes on AWS
SageMaker predicts where the car will be over the next 5.5 seconds; a risk layer checks that path
for obstacles; every drive gets a 0–100 safety score, and only the close calls go into a review
queue. On 28 minutes of real Boston and Singapore driving, that queue is **10 moments, 0.8 minutes
of footage, 3% of the recording.**

<p align="center">
  <img src="assets/demo_risk_clear.gif" width="32%"/>
  <img src="assets/demo_risk_caution.gif" width="32%"/>
  <img src="assets/demo_risk_brake.gif" width="32%"/>
</p>
<p align="center"><em>Real LiDAR scans, top-down. Blue: the path planner v4 predicts for the next 5.5 s, the lane that is checked. CLEAR · CAUTION · BRAKE.</em></p>

▶️ **[Watch the 50-second walkthrough](demo/demo.mp4)**: the question, the answer, one real drive, the model's results, how it's built.

---

## In plain words

1. **See.** Each LiDAR scan becomes a 100 m × 100 m top-down map of the car's surroundings.
2. **Predict.** A bird's-eye-view network (PyTorch, ~12 M parameters, trained on ~200 GB of
   nuScenes on AWS SageMaker) reads the map. Planner v4 adds the car's motion over the last second,
   measured by registering consecutive scans, and predicts three candidate paths for the next 5.5 s.
3. **Check.** The most likely path is searched for anything at least 0.5 m tall. First obstacle less
   than 5 m from the front bumper: **BRAKE**; under 15 m: **CAUTION**; otherwise **CLEAR**, with the gap
   and the time to reach it.
4. **Review.** Every drive is scored; every BRAKE moment *while moving* (not at a red light) is queued
   for a human, worst first, with a plain-language note written by a local language model.

## Who would use the output

| User | Question they have | What they get here |
|---|---|---|
| Fleet safety manager | Which clips do I watch, which drivers do I coach? | `fleet_review.py`: ranked close calls, minutes to review, per-drive score |
| Usage-based insurer | How risky is this vehicle's driving? | per-drive safety score and time at each risk level |
| AV / ADAS data team | Where are the hard scenes in our logs? | risk timeline per scene, multi-modal predicted paths, near-miss list |
| ADAS engineer | Is something in my path, and how soon? | per-frame CLEAR / CAUTION / BRAKE with gap and time-to-collision |

## Results

### The trajectory planner, tested on drives it never saw

5-fold cross-validation split by drive, on 1,945 key-frames from 67 scenes. Ground truth is the
car's real path over the next 5.5 s, recovered from the LiDAR itself (`odometry_labels.py`).

| Method | ADE | FDE | Sideways error at 3 s |
|---|---|---|---|
| **Planner v4** (BEV encoder trained at scale + measured motion) | **5.26 m** | **11.15 m** | **0.60 m** |
| Keep heading, constant speed | 5.56 m | 11.95 m | 0.69 m |
| Constant turn rate and speed (CTRV) | 5.69 m | 12.46 m | 0.78 m |
| Constant acceleration | 6.33 m | 14.71 m | 0.69 m |
| Epoch-5 planner (one scan, first release) | 18.65 m | 35.27 m | 1.23 m |

| Three candidate paths (best of 3, minADE@3) | ADE |
|---|---|
| **Planner v4, its 3 paths** | **2.94 m** |
| Physics, 3 speeds (slower / same / faster, tuned on the training drives) | 3.99 m |

Planner v4 beats every physics baseline on every measure: 7% lower final error and 13% lower
sideways error than the best one, and its three candidate paths cover the real future 26% better
than three physics guesses. It is 72% more accurate than the first release, which saw a single scan
(so it could not know the car's speed) and was trained with a flip augmentation on the wrong axis.

### The fleet review

![Fleet overview](assets/fleet_overview.png)

| Fleet review (all local data) | Value |
|---|---|
| Driving analysed | 28 min: 3,376 LiDAR key-frames, 68 scenes, Boston and Singapore |
| Time at each risk level (moving) | CLEAR 21.6 min · CAUTION 6.4 min · BRAKE 0.2 min |
| Close calls queued (BRAKE while moving) | 10, in 6 of 68 scenes |
| Footage left to watch (2 s lead-in each) | 0.8 min, 3% of the recording, worst first |

<p align="center">
  <img src="assets/safety_report_p1.png" width="45%"/>
  <img src="assets/safety_report_p2.png" width="45%"/>
</p>
<p align="center"><em>The per-drive safety report for the demo scene (<code>safety_report.py</code>).</em></p>

### The camera detector

| YOLOv8n fine-tuned on 20,256 nuScenes frames | Value | Note |
|---|---|---|
| Precision / recall | 0.64 / 0.30 | one generic "object" class, 20 epochs on CPU |
| mAP@0.50 / @0.50–0.95 | 0.35 / 0.21 | rose every epoch (0.04 → 0.35): compute-limited, not converged |

Every number comes from the training logs, the checkpoint metadata, `motion_planner.py` or
`fleet_review.py` output in this repo.

---

## How it works

```
   nuScenes (~200 GB, AWS SageMaker) ─► BEV encoder (PyTorch) ─┐
                                                                ▼
   LIDAR_TOP scans ─► BEV rasters ─► scene embedding ─► planner v4 heads ─► 3 paths + probabilities
   (.pcd.bin, 2 Hz)   500×500×3                              ▲
         │                                                   │
         ├─► scan registration (yaw + shift) ─► measured motion, 1 s history
         │
         └─► obstacle map (≥ 0.5 m tall) ─► risk layer ─► CLEAR / CAUTION / BRAKE + gap, TTC
                                                 ▼
                 safety_report.py (one drive) · fleet_review.py (all drives, review queue)
                                                 ▼
                         local language model (Ollama) ─► review notes
   CAM_FRONT images ─► YOLOv8 detector ─► named boxes (alongside; not yet fused)
```

- **Data pipeline** (`data_preparation_v2.py`): parses the nuScenes tables, builds the ground-truth
  future path in the car's frame (`p_ego = R⁻¹ · (p_world − t)`), and rasterises each LiDAR sweep
  into height / intensity / density channels at 0.2 m per cell (vectorised: ~8 ms per scan).
- **BEV network** (`trajectory_model.py`, `train_trajectory_v3.py`): VGG-style encoder → 256-d scene
  embedding → 3 trajectory heads, a mode-probability head and a per-waypoint confidence head
  (~12 M parameters). Trained on AWS SageMaker with a winner-takes-all Huber loss, a mode loss and a
  diversity term against "always go straight".
- **Self-supervised labels** (`odometry_labels.py`): each scan is registered on the previous one (a yaw
  search, then an FFT cross-correlation for the shift) to recover how the car moved; chaining 12 steps
  gives its real path over the next 5.5 s, with no poses or annotations needed.
- **Planner v4** (`motion_planner.py`): the trained encoder plus new heads that also see the measured
  motion (speed, drift, yaw rate and their change over the last second). Five heads trained from
  different seeds are averaged; they predict corrections to a constant-turn-rate extrapolation.
- **Risk layer** (`collision_risk.py`, pure NumPy): converts the path from the car's frame (x forward)
  into the LiDAR's frame (y forward), walks it from the front bumper inside a vehicle-wide corridor and
  stops at the first obstacle cell. TTC = gap / speed. Gaps while the car is stopped or crawling under
  7 km/h don't count as close calls.
- **Detector**: YOLOv8n fine-tuned on 20,256 nuScenes frames, or the stock COCO model filtered to road
  classes for named boxes.
- **Language layer** (`vlm_reasoner.py`): a local Ollama model (`qwen3:8b`) turns the measured numbers
  into a plain-language note and a spoken driver instruction. A vision path (six cameras + the LiDAR
  view to Qwen2.5-VL) is built and tested against a stub; it needs a time-synchronised camera set.
- **Recovering a lost model**: the planner's class definition was lost with the SageMaker instance.
  It was rebuilt from the 146 MB checkpoint alone and verified to load all 130 weight tensors exactly.

Full details, formulas and known issues: **[docs/Technical_Report.md](docs/Technical_Report.md)**.

---

## Run it

```bash
pip install -r requirements.txt               # CPU is enough for inference and for planner v4
# put nuScenes LIDAR_TOP key-frames (*.pcd.bin) in ./LIDAR_TOP (the free mini split works)

python odometry_labels.py                     # self-supervised labels + scene embeddings
python motion_planner.py                      # planner v4: 5-fold CV report + final heads
python fleet_review.py                        # all drives -> scores, review queue, chart
python fleet_review.py --from-csv --notes qwen3:8b   # + review notes from a local Ollama model
python safety_report.py --scene 66            # one drive -> PDF safety report
python demo/make_demo.py                      # rebuild every GIF, image and the walkthrough video
streamlit run app.py                          # interactive app: detection, trajectory, dashboard
python trajectory_model.py                    # sanity check: weights load with strict=True
```

Weights are tracked with Git LFS. nuScenes data is downloaded separately from nuscenes.org.

## Repo map

```
motion_planner.py      planner v4: heads on the trained encoder, cross-validation vs baselines
odometry_labels.py     self-supervised future paths from LiDAR scan registration
fleet_review.py        all drives -> per-drive score, ranked close calls, review time, notes
safety_report.py       one drive -> PDF safety report
collision_risk.py      frames, obstacle map, path check -> CLEAR / CAUTION / BRAKE (NumPy)
trajectory_model.py    BEV network architecture (encoder + heads)
train_trajectory_v3.py BEV network training (augmentation, diversity loss)
data_preparation_v2.py nuScenes -> BEV rasters + ground-truth future paths
risk_monitor.py        sequence -> annotated risk video
scene_dashboard.py     scene splitting, timestamp sync, dashboard frames
vlm_reasoner.py        language / vision-language layer (Ollama backend, prompts, JSON parsing)
vlm_analyze.py         surround cameras + BEV -> reasoning card
scene_narrator.py      per-scene narration (needs a vision model and synced cameras)
app.py                 Streamlit app
run_inference.py       batch inference (yolo, named, bev)
demo/make_demo.py      rebuilds the demo GIFs, report images and walkthrough video
docs/                  technical report
```

## What's next

1. Retrain the full BEV network end to end with the fixes in the technical report (lateral mirror,
   logits in the mode loss, a scene-level split) and the measured motion as an input from the start.
2. Train the detector multi-class on a GPU and project its boxes into the BEV once the camera
   calibration is wired in, so both sensors feed one obstacle map.
3. Track obstacles over time so TTC uses the closing speed, not only the car's own speed.
4. Run the vision-language path on synchronised cameras and score its hazard calls against nuScenes
   labels (construction zones, emergency vehicles, pedestrians).
5. Latency profiling and ONNX/TensorRT export for an embedded target.

## Changelog

- **November 2026: planner v4.** The first release's planner lost to a "keep heading" baseline. Root
  causes found in an audit: it saw one scan (so no speed), and its mirror augmentation flipped the
  forward axis instead of the lateral one. v4 keeps the encoder trained at scale, adds the measured
  motion, trains on self-supervised labels recovered from the LiDAR, and now beats every physics
  baseline on held-out drives (table above). It drives the risk check again.
- **November 2026: risk-layer audit** (details in the [technical report](docs/Technical_Report.md#85-corrections-made-in-november-2026)).
  The path was looked up in the LiDAR raster without converting frames (it searched to the right of
  the car); the obstacle height test ignored the sensor's 1.84 m mounting height; speed and TTC used
  a 10 s horizon instead of 5.5 s; distances are now gaps from the front bumper; stopped or crawling
  frames no longer count as close calls; scene splitting merged two recordings. Every demo was rebuilt
  from the model; clips produced by the offline stub instead of a real model were removed.
