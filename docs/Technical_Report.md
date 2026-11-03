---
title: "Autonomous Perception & Planning for ADAS"
subtitle: "A YOLO + BEV stack on nuScenes — architecture, algorithms, and business value"
date: "Technical Report"
---

# 1. Abstract

This report documents an end-to-end autonomous-driving perception and motion-planning
system built on the [nuScenes](https://www.nuscenes.org/) dataset and trained on AWS
SageMaker. It combines a **YOLOv8 camera object detector**, a **custom bird's-eye-view
(BEV) multi-modal trajectory planner** operating on LiDAR (planner v4: the encoder trained at
scale plus heads that see the car's measured motion, cross-validated against physics baselines),
and a **collision-risk fusion layer** that checks the predicted path against the LiDAR obstacle map
and returns an actionable CLEAR / CAUTION / BRAKE decision with distance-to-obstacle and
time-to-obstacle (TTC). The system is delivered as an interactive Streamlit application, a risk-monitor video
generator, a per-scene PDF safety report, and a fleet review that scores every recorded scene
and queues the moments worth a human look. This document covers the data pipeline, both model
architectures, the fusion algorithm, the camera–LiDAR synchronisation, the engineering
practices used (including reconstructing a lost model definition from its weights), the
measured results, and the business rationale.

# 2. Problem statement and business value

Forward-collision warning (FCW) and automatic emergency braking (AEB) are increasingly
mandated and incentivised by safety regulators worldwide. The underlying technical
capability is a closed loop: **perceive the surrounding scene, predict the ego vehicle's
near-future motion, and flag any conflict between the two.**

The same fused module is the input to three distinct commercial products, which is what
makes it valuable to build well once:

| Market | Consumes | Concrete output |
|---|---|---|
| ADAS / OEM safety systems | a real-time warn/brake trigger | per-frame CLEAR/CAUTION/BRAKE + TTC |
| Fleet management & usage-based insurance | a per-drive risk score | risk timeline + ranked near-miss events |
| AV research & data operations | scene triage and auto-labelling | multi-modal trajectories + scene risk profile |

The decisive quantities — **distance to the nearest in-path obstacle** and **time to reach
it** — are common to all three. This project therefore optimises for a correct, reproducible
decision pipeline rather than for a single benchmark metric.

# 3. System overview

```
   CAM_FRONT image ─► YOLOv8 detector ─► obstacles (boxes, named)
                                                         │
   LIDAR_TOP sweep ─► BEV rasterizer ─► BEV planner ─► predicted ego path
   (.pcd.bin)                          (multi-modal)    (3 modes, 20 wpts)
                                                         │
                          collision-risk fusion ◄────────┘
                          (path × LiDAR obstacle map)
                                   │
                          CLEAR / CAUTION / BRAKE  +  distance, TTC
```

Two sensors, two networks, one decision surface. The camera detector establishes *what*
is present; the BEV planner establishes *where the ego is going*; the fusion layer
establishes *whether that constitutes a hazard*.

# 4. Data pipeline (`data_preparation_v2.py`)

Development used the full ~200 GB nuScenes release on an AWS SageMaker notebook instance.
nuScenes stores its annotations as a relational set of JSON tables (`sample`,
`sample_data`, `ego_pose`, `calibrated_sensor`, `sample_annotation`, `instance`,
`category`). The pipeline performs four stages.

## 4.1 Metadata resolution
Each keyframe `sample` is linked to its `LIDAR_TOP` `sample_data` record (filtered on
`channel == 'LIDAR_TOP'` and `is_key_frame`), its `ego_pose`, and its 3-D box annotations
with resolved category names.

## 4.2 Ground-truth ego trajectory
The supervision signal is the ego vehicle's own future path. Future keyframe ego positions
are read, then transformed from the global frame into the **current ego frame** so the
trajectory is expressed relative to the car "now". For a future global position `p_world`,
current ego translation `t` and current ego yaw rotation `R`:

```
p_ego = R⁻¹ · (p_world − t)
```

The positions of the next **12 key-frames** (nuScenes key-frames are 2 Hz, so 0.5 s to 6.0 s
ahead) are interpolated to a fixed **20 waypoints** in `(x, y)` (x = forward, y = left),
giving a consistent regression target. The 20 waypoints therefore span **5.5 s**
(`PRED_HORIZON_S` in `collision_risk.py`).

## 4.3 LiDAR → BEV rasterization
Each LiDAR point cloud (nuScenes stores 5 floats/point: x, y, z, intensity, ring) is
filtered to a region of interest and projected onto a top-down grid:

- ROI: `x, y ∈ [−50, 50] m`, `z ∈ [−3, 3] m`; resolution `r = 0.2 m/cell` → a **500 × 500** grid.
- Grid index: `gx = ⌊(x + 50)/0.2⌋`, `gy = ⌊(y + 50)/0.2⌋`.
- **Channel 0 — height:** per-cell max of normalised `z`, `(z + 3)/6`.
- **Channel 1 — intensity:** per-cell max of `min(intensity/255, 1)`.
- **Channel 2 — density:** point count per cell, clipped to `count/10 ∈ [0, 1]`.

The output tensor is `(500, 500, 3)`, permuted to `(3, 500, 500)` for PyTorch.

## 4.4 Augmentation (training, `train_trajectory_v3.py`)
Because most driving is straight, a naive model collapses to "go straight". To force it to
learn turns, training applies, jointly to the BEV raster and the trajectory: **random
rotation** (±25°, applied as a 2-D rotation matrix to the waypoints and an image rotation to
the BEV) and **random horizontal flip** (mirror y and flip the BEV), turning left manoeuvres
into right ones and vice versa.

# 5. Model 1 — YOLOv8 obstacle detector

A YOLOv8-nano detector fine-tuned with Ultralytics. Training configuration (from
`runs/train/nuscenes_yolo/args.yaml`): base `yolov8n.pt`, image size 640, batch 16, 20
epochs, auto optimizer, **device = CPU**, single class (`nc = 1, names = ['object']`),
20,256 training images.

**Single-class rationale.** Collapsing all road objects to one `object` class keeps the
downstream warning logic trivial — any sufficiently-confident detection is a hazard — and
halves the labelling/learning burden on a CPU budget.

**Named detection without retraining.** Where object *type* matters, the system swaps in a
stock **COCO-pretrained YOLOv8** restricted to road classes (ids `0,1,2,3,5,7` = person,
bicycle, car, motorcycle, bus, truck). nuScenes frames are ordinary street scenes the COCO
model handles directly, so named detection is available at zero training cost.

**Metrics (epoch 20).** Precision 0.64, Recall 0.30, mAP@0.50 0.35, mAP@0.50–0.95 0.21.
mAP@0.50 rose monotonically from 0.04 (epoch 1) to 0.35 (epoch 20) and had not plateaued —
the detector is compute-limited, not converged.

# 6. Model 2 — BEV multi-modal trajectory planner

## 6.1 Architecture
Input `(3, 500, 500)` BEV → feature encoder → 256-d embedding → four heads. ~12M parameters.

| Block | Layers | Output |
|---|---|---|
| Encoder stem | Conv 7×7 s2 (3→64), BN, ReLU, MaxPool 3×3 s2 | 64 ch |
| Encoder layer1 | 2 × (Conv-BN-ReLU-Conv-BN), downsample | 128 ch |
| Encoder layer2 | 2 × double-conv, downsample | 256 ch |
| Encoder layer3 | 2 × double-conv, downsample | 512 ch |
| Embedding | GlobalAvgPool → Linear(512→256) → ReLU | 256-d |
| Trajectory heads | 3 × [Linear(256→256), ReLU, Dropout, Linear(256→40)] | 3 × (20×2) |
| Mode head | Linear(256→128), ReLU, Linear(128→3) → softmax | 3 probs |
| Confidence head | Linear(256→128), ReLU, Linear(128→20) | 20 (sigmoid) |
| Segmentation head | 4 × ConvTranspose (512→256→128→64→32), Conv 1×1 → 8 | 8-class BEV map |

## 6.2 Multi-modal formulation
Predicting one path hides risk: at a junction several futures are plausible. The network
therefore emits `M = 3` candidate trajectories `{τ_m}`, each 20 waypoints, with a categorical
mode distribution `π = softmax(mode_logits)`. The selected path is `τ* = τ_argmax(π)`. A
per-waypoint confidence and an auxiliary 8-class BEV semantic segmentation (background,
car, truck, bus, bicycle, motorcycle, pedestrian, movable-object) regularise the encoder.

## 6.3 Training objective
The v3 recipe (`train_trajectory_v3.py`: AdamW, weight decay 1e-4, 5 warm-up epochs then cosine decay,
lr 3e-4, batch 16, 80 epochs planned) combines:

- a **winner-takes-all Huber (smooth-L1) regression loss** on the mode closest to the ground
  truth (robust to outlier waypoints), weight 1.0;
- a **mode-classification loss** (cross-entropy towards the best mode, label smoothing 0.15),
  weight 0.5;
- a **diversity hinge** `max(0, 10 m − mean distance between mode endpoints)`, weight 0.15,
  directly countering the straight-line collapse;
- **lane and collision penalties** (weights 0.2 and 0.5) that look up each waypoint in a lane
  mask and an obstacle mask.

Known issues found in the November 2026 review, to fix before the next training run:

1. The mode loss received the softmaxed `mode_probs` instead of logits, so softmax was applied
   twice and the mode head learned from a flattened signal. Fixed in the script (now uses
   `mode_logits`); the epoch-5 checkpoint was trained before the fix.
2. The lane and collision penalties convert waypoint coordinates to integer grid indices, so no
   gradient flows through them; they add a constant to the loss. A differentiable version would
   sample the masks with `grid_sample`.
3. Train and validation are split 80/20 by sample, not by scene, so neighbouring frames of the
   same drive appear in both and the validation ADE/FDE are optimistic.
4. The segmentation head exists in the architecture but is not supervised by the v3 loss.
5. The mirror augmentation flips the forward axis instead of the lateral one (`TF.hflip` on a
   `(lateral, forward)` tensor; should be `TF.vflip`). See 8.7 for the measured effect.

## 6.4 Evaluation metrics
Standard motion-forecasting metrics, over `T = 20` waypoints with prediction `p_t` and
ground truth `g_t`:

```
ADE = (1/T) Σ_t ‖ p_t − g_t ‖₂        (average displacement error)
FDE = ‖ p_T − g_T ‖₂                   (final displacement error)
```

Checkpoint metadata (epoch 5): **val ADE 3.29 m, val FDE 7.23 m, val loss 3.83**. This is an
early checkpoint with a sensible "continue forward" prior but limited turn commitment.

# 7. Recovering a lost model from its weights

The original `trajectory_model.py` class definition was lost with the SageMaker instance;
only the 146 MB checkpoint survived. The architecture was reconstructed by:

1. Reading the checkpoint's `state_dict` and extracting every parameter's **name and tensor
   shape** (a custom `pickle` unpickler stubbing out the tensor-rebuild ops, so no GPU/torch
   build was needed).
2. Inferring each module from its key path and shapes (e.g. `bev_encoder.layer1.0.0.weight`
   of shape `(128, 64, 3, 3)` → first conv of a double-conv block, 64→128).
3. Rebuilding the `nn.Module` so it reproduces **all 130 parameter tensors** with identical
   names and shapes — verified by a set-difference check (0 missing, 0 unexpected, 0 shape
   mismatches), guaranteeing `load_state_dict(strict=True)` succeeds.

The reconstructed class also matched the original output visualisations independently
(3 modes, 20-waypoint confidence, the 8 segmentation classes), confirming correctness.

# 8. Collision-risk fusion (`collision_risk.py`)

The fusion layer is pure NumPy (unit-testable, no torch) and converts a predicted path plus
the BEV raster into a decision.

## 8.1 Two coordinate frames
The planner predicts waypoints in the **ego frame** (x forward, y left, origin at the rear
axle, as nuScenes defines ego poses). The BEV raster is built from LIDAR_TOP points in the
**sensor frame**, where **x points right and y points forward**, origin at the roof LiDAR
(about 0.94 m ahead of the rear axle and 1.84 m above the road). Raster axis 0 is sensor x,
axis 1 is sensor y. Every lookup and every plot converts between the two:

```
sensor_x = -ego_y
sensor_y =  ego_x - 0.94
```

This was verified on the data itself: aligning consecutive scans of a moving car by
cross-correlation, the scene shifts along sensor y (up to 12 m per second) and never along
sensor x. The planner learned the mapping from raster to ego-frame path during training, so
it is unaffected; the risk layer and the plots were not, see 8.5.

## 8.2 Obstacle map
A BEV cell is an obstacle if it is **dense** (more than 2 returns) and **tall** (its highest
return is at least 0.5 m above the road). The height channel stores `(z + 3) / 6` in the
sensor frame and the road sits at `z ≈ -1.84 m`, so:

```
obstacle(c) = density(c) > 0.2  AND  height(c) > (0.5 - 1.84 + 3) / 6  (= 0.277)
```

Cells inside the car's own footprint (rear bumper -0.9 m to front bumper +3.4 m, ±1 m
sideways) are removed, because the roof LiDAR also sees the vehicle itself.

## 8.3 Which path is scanned, and the corridor scan
By default the scanned path is **planner v4's most likely path** (section 8.7), extended along its
last direction until it covers at least the CAUTION range. When the car's motion could not be
measured (the first frames of a scene, or a weak scan registration), the fallback is the lane along
the current heading, as long as the distance covered in 5.5 s. `--path heading` forces the
fallback everywhere.

The path, prefixed with the car's own position (the first waypoint is already 0.5 s ahead), is
resampled by arc length and walked outward from the front bumper. At each point a ±1.0 m window
(±5 cells, the car's half-width plus a margin) is tested against the obstacle map. On a hit, the
distance is measured to the obstacle cells themselves, giving the **gap from the front bumper**, `d`.

```
for point (x, y) along the path, from the car outward:
    if x < 3.4 m: continue                                 # still inside the car
    if any obstacle cell within ±1 m of (x, y) in the raster:
        d = min over those cells of hypot(cx, cy) - 3.4 m; break
```

## 8.4 Speed, TTC and decision
**Motion is measured, not predicted** (`odometry_labels.register`). Returns 0.3–3 m above the road
are taken from two consecutive scans, in the ego frame. The later scan is rotated through a yaw search
(±8° in 0.5° steps, scored on a coarse 0.4 m grid), then the shift is found at full 0.2 m resolution
from the FFT cross-correlation of the two occupancy grids. The result is how far the car moved
forward and sideways and how much it turned in 0.5 s. The same filtering as in training (a median
over the last three steps) is applied before the motion is used. A weak peak (normalised
correlation below 0.15) returns no motion.

```
v = |shift| / 0.5 s
TTC = d / v        (the car's own speed; obstacles treated as static)
```

Decision thresholds on the gap from the front bumper: `d < 5 m → BRAKE`,
`5 ≤ d < 15 m → CAUTION`, else `CLEAR`. For scores and review, a frame where the car is
stopped or crawling (measured speed under 2 m/s, about 7 km/h) counts as CLEAR: a short gap at a
red light or in a queue is not a close call (`effective_level`).

The decision uses the LiDAR obstacle map only. The camera detector runs alongside it but does
not feed the decision yet: projecting camera boxes into the BEV needs the calibration
matrices (section 9).

## 8.5 Corrections made in November 2026
1. **Frames.** The corridor scan indexed the sensor-frame raster with ego-frame coordinates,
   so it searched along sensor x, i.e. **to the right of the car**, not ahead of it. Plots
   drew the raster with its lateral axis mirrored. All outputs before this fix (decisions,
   reports, demo GIFs) were measured on the wrong axis and have been regenerated.
2. **Obstacle height.** The old threshold (`height > 0.55`, i.e. z > 0.3 m in the sensor
   frame) meant "more than about 2.1 m above the road", so most cars and pedestrians did not
   count as obstacles. It is now 0.5 m above the road.
3. **Horizon.** Every caller passed `20 × 0.5 s = 10 s` as the path duration, which
   under-stated speed by about 45% and over-stated TTC by about 1.8x. All callers now use
   `PRED_HORIZON_S = 5.5 s`.
4. **Reference point.** Distances were measured from the rear axle with a 3 m dead zone,
   which let obstacles between 3 m and the bumper count as "in the car". Distances are now
   gaps from the front bumper, the car's own footprint is masked, and the thresholds were
   restated accordingly (5 m / 15 m from the bumper ≈ the old 8 m / 20 m from the axle).
5. **Path and speed.** Speed used to be implied by the epoch-5 planner's path. It is now measured
   by scan registration, the path comes from planner v4 (8.7), and stopped or crawling frames no
   longer count as close calls.

## 8.6 Validation
On all 3,376 key-frames, with planner v4 choosing the path, the layer gives 1% BRAKE, 23% CAUTION and
77% CLEAR (stopped or crawling frames, 31% of the drive, counted as CLEAR), with a median gap of 10.9 m
when something is in the path: plausible for dense urban driving where the car often follows
traffic. Rendered frames were checked by eye: road edges run parallel to the predicted path, which
bends with the road in turns, and BRAKE hits sit ahead of the car. The thresholds are fixed distances;
nuScenes has no crash or near-miss labels, so they are not tuned or validated against outcomes.

## 8.7 Planner v4: the trained encoder plus measured motion (`motion_planner.py`)
**Why a v4.** Tested against the car's real motion, the epoch-5 planner was worse than simply
assuming the car keeps its heading. Two causes in the original design and training code:

1. It sees a single LiDAR scan, from which the car's speed cannot be known, so its forward distance
   was badly off.
2. The BEV tensor is `(channels, sensor x, sensor y)`, i.e. `(lateral, forward)`, and the "mirror left
   and right" augmentation used `TF.hflip`, which flips the **last** axis, the forward one. Half of the
   training samples paired a scene mirrored front-to-back with a target path mirrored left-to-right,
   which teaches an inconsistent sideways direction. (Fixed in the script: `TF.vflip`.)

**Labels without poses (`odometry_labels.py`).** The local data has no ego poses, so the car's real
path is recovered from the LiDAR: every consecutive pair of scans is registered (yaw search, then an
FFT shift, section 8.4), each 0.5 s step is median-filtered against its neighbours to remove the
occasional misfire, and 12 steps are chained into the positions 0.5 s to 6.0 s ahead, resampled to the
same 20 waypoints the planner was trained on. Frames whose registrations are weak are left out:
1,945 of the 3,376 key-frames have clean labels.

**Model.** The BEV encoder trained on ~200 GB of nuScenes on AWS SageMaker is kept as is; its 256-d
scene embedding is computed once per frame (and once on the correctly mirrored raster). New heads
(two 256-unit layers, dropout 0.3) take the embedding plus five motion features (forward speed,
sideways speed, yaw rate, and the change in forward speed and yaw rate over the last 0.5 s) and predict
three candidate paths as corrections to a constant-turn-rate-and-velocity extrapolation, plus a
probability for each. Loss: winner-takes-all Huber, mode cross-entropy on logits (label smoothing 0.1),
and a small diversity hinge. AdamW, weight decay 1e-2, 40 epochs. Five heads trained from different
seeds are averaged (the most likely path of each); the three candidate paths come from the first.

**Evaluation.** 5-fold cross-validation split by scene: no frame of a test drive is seen in training.
Every method is scored against the same labels.

| Method (5-fold, held-out scenes) | ADE | FDE | sideways error at 3 s |
|---|---|---|---|
| Epoch-5 planner (one scan) | 18.65 m | 35.27 m | 1.23 m |
| Keep heading, constant speed | 5.56 m | 11.95 m | 0.69 m |
| Constant acceleration | 6.33 m | 14.71 m | 0.69 m |
| Constant turn rate and speed (CTRV) | 5.69 m | 12.46 m | 0.78 m |
| Planner v4 (this work) | 5.26 m | 11.15 m | 0.60 m |
| Three candidate paths (best of 3, minADE@3) | ADE |
|---|---|
| Physics, 3 speeds (slower / same / faster) | 3.99 m |
| Planner v4, its 3 paths | 2.94 m |

Planner v4 beats every physics baseline on every measure: 5% lower average error, 7% lower final
error and 13% lower sideways error than the best one. Its three candidate paths reach a best-of-3
error of 2.94 m against 3.99 m for three physics hypotheses (slower / same / faster speed, with the
factors tuned on the training drives), 26% lower. It is 72% more accurate than the epoch-5 planner.

**Limits.** The labels come from translation-plus-yaw registration, not from the vehicle's poses, so
they carry some noise (scans dominated by moving traffic align less well, and are filtered out). The
numbers compare methods on equal terms; they are not nuScenes benchmark scores.

# 9. Camera–LiDAR synchronisation (`scene_dashboard.py`)

The dashboard pairs the front camera with the BEV view. Because the bundled package ships
**without nuScenes calibration matrices** (camera intrinsics `K` and the LiDAR→camera
extrinsic transform), a true point-on-image projection is not possible; the field-standard
multi-panel layout is used instead (as the nuScenes devkit itself renders).

Synchronisation is exact in **time**, however, using filename timestamps:

- **Timestamp parse** — the trailing microsecond integer in each nuScenes filename.
- **Scene grouping** — a time-sorted frame list is split wherever the inter-frame gap
  exceeds 2 s, yielding the dataset's continuous scenes (68 in the bundled LiDAR set).
- **Front-camera filter** — keep only the exact `CAM_FRONT` token (excluding
  `CAM_FRONT_LEFT/_RIGHT` and the rear cameras), so the dashboard never mixes viewing
  directions.
- **Scene split fix** — the gap test uses the absolute time difference, so a new log whose
  timestamps start earlier than the previous one also starts a new scene (68 scenes in the
  bundled set, previously reported as 67).
- **Nearest-timestamp match** — each selected LiDAR frame is paired to the front-camera
  frame with the closest timestamp (within ±0.5 s), giving frame-accurate sync.

A `project_lidar_to_image()` function is included as the explicit upgrade path: supply `K`
and the 4×4 transform and the dashboard switches to a true depth-coloured LiDAR overlay
(`uv = K · [R|t] · p`, keeping points with positive camera-frame depth).

# 10. VLM reasoning layer (`vlm_reasoner.py`, `vlm_analyze.py`)

The geometric pipeline is closed-vocabulary and non-explanatory. A local Vision-Language
Model (VLM) adds an open-vocabulary semantic layer that reasons over the entire sensor suite.

## 10.1 Inputs
The VLM receives one composite image — a 2×3 montage of the six nuScenes cameras
(`CAM_FRONT_LEFT/FRONT/FRONT_RIGHT`, `CAM_BACK_LEFT/BACK/BACK_RIGHT`) plus the rendered LiDAR
BEV tile — together with a text summary of the geometric stack's output (risk level, nearest
in-path obstacle, TTC, ego speed, per-camera detections). Cameras are paired to the LiDAR
frame by nearest filename timestamp (±0.5 s); the LiDAR is supplied as the BEV render because
VLMs cannot ingest raw point clouds. Reasoning over the full surround — not just the front
camera the driver already sees — is the point: the model synthesises views no single human
perspective provides.

## 10.2 Task and output
A constrained prompt casts the model as the high-level ADAS reasoning module and requests
strict JSON (Ollama `format=json`): a scene caption, an open-vocabulary hazard list,
agreement with the CLEAR/CAUTION/BRAKE decision, a rationale, and a recommended action. The
hazard vocabulary deliberately targets what a bounding-box detector cannot label —
construction zones, emergency vehicles, traffic-light state, debris, animals, vulnerable road
users, and road/weather conditions.

## 10.3 Backend
`OllamaVLM` posts to a local [Ollama](https://ollama.com) server (`/api/chat`, base64
images). The default model is `qwen2.5vl:7b` (Qwen2.5-VL, multi-image). Text-only models such
as `qwen3:8b` are auto-detected and receive only the text summary (no images). A `MockVLM`
returns schema-valid output so the full pipeline — montage assembly, PNG encoding, prompt
construction, JSON parsing, card rendering — is testable with no GPU and no server.

## 10.4 What has actually been run
- **Text path, run live.** `qwen3:8b` through Ollama, reasoning over the measured numbers
  (risk level, distance, TTC, speed). It writes the plain-language review notes in the fleet
  review (section 11.1) and the spoken driver instruction in the app.
- **Vision path, built and tested against the stub only.** Sending the six cameras and the BEV
  to Qwen2.5-VL is implemented end to end, but it has not been run with a real vision model
  on time-synchronised cameras: the camera images kept locally are named by token, not by
  timestamp, and the metadata that links them to LiDAR frames went with the SageMaker
  instance. Earlier demo cards and narrated clips produced by the stub were removed from the
  repository in November 2026 so that no placeholder text is presented as model output.
- **Next step.** Run Qwen2.5-VL on a synchronised split (nuScenes mini) and score its hazard
  list against nuScenes' own labels (construction vehicles, barriers, traffic cones, emergency
  vehicles, pedestrians), so the language layer gets a measured accuracy like every other part.

## 10.5 Why it matters
This converts a purely geometric system into a **neuro-symbolic** one: geometry supplies
precise, calibrated measurement (distance, TTC); the VLM supplies meaning (what the obstacle
is, what the scene context implies) and a human-readable justification — an independent
corroboration channel and the foundation for explainable, auditable ADAS decisions.

## 10.6 Narrated-scene tool (`scene_narrator.py`)
The LiDAR stream is split into scenes by timestamp gaps; 4–5 are auto-selected for risk
variety (clear / caution / brake). For each scene the tool builds, per frame, the 6-camera
surround montage plus the LiDAR BEV, then asks the VLM for a short "co-pilot" comment per
temporal segment and a scene summary over several key-frames (one multi-image call). Output:
a subtitled clip (GIF + MP4) and a transcript per scene. Like 10.4, it needs a real vision
model and synchronised cameras to produce meaningful narration.

# 11. Inference and delivery

- **`app.py`** — Streamlit app with three tabs: detection (fine-tuned/named), single-frame
  trajectory + risk, and the multi-frame scene dashboard (scene/range selection, CAM_FRONT
  folder, timestamp pairing) exporting an animated GIF.
- **`run_inference.py`** — batch inference (`yolo` / `named` / `bev` / `all`).
- **`risk_monitor.py`** — sequence → annotated MP4/GIF risk monitor.
- **`safety_report.py`** — one scene → PDF (safety score, risk composition, timeline,
  ranked risk events).
- **`fleet_review.py`** — every recorded scene → per-scene scores, a ranked list of BRAKE
  events, the minutes of footage worth watching, and optional plain-language review notes.

## 11.1 The business question: which moments should a safety manager watch?
A fleet or an insurer cannot watch every hour of recorded driving. `fleet_review.py` runs the
whole recording through planner v4 and the risk layer and turns it into a review queue.

Run on all 3,376 local key-frames (28 minutes, 68 scenes, Boston and Singapore):

- Time at each level, counting stopped or crawling frames as CLEAR: CLEAR 21.6 min, CAUTION 6.4 min,
  BRAKE 0.2 min.
- 10 close calls (BRAKE while moving) in 6 scenes. Watching each one with up to 2 s of lead-in takes
  0.8 minutes, 3% of the recording.
- Each scene gets the `safety_report.py` score; events are ranked by the smallest gap. With
  `--notes qwen3:8b`, a local language model writes a one-line note and a "what to check" line per
  top event from the measured numbers only.

Caveats: TTC uses the car's own speed and treats obstacles as static, so following a car closely
reads as a close call (which is also what a driver coach wants to see); the predicted path comes from
one scan plus the measured motion, so it can be wrong in a sharp turn.

# 12. Engineering practices

- **Reproducibility** — relative paths, pinned `requirements.txt`, one-command entry points.
- **Honesty** — every reported number is read from the training logs / checkpoint metadata;
  the detector's under-training and the planner's early-checkpoint straight-line bias are
  documented rather than hidden.
- **Testability** — the risk fusion and synchronisation logic are pure NumPy and validated
  directly on real frames.
- **Recoverability** — a lost model architecture was rebuilt from weights and formally
  verified.

# 13. Limitations and future work

1. **Planner** — v4 trains new heads on a frozen epoch-5 encoder. Retraining the whole network
   end to end with the fixes in 6.3 and the measured motion as an input would lower the error
   further; labels from vehicle poses would remove the registration noise.
2. **Detector** — single-class, CPU-trained, still improving at epoch 20; multi-class GPU
   training would raise mAP and enable typed hazards end-to-end.
3. **Cross-sensor fusion** — risk currently uses the LiDAR obstacle map only; with
   calibration, camera detections can be projected into the BEV frame for one fused map.
4. **Deployment** — latency profiling and ONNX/TensorRT export for an embedded target.
5. **Validation of the risk layer** — thresholds are fixed distances; with labelled
   near-misses (or the ground-truth future path from nuScenes poses) the queue could be scored
   for precision and recall.
6. **Speed and TTC** — speed comes from scan registration (good over 0.5 s, weaker in heavy
   traffic); TTC ignores the obstacle's own motion. Tracking obstacles over time would give the
   closing speed.
7. **Vision-language path** — see 10.4.

# 14. Technology stack

| Layer | Tools |
|---|---|
| Modelling | PyTorch, Ultralytics YOLOv8 |
| Language layer | Ollama · qwen3:8b (run) · Qwen2.5-VL (vision path, stub-tested) |
| Data | nuScenes, NumPy, custom BEV rasterizer |
| Training | AWS SageMaker (notebook instances) |
| Fusion / analytics | NumPy, SciPy |
| Visualisation / delivery | Matplotlib, OpenCV, Pillow, Streamlit |
| Reporting | Matplotlib PdfPages, pandoc |

# 15. Key figures

| Item | Value |
|---|---|
| nuScenes raw dataset | ~200 GB |
| YOLO training images | 20,256 |
| Bundled LiDAR frames / scenes | 3,376 / 68 (28 min of driving) |
| Planner v4 (5-fold CV) ADE / FDE / sideways at 3 s | 5.26 m / 11.15 m / 0.60 m |
| Best physics baseline ADE / FDE / sideways at 3 s | 5.56 m / 11.95 m / 0.69 m |
| Planner v4 best of 3 paths vs 3 physics guesses | 2.94 m vs 3.99 m |
| Review queue | 10 close calls, 0.8 of 28 min (3%) |
| BEV raster | 500 × 500 × 3 |
| Planner parameters / tensors | ~12M / 130 |
| Trajectory modes / horizon | 3 / 20 waypoints spanning 5.5 s |
| YOLO mAP@0.50 / @0.50–0.95 | 0.35 / 0.21 |
| Epoch-5 planner val ADE / FDE (training split) | 3.29 m / 7.23 m |
