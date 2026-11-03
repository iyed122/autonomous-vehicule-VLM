"""
motion_planner.py
=================
Planner v4: the trained BEV encoder from the epoch-5 checkpoint, plus new prediction
heads that also see the car's measured motion over the last second (speed, sideways
drift, yaw rate, and how speed and yaw rate changed). The original planner saw a single
LiDAR scan, from which the car's speed cannot be known; that, and a flip augmentation on
the wrong axis, is why it lost to a "keep heading" baseline (first row of the table).

The heads predict 3 candidate paths (20 waypoints, 0.5-6.0 s) as corrections to a
constant-turn-rate-and-velocity (CTRV) extrapolation, plus a probability per path.
Labels are self-supervised: the car's real future path recovered from the LiDAR by
odometry_labels.py, with each 0.5 s registration step median-filtered against its
neighbours to remove the occasional misfire. Training mirrors every scene left-right
(the lateral axis, this time); five heads trained from different seeds are averaged
(an ensemble), which takes a few minutes on a CPU.

    python odometry_labels.py      # once: per-step motion + encoder embeddings
    python motion_planner.py       # 5-fold cross-validation by scene, then the final heads
                                   # -> checkpoint_v4/motion_heads.pt
                                   # -> inference_outputs/motion_planner_eval.md

Every result is measured on scenes the heads never saw (5 folds, split by scene), against
the same self-supervised labels for every method, so it is a like-for-like comparison,
not a nuScenes-benchmark number.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "inference_outputs"
CKPT = ROOT / "checkpoint_v4" / "motion_heads.pt"
K, T, DT, STEPS = 3, 20, 0.5, 12
TIMES = 0.5 + 5.5 * np.arange(T) / (T - 1)          # seconds ahead of each waypoint
MIN_QUALITY = 0.25
N_MOTION = 5      # v_fwd, v_left, yaw_rate, change in v_fwd, change in yaw_rate (per 0.5 s)
EPOCHS, WEIGHT_DECAY, DROPOUT, SEEDS = 40, 1e-2, 0.3, (0, 1, 2, 3, 4)


# --------------------------------------------------------------------- motion ----
def median_steps(steps, centered):
    """Median-filter (fwd, left, yaw) registration steps over 3 neighbours.
    centered=True for labels (may use the future); False for inputs (past only)."""
    s = np.asarray(steps, dtype=np.float64)[:, :3]
    out = np.empty_like(s)
    for j in range(len(s)):
        lo, hi = (j - 1, j + 2) if centered else (j - 2, j + 1)
        out[j] = np.median(s[max(0, lo):min(len(s), hi)], axis=0)
    return out


def motion_features(past, j):
    """Inputs for frame j from causally filtered steps (step i = scan i -> i+1).
    Needs the two steps before frame j; returns None otherwise."""
    if j < 2:
        return None
    cur, prev = past[j - 1] / DT, past[j - 2] / DT
    return np.array([cur[0], cur[1], cur[2], cur[0] - prev[0], cur[2] - prev[2]], np.float32)


def chain(steps):
    """Chain (fwd, left, yaw) steps into positions in the first frame."""
    pos, heading, out = np.zeros(2), 0.0, []
    for fx, fy, psi in steps:
        c, s = np.cos(heading), np.sin(heading)
        pos = pos + np.array([c * fx - s * fy, s * fx + c * fy])
        heading += psi
        out.append(pos.copy())
    out = np.array(out)
    t0, t1 = np.linspace(0, 1, len(out)), np.linspace(0, 1, T)
    return np.stack([np.interp(t1, t0, out[:, 0]), np.interp(t1, t0, out[:, 1])], axis=1)


# ------------------------------------------------------------------ baselines ----
def ctrv(motion, times=TIMES):
    """Constant turn rate and velocity from [v_fwd, v_left, yaw_rate, ...] -> (N, T, 2)."""
    m = np.asarray(motion, dtype=np.float64)
    v, w = np.hypot(m[:, 0], m[:, 1])[:, None], m[:, 2][:, None]
    slip = np.arctan2(m[:, 1], np.maximum(m[:, 0], 1e-6))[:, None]
    t = times[None, :]
    small = np.abs(w) < 1e-3
    w_safe = np.where(small, 1.0, w)
    x = np.where(small, v * t, v / w_safe * np.sin(w_safe * t))
    y = np.where(small, 0.0, v / w_safe * (1 - np.cos(w_safe * t)))
    c, s = np.cos(slip), np.sin(slip)
    return np.stack([c * x - s * y, s * x + c * y], axis=-1).astype(np.float32)


def heading_constant_speed(motion, times=TIMES):
    v = np.hypot(motion[:, 0], motion[:, 1])[:, None]
    return np.stack([v * times, np.zeros_like(v * times)], -1).astype(np.float32)


def constant_acceleration(motion, times=TIMES):
    """Straight ahead, speed changing at the last measured rate (clipped), never reversing."""
    v = np.hypot(motion[:, 0], motion[:, 1])[:, None]
    a = np.clip(motion[:, 3] / DT, -4.0, 3.0)[:, None]
    t = times[None, :]
    t_stop = np.where(a < 0, -v / np.where(a < 0, a, -1.0), np.inf)
    te = np.minimum(t, t_stop)
    x = v * te + 0.5 * a * te ** 2
    return np.stack([x, np.zeros_like(x)], -1).astype(np.float32)


# ---------------------------------------------------------------------- model ----
class MotionHeads(nn.Module):
    def __init__(self, emb_dim=256, hidden=256, n_motion=N_MOTION, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim + n_motion, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.paths = nn.Linear(hidden, K * T * 2)
        self.logits = nn.Linear(hidden, K)
        nn.init.zeros_(self.paths.weight)
        nn.init.zeros_(self.paths.bias)

    def forward(self, emb, motion_scaled, base):
        h = self.net(torch.cat([emb, motion_scaled], dim=1))
        resid = self.paths(h).view(-1, K, T, 2)
        return base[:, None] + resid, self.logits(h)


def wta_loss(paths, logits, target):
    err = F.smooth_l1_loss(paths, target[:, None].expand_as(paths), reduction="none").sum(-1).mean(-1)
    best = err.argmin(1)
    reg = err.gather(1, best[:, None]).mean()
    cls = F.cross_entropy(logits, best, label_smoothing=0.1)
    ends = paths[:, :, -1]
    spread = torch.cdist(ends, ends).sum((1, 2)) / (K * (K - 1))
    return reg + 0.5 * cls + 0.05 * F.relu(2.0 - spread).mean()


def scale_motion(m, stats):
    return (m - stats["mean"]) / stats["std"]


def fit(emb, motion, target, stats, epochs=EPOCHS, seed=0):
    torch.manual_seed(seed)
    model = MotionHeads(emb.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    E = torch.from_numpy(emb).float()
    M = torch.from_numpy(scale_motion(motion, stats)).float()
    B = torch.from_numpy(ctrv(motion))
    Y = torch.from_numpy(target).float()
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(E)).split(64):
            paths, logits = model(E[idx], M[idx], B[idx])
            loss = wta_loss(paths, logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def predict(model, emb, motion, stats):
    paths, logits = model(torch.from_numpy(emb).float(),
                          torch.from_numpy(scale_motion(motion, stats)).float(),
                          torch.from_numpy(ctrv(motion)))
    return paths.numpy(), torch.softmax(logits, 1).numpy()


def fit_ensemble(emb, motion, target, stats):
    return [fit(emb, motion, target, stats, seed=s) for s in SEEDS]


def predict_ensemble(models, emb, motion, stats):
    """Best path = average of each head's most likely path; the 3 candidate paths and
    their probabilities come from the first head (modes are not aligned across heads)."""
    outs = [predict(m, emb, motion, stats) for m in models]
    best = np.mean([p[np.arange(len(p)), pr.argmax(1)] for p, pr in outs], axis=0)
    return best, outs[0][0], outs[0][1]


def physics_three_paths(motion, factors):
    """Three physics hypotheses: keep heading at slower / same / faster speed."""
    base = heading_constant_speed(motion)
    return np.stack([base * factors[0], base, base * factors[1]], axis=1)


def min_ade(paths, target):
    return np.linalg.norm(paths - target[:, None], axis=-1).mean(-1).min(1)


def metrics(pred, target):
    """ADE / FDE / sideways error at 3 s for single paths (N, T, 2)."""
    d = np.linalg.norm(pred - target, axis=-1)
    j = (3.0 - 0.5) / 5.5 * (T - 1)
    lo, w = int(j), j - int(j)
    lat = np.abs((pred[:, lo, 1] * (1 - w) + pred[:, lo + 1, 1] * w)
                 - (target[:, lo, 1] * (1 - w) + target[:, lo + 1, 1] * w))
    return np.array([d.mean(), d[:, -1].mean(), lat.mean()])


def mirror(motion, target):
    m = motion.copy()
    m[:, [1, 2, 4]] *= -1
    t = target.copy()
    t[..., 1] *= -1
    return m, t


def old_planner_paths(emb):
    """The epoch-5 heads applied to the same embeddings: best-probability path."""
    from safety_report import load_planner
    model, _ = load_planner()
    with torch.no_grad():
        feat = torch.from_numpy(emb).float()
        trajs = torch.stack([h(feat).view(-1, T, 2) for h in model.trajectory_heads], 1)
        best = model.mode_head(feat).argmax(1)
        return trajs[torch.arange(len(feat)), best].numpy()


class MotionPlanner:
    """Inference: epoch-5 encoder + v4 heads. predict(bev, motion) -> (best path, paths, probs)."""

    def __init__(self):
        from safety_report import load_planner
        self.encoder, self.device = load_planner()
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        self.heads = []
        for sd in ck["state_dicts"]:
            h = MotionHeads()
            h.load_state_dict(sd)
            self.heads.append(h.eval())
        self.stats = {"mean": ck["motion_mean"], "std": ck["motion_std"]}
        self.beats_baseline = bool(ck["beats_baseline"])

    @torch.no_grad()
    def predict(self, bev, motion):
        t = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        fmap = self.encoder.bev_encoder(t)
        emb = self.encoder.fc_features(F.adaptive_avg_pool2d(fmap, 1).flatten(1)).cpu().numpy()
        best, paths, probs = predict_ensemble(self.heads, emb, np.asarray([motion], np.float32), self.stats)
        return best[0], paths[0], probs[0]


# --------------------------------------------------------------------- dataset ----
def build_samples(d):
    """Rebuild inputs and denoised labels per scene from the stored per-step motion."""
    emb, emb_flip, motion, target, scene = [], [], [], [], []
    for k in np.unique(d["scene"]):
        idx = np.where(d["scene"] == k)[0]
        raw = d["motion"][idx[1:]] * DT                  # step i = scan i -> i+1
        if len(raw) < STEPS + 2:
            continue
        past, label = median_steps(raw, centered=False), median_steps(raw, centered=True)
        for j in range(2, len(idx) - STEPS):
            i = idx[j]
            if d["quality"][i] < MIN_QUALITY:
                continue
            emb.append(d["emb"][i])
            emb_flip.append(d["emb_flip"][i])
            motion.append(motion_features(past, j))
            target.append(chain(label[j:j + STEPS]))
            scene.append(k)
    return (np.array(emb), np.array(emb_flip), np.array(motion, np.float32),
            np.array(target, np.float32), np.array(scene))


def main(folds=5):
    d = np.load(OUT / "motion_dataset.npz")
    emb, emb_flip, motion, target, scene = build_samples(d)
    print(f"{len(emb)} key-frames with clean labels from {len(np.unique(scene))} scenes")

    rng = np.random.default_rng(0)
    order = rng.permutation(np.unique(scene))
    fold_of = {s: i % folds for i, s in enumerate(order)}
    fold = np.array([fold_of[s] for s in scene])
    old = old_planner_paths(emb)
    names = ["Epoch-5 planner (one scan)", "Keep heading, constant speed",
             "Constant acceleration", "Constant turn rate and speed (CTRV)", "Planner v4 (this work)"]
    sums = {n: np.zeros(3) for n in names}
    multi = {"Physics, 3 speeds (slower / same / faster)": 0.0, "Planner v4, its 3 paths": 0.0}
    for f in range(folds):
        tr, va = fold != f, fold == f
        m_flip, t_flip = mirror(motion[tr], target[tr])
        m_tr = np.concatenate([motion[tr], m_flip])
        stats = {"mean": m_tr.mean(0), "std": m_tr.std(0) + 1e-6}
        models = fit_ensemble(np.concatenate([emb[tr], emb_flip[tr]]), m_tr,
                              np.concatenate([target[tr], t_flip]), stats)
        best, paths, probs = predict_ensemble(models, emb[va], motion[va], stats)
        # the physics hypotheses get their speed factors tuned on the training scenes
        grid = [(lo, hi) for lo in (0.3, 0.5, 0.6, 0.7, 0.8) for hi in (1.2, 1.3, 1.4, 1.6)]
        factors = min(grid, key=lambda fh: min_ade(physics_three_paths(motion[tr], fh), target[tr]).mean())
        multi["Physics, 3 speeds (slower / same / faster)"] += min_ade(
            physics_three_paths(motion[va], factors), target[va]).sum()
        multi["Planner v4, its 3 paths"] += min_ade(paths, target[va]).sum()
        preds = {
            names[0]: old[va],
            names[1]: heading_constant_speed(motion[va]),
            names[2]: constant_acceleration(motion[va]),
            names[3]: ctrv(motion[va]),
            names[4]: best,
        }
        for n, p in preds.items():
            sums[n] += metrics(p, target[va]) * va.sum()
    res = {n: s / len(emb) for n, s in sums.items()}
    multi = {n: v / len(emb) for n, v in multi.items()}
    best_base = min(names[1:4], key=lambda n: res[n][0])
    v4, base = res[names[4]], res[best_base]
    table = ["| Method (5-fold, held-out scenes) | ADE | FDE | sideways error at 3 s |", "|---|---|---|---|"]
    table += [f"| {n} | {r[0]:.2f} m | {r[1]:.2f} m | {r[2]:.2f} m |" for n, r in res.items()]
    text = "\n".join([
        "# Planner v4: cross-validated on held-out scenes",
        "",
        f"{len(emb)} key-frames with clean labels from {len(np.unique(scene))} scenes, 5 folds split "
        "by scene. Labels: the car's real path over 5.5 s, recovered from the LiDAR "
        "(odometry_labels.py), each 0.5 s step median-filtered.",
        "",
        *table,
        "",
        f"Against the strongest baseline ({best_base}): ADE {100 * (v4[0] - base[0]) / base[0]:+.0f}%, "
        f"FDE {100 * (v4[1] - base[1]) / base[1]:+.0f}%, sideways {100 * (v4[2] - base[2]) / base[2]:+.0f}%. "
        f"Against the epoch-5 planner: ADE {100 * (v4[0] - res[names[0]][0]) / res[names[0]][0]:+.0f}%.",
        "",
        "| Three candidate paths (best of 3, minADE@3) | ADE |",
        "|---|---|",
        *[f"| {n} | {v:.2f} m |" for n, v in multi.items()],
    ])

    m_flip, t_flip = mirror(motion, target)
    m_all = np.concatenate([motion, m_flip])
    stats = {"mean": m_all.mean(0), "std": m_all.std(0) + 1e-6}
    final = fit_ensemble(np.concatenate([emb, emb_flip]), m_all, np.concatenate([target, t_flip]), stats)
    CKPT.parent.mkdir(exist_ok=True)
    torch.save({"state_dicts": [m.state_dict() for m in final], "motion_mean": stats["mean"],
                "motion_std": stats["std"], "cv": {n: r.tolist() for n, r in res.items()},
                "multi": multi, "beats_baseline": bool(v4[0] < base[0] and v4[1] < base[1])}, CKPT)
    (OUT / "motion_planner_eval.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
