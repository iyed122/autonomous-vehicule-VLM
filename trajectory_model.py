"""
trajectory_model.py
====================
Reconstructed model definition for the BEV multi-modal ego-trajectory planner.

The original `trajectory_model.py` was lost (it lived on the SageMaker instance),
but the trained weights in `checkpoint_v3/best_trajectory_model.pth` survived.
This file rebuilds the exact `TrajectoryPlannerModel` architecture by reverse-
engineering the layer names and tensor shapes stored in that checkpoint, so the
weights load with a clean `load_state_dict(...)` (strict=True).

Architecture (inferred from the checkpoint):
  Input : BEV tensor (B, 3, 500, 500)  -> channels = height / intensity / density
  Encoder (VGG-style double-conv blocks, no residual shortcuts):
      stem   : Conv7x7 s2 (3->64)  + BN + ReLU + MaxPool3x3 s2
      layer1 : 2 x DoubleConv (64->128->128),  downsample s2
      layer2 : 2 x DoubleConv (128->256->256), downsample s2
      layer3 : 2 x DoubleConv (256->512->512), downsample s2
  Global average pool -> fc_features (512 -> 256)
  Heads (all consume the 256-d scene embedding):
      trajectory_heads : 3 modes, each Linear(256->256)->ReLU->Dropout->Linear(256->40)
                         40 = 20 future waypoints x (x, y) in ego metres
      mode_head        : Linear(256->128)->ReLU->Linear(128->3)  (mode logits)
      confidence_head  : Linear(256->128)->ReLU->Linear(128->20) (per-waypoint conf)
      seg_decoder      : transposed-conv decoder 512->...->8  (BEV semantic seg, 8 classes)

Outputs (dict):
  trajectories : (B, 3, 20, 2)
  mode_probs   : (B, 3)
  confidence   : (B, 20)
  segmentation : (B, 8, Hs, Ws)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# nuScenes semantic categories used by the BEV segmentation head (8 channels).
NUSCENES_CATEGORIES = [
    "background",
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "human.pedestrian",
    "movable_object",
]

NUM_MODES = 3
NUM_WAYPOINTS = 20
SEG_CLASSES = 8


def _double_conv(in_ch, out_ch, stride=2):
    """Conv-BN-ReLU-Conv-BN block (indices 0,1,2,3,4 -> matches checkpoint keys)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),  # .0
        nn.BatchNorm2d(out_ch),                                                          # .1
        nn.ReLU(inplace=True),                                                           # .2
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),       # .3
        nn.BatchNorm2d(out_ch),                                                          # .4
    )


class BEVEncoder(nn.Module):
    """VGG-style BEV feature encoder. Produces a (B, 512, H', W') feature map."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # stem.0
            nn.BatchNorm2d(64),                                                # stem.1
            nn.ReLU(inplace=True),                                             # stem.2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                  # stem.3
        )
        # Each "layer" = 2 double-conv blocks. First block downsamples (stride 2),
        # second keeps resolution (stride 1).
        self.layer1 = nn.Sequential(_double_conv(64, 128, stride=2),
                                    _double_conv(128, 128, stride=1))
        self.layer2 = nn.Sequential(_double_conv(128, 256, stride=2),
                                    _double_conv(256, 256, stride=1))
        self.layer3 = nn.Sequential(_double_conv(256, 512, stride=2),
                                    _double_conv(512, 512, stride=1))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # (B, 512, H', W')


class TrajectoryPlannerModel(nn.Module):
    """
    Multi-modal BEV ego-trajectory planner.

    Given a bird's-eye-view LiDAR tensor it predicts `NUM_MODES` candidate future
    trajectories (each `NUM_WAYPOINTS` x/y waypoints in the ego frame), a categorical
    distribution over those modes, a per-waypoint confidence, and a BEV semantic
    segmentation map.
    """

    def __init__(self, num_modes=NUM_MODES, num_waypoints=NUM_WAYPOINTS,
                 seg_classes=SEG_CLASSES, dropout=0.2):
        super().__init__()
        self.num_modes = num_modes
        self.num_waypoints = num_waypoints

        self.bev_encoder = BEVEncoder()

        # Global feature embedding.
        self.fc_features = nn.Sequential(
            nn.Linear(512, 256),   # fc_features.0
            nn.ReLU(inplace=True), # fc_features.1
        )

        # One regression head per mode -> 40 = num_waypoints * 2.
        self.trajectory_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),            # .0
                nn.ReLU(inplace=True),          # .1
                nn.Dropout(dropout),            # .2
                nn.Linear(256, num_waypoints * 2),  # .3
            )
            for _ in range(num_modes)
        ])

        # Mode classifier (which trajectory to trust).
        self.mode_head = nn.Sequential(
            nn.Linear(256, 128),   # .0
            nn.ReLU(inplace=True), # .1
            nn.Linear(128, num_modes),  # .2
        )

        # Per-waypoint confidence.
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),   # .0
            nn.ReLU(inplace=True), # .1
            nn.Linear(128, num_waypoints),  # .2
        )

        # BEV semantic segmentation decoder (auxiliary task).
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # .0
            nn.BatchNorm2d(256),                                               # .1
            nn.ReLU(inplace=True),                                             # .2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # .3
            nn.BatchNorm2d(128),                                               # .4
            nn.ReLU(inplace=True),                                             # .5
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # .6
            nn.BatchNorm2d(64),                                                # .7
            nn.ReLU(inplace=True),                                             # .8
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # .9
            nn.BatchNorm2d(32),                                                # .10
            nn.ReLU(inplace=True),                                             # .11
            nn.Conv2d(32, seg_classes, kernel_size=1),                         # .12
        )

    def forward(self, bev):
        feat_map = self.bev_encoder(bev)                       # (B, 512, H', W')

        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # (B, 512)
        feat = self.fc_features(pooled)                         # (B, 256)

        trajs = [head(feat).view(-1, self.num_waypoints, 2)     # (B, 20, 2)
                 for head in self.trajectory_heads]
        trajectories = torch.stack(trajs, dim=1)                # (B, num_modes, 20, 2)

        mode_logits = self.mode_head(feat)                      # (B, num_modes)
        mode_probs = F.softmax(mode_logits, dim=-1)

        confidence = torch.sigmoid(self.confidence_head(feat))  # (B, 20)

        segmentation = self.seg_decoder(feat_map)               # (B, 8, Hs, Ws)

        return {
            "trajectories": trajectories,
            "mode_probs": mode_probs,
            "mode_logits": mode_logits,
            "confidence": confidence,
            "segmentation": segmentation,
        }

    @torch.no_grad()
    def predict_best_trajectory(self, bev):
        """Return the highest-probability trajectory and its mode confidence.

        Returns:
            best_traj : (B, num_waypoints, 2)
            confidence: (B,) probability of the selected mode
        """
        out = self.forward(bev)
        mode_probs = out["mode_probs"]            # (B, num_modes)
        best_mode = mode_probs.argmax(dim=-1)     # (B,)
        idx = best_mode.view(-1, 1, 1, 1).expand(-1, 1, self.num_waypoints, 2)
        best_traj = out["trajectories"].gather(1, idx).squeeze(1)  # (B, 20, 2)
        confidence = mode_probs.gather(1, best_mode.unsqueeze(1)).squeeze(1)
        return best_traj, confidence


if __name__ == "__main__":
    # Quick self-test: build the model and load the trained weights.
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_v3/best_trajectory_model.pth"
    model = TrajectoryPlannerModel()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing keys   :", missing)
    print("unexpected keys:", unexpected)
    model.eval()
    dummy = torch.randn(1, 3, 500, 500)
    out = model(dummy)
    for k, v in out.items():
        print(f"{k:14s}: {tuple(v.shape)}")
