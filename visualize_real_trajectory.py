"""
visualize_real_trajectory.py
Visualize REAL trajectories with proper curved arrows
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from pathlib import Path
import pickle
from typing import Dict, List, Optional
import json

from trajectory_model import TrajectoryPlannerModel, NUSCENES_CATEGORIES
from data_preparation_v2 import load_lidar_points, create_bev_representation


def load_model(checkpoint_path: str, device: torch.device) -> TrajectoryPlannerModel:
    """Load trained model"""
    model = TrajectoryPlannerModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?'):.4f}")
    return model


def draw_smooth_trajectory(ax, trajectory, color='red', linewidth=2, alpha=0.8, label=None, show_arrows=True):
    """Draw smooth curved trajectory with direction arrows"""
    from scipy import interpolate
    
    # Interpolate for smoother curve
    if len(trajectory) >= 4:
        t = np.linspace(0, 1, len(trajectory))
        t_smooth = np.linspace(0, 1, 100)
        
        try:
            cs_x = interpolate.CubicSpline(t, trajectory[:, 0])
            cs_y = interpolate.CubicSpline(t, trajectory[:, 1])
            x_smooth = cs_x(t_smooth)
            y_smooth = cs_y(t_smooth)
            trajectory_smooth = np.column_stack([x_smooth, y_smooth])
        except:
            trajectory_smooth = trajectory
    else:
        trajectory_smooth = trajectory
    
    # Draw the path
    ax.plot(trajectory_smooth[:, 0], trajectory_smooth[:, 1], 
            color=color, linewidth=linewidth, alpha=alpha, label=label)
    
    # Add direction arrows along the path
    if show_arrows:
        arrow_indices = np.linspace(0, len(trajectory_smooth)-2, min(8, len(trajectory_smooth)-1)).astype(int)
        for i in arrow_indices:
            dx = trajectory_smooth[i+1, 0] - trajectory_smooth[i, 0]
            dy = trajectory_smooth[i+1, 1] - trajectory_smooth[i, 1]
            if np.sqrt(dx**2 + dy**2) > 0.1:
                ax.annotate('', 
                    xy=(trajectory_smooth[i+1, 0], trajectory_smooth[i+1, 1]),
                    xytext=(trajectory_smooth[i, 0], trajectory_smooth[i, 1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=linewidth*0.8, alpha=alpha)
                )
    
    # Mark start and end
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=color, markersize=8, alpha=alpha)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=color, markersize=10, alpha=alpha)


def draw_vehicle(ax, x, y, yaw=0, width=2, length=4, color='blue', alpha=0.7):
    """Draw a vehicle as a rectangle"""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Affine2D
    
    # Create rectangle centered at origin
    rect = Rectangle((-length/2, -width/2), length, width, 
                     linewidth=1, edgecolor='black', facecolor=color, alpha=alpha)
    
    # Create transformation: rotate then translate
    t = Affine2D().rotate(yaw).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


def visualize_comparison(
    model: TrajectoryPlannerModel,
    sample: Dict,
    nuscenes_root: Path,
    device: torch.device,
    output_path: Optional[str] = None
):
    """
    Visualize prediction vs ground truth with proper curved trajectories
    """
    # Load BEV
    lidar_path = nuscenes_root / sample['lidar_path']
    points = load_lidar_points(str(lidar_path))
    bev = create_bev_representation(points)
    
    # Get prediction
    bev_tensor = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        outputs = model(bev_tensor)
        all_trajs = outputs['trajectories'][0].cpu().numpy()
        mode_probs = outputs['mode_probs'][0].cpu().numpy()
        best_traj, confidence = model.predict_best_trajectory(bev_tensor)
        best_traj = best_traj[0].cpu().numpy()
    
    # Ground truth
    gt_traj = sample['gt_trajectory']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x_range = (-50, 50)
    y_range = (-50, 50)
    
    # Plot 1: BEV with all trajectories
    ax1 = axes[0]
    ax1.imshow(bev[:, :, 2], cmap='gray', origin='lower', 
               extent=[x_range[0], x_range[1], y_range[0], y_range[1]], alpha=0.8)
    
    # Draw multi-modal predictions
    mode_colors = ['#2ecc71', '#3498db', '#f39c12']  # green, blue, orange
    for m, (traj, prob) in enumerate(zip(all_trajs, mode_probs)):
        alpha = 0.3 + 0.5 * prob
        draw_smooth_trajectory(ax1, traj, color=mode_colors[m], 
                              linewidth=2, alpha=alpha, 
                              label=f'Mode {m+1} ({prob:.2f})', show_arrows=False)
    
    # Draw best prediction with arrows
    draw_smooth_trajectory(ax1, best_traj, color='#e74c3c', linewidth=3, alpha=0.9,
                          label='Best Pred', show_arrows=True)
    
    # Draw ground truth
    draw_smooth_trajectory(ax1, gt_traj, color='#9b59b6', linewidth=3, alpha=0.9,
                          label='Ground Truth', show_arrows=True)
    
    # Draw ego vehicle
    draw_vehicle(ax1, 0, 0, yaw=0, width=2, length=4.5, color='#1abc9c', alpha=0.9)
    
    # Draw obstacles from annotations
    for ann in sample.get('annotations', []):
        x, y, z = ann.get('translation', [0, 0, 0])
        w, l, h = ann.get('size', [2, 2, 2])
        cat = ann.get('category', '')
        
        if 'vehicle' in cat.lower():
            color = '#e74c3c'
        elif 'human' in cat.lower():
            color = '#3498db'
        else:
            color = '#95a5a6'
        
        draw_vehicle(ax1, x, y, width=w, length=l, color=color, alpha=0.5)
    
    ax1.set_title('BEV: Prediction vs Ground Truth', fontsize=12)
    ax1.set_xlabel('X (m) - Forward')
    ax1.set_ylabel('Y (m) - Left')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Trajectory error analysis
    ax2 = axes[1]
    
    # Compute per-waypoint error
    errors = np.sqrt(np.sum((best_traj - gt_traj) ** 2, axis=1))
    waypoint_idx = np.arange(len(errors))
    
    colors = plt.cm.RdYlGn_r(errors / (errors.max() + 1e-6))
    ax2.bar(waypoint_idx, errors, color=colors, edgecolor='black', alpha=0.8)
    
    ade = np.mean(errors)
    fde = errors[-1]
    
    ax2.axhline(y=ade, color='blue', linestyle='--', label=f'ADE: {ade:.2f}m')
    ax2.set_title(f'Per-Waypoint Error (FDE: {fde:.2f}m)', fontsize=12)
    ax2.set_xlabel('Waypoint Index')
    ax2.set_ylabel('Error (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mode probability distribution
    ax3 = axes[2]
    
    mode_names = [f'Mode {i+1}' for i in range(len(mode_probs))]
    bars = ax3.bar(mode_names, mode_probs, color=mode_colors, edgecolor='black', alpha=0.8)
    
    for bar, prob in zip(bars, mode_probs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.2f}', ha='center', fontsize=10)
    
    ax3.set_title('Mode Selection Probability', fontsize=12)
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def visualize_trajectory_sequence(
    model: TrajectoryPlannerModel,
    data: List[Dict],
    nuscenes_root: Path,
    device: torch.device,
    output_dir: str,
    num_samples: int = 10
):
    """Visualize multiple samples"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nVisualizing {num_samples} samples...")
    
    ade_list = []
    fde_list = []
    
    for i, sample in enumerate(data[:num_samples]):
        try:
            lidar_path = nuscenes_root / sample['lidar_path']
            if not lidar_path.exists():
                continue
            
            output_path = output_dir / f"trajectory_{i:03d}_{sample['sample_token'][:8]}.png"
            
            fig = visualize_comparison(
                model=model,
                sample=sample,
                nuscenes_root=nuscenes_root,
                device=device,
                output_path=str(output_path)
            )
            
            # Calculate metrics
            points = load_lidar_points(str(lidar_path))
            bev = create_bev_representation(points)
            bev_tensor = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                best_traj, _ = model.predict_best_trajectory(bev_tensor)
                best_traj = best_traj[0].cpu().numpy()
            
            gt_traj = sample['gt_trajectory']
            errors = np.sqrt(np.sum((best_traj - gt_traj) ** 2, axis=1))
            ade_list.append(np.mean(errors))
            fde_list.append(errors[-1])
            
            plt.close(fig)
            print(f"  [{i+1}/{num_samples}] ADE={ade_list[-1]:.2f}m, FDE={fde_list[-1]:.2f}m")
            
        except Exception as e:
            print(f"  [{i+1}/{num_samples}] Error: {e}")
    
    # Print summary
    if ade_list:
        print(f"\nMetrics Summary:")
        print(f"  Mean ADE: {np.mean(ade_list):.2f}m (+/- {np.std(ade_list):.2f})")
        print(f"  Mean FDE: {np.mean(fde_list):.2f}m (+/- {np.std(fde_list):.2f})")
    
    print(f"\nVisualizations saved to {output_dir}")


if __name__ == '__main__':
    MODEL_PATH = './checkpoint_v3/best_trajectory_model.pth'
    DATA_PKL = '/home/ec2-user/SageMaker/nuscenes/trajectory_output/training_data_real.pkl'
    NUSCENES_ROOT = '/home/ec2-user/SageMaker/nuscenes'
    OUTPUT_DIR = './visualizations'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(MODEL_PATH, device)
    
    with open(DATA_PKL, 'rb') as f:
        data = pickle.load(f)
    
    visualize_trajectory_sequence(
        model=model,
        data=data,
        nuscenes_root=Path(NUSCENES_ROOT),
        device=device,
        output_dir=OUTPUT_DIR,
        num_samples=20
    )
