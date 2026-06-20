"""
visualize_trajectory.py
Improved visualization with ground truth comparison, lanes, and curved trajectories

Run: python scripts/visualize_trajectory.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.collections import LineCollection
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import CubicSpline
import os

from trajectory_model import TrajectoryPlannerModel, NUSCENES_CATEGORIES, NUM_CLASSES
from data_preparation import load_lidar_points, create_bev_representation


# Colors for different categories
CATEGORY_COLORS = {
    0: [0.3, 0.3, 0.3],    # background - dark gray
    1: [1.0, 0.2, 0.2],    # car - red
    2: [1.0, 0.6, 0.0],    # truck - orange
    3: [1.0, 0.9, 0.0],    # bus - yellow
    4: [0.0, 0.8, 0.0],    # motorcycle - green
    5: [0.0, 0.9, 0.9],    # bicycle - cyan
    6: [0.2, 0.4, 1.0],    # pedestrian - blue
    7: [0.8, 0.0, 0.8],    # movable_object - magenta
}

# Trajectory mode colors
MODE_COLORS = ['#00FF00', '#00BFFF', '#FF8C00']  # Green, DeepSkyBlue, DarkOrange


def smooth_trajectory(points: np.ndarray, num_points: int = 50) -> np.ndarray:
    """Smooth trajectory using cubic spline interpolation"""
    if len(points) < 3:
        return points
    
    try:
        t = np.arange(len(points))
        t_new = np.linspace(0, len(points) - 1, num_points)
        
        cs_x = CubicSpline(t, points[:, 0])
        cs_y = CubicSpline(t, points[:, 1])
        
        return np.column_stack([cs_x(t_new), cs_y(t_new)])
    except:
        return points


def draw_vehicle(ax, x, y, heading, length=4.5, width=2.0, color='blue', alpha=0.8, label=None):
    """Draw a vehicle as a rectangle with heading indicator"""
    from matplotlib.patches import Rectangle, FancyArrow
    import matplotlib.transforms as transforms
    
    # Create rectangle centered at origin
    rect = Rectangle((-length/2, -width/2), length, width, 
                     facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5)
    
    # Create rotation + translation transform
    t = transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    
    # Add heading arrow
    arrow_len = length * 0.6
    dx = arrow_len * np.cos(heading)
    dy = arrow_len * np.sin(heading)
    ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.3, fc='white', ec='black', linewidth=1)
    
    if label:
        ax.annotate(label, (x, y), fontsize=8, ha='center', va='bottom', 
                   xytext=(0, width/2 + 1), textcoords='offset points')


def draw_curved_arrow(ax, trajectory, color='red', linewidth=3, alpha=0.8, arrow_spacing=5):
    """Draw trajectory as curved line with directional arrows"""
    if len(trajectory) < 2:
        return
    
    # Smooth the trajectory
    smooth_traj = smooth_trajectory(trajectory, num_points=100)
    
    # Draw the main trajectory line with gradient
    points = smooth_traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create color gradient from start to end
    colors = np.zeros((len(segments), 4))
    for i in range(len(segments)):
        t = i / len(segments)
        colors[i] = [*plt.cm.colors.to_rgb(color), alpha * (0.4 + 0.6 * t)]
    
    lc = LineCollection(segments, colors=colors, linewidth=linewidth)
    ax.add_collection(lc)
    
    # Add direction arrows at intervals
    for i in range(0, len(smooth_traj) - 1, arrow_spacing):
        dx = smooth_traj[i + 1, 0] - smooth_traj[i, 0]
        dy = smooth_traj[i + 1, 1] - smooth_traj[i, 1]
        
        if np.sqrt(dx**2 + dy**2) > 0.1:
            ax.annotate('', xy=(smooth_traj[i + 1, 0], smooth_traj[i + 1, 1]),
                       xytext=(smooth_traj[i, 0], smooth_traj[i, 1]),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=1.5))


def load_ground_truth_trajectory(trajectory_sequences_path: str, sample_token: str) -> Optional[np.ndarray]:
    """Load ground truth ego trajectory from trajectory sequences"""
    try:
        with open(trajectory_sequences_path, 'rb') as f:
            sequences = pickle.load(f)
        
        # Find sequence containing this sample
        for seq in sequences:
            for i, frame in enumerate(seq['frames']):
                if frame.get('sample_token') == sample_token:
                    # Extract future trajectory from ego_pose
                    future_poses = []
                    for j in range(i, min(i + 12, len(seq['frames']))):
                        pose = seq['frames'][j].get('ego_pose', {})
                        if 'translation' in pose:
                            future_poses.append(pose['translation'][:2])
                    
                    if len(future_poses) >= 2:
                        # Convert to relative coordinates
                        origin = np.array(future_poses[0])
                        relative_poses = np.array(future_poses) - origin
                        return relative_poses
        return None
    except Exception as e:
        print(f"Could not load GT trajectory: {e}")
        return None


def load_model(checkpoint_path: str, device: torch.device) -> TrajectoryPlannerModel:
    """Load trained model from checkpoint"""
    model = TrajectoryPlannerModel(num_classes=NUM_CLASSES).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint.get('val_loss'):.4f}")
    
    return model


def visualize_detailed(
    model: TrajectoryPlannerModel,
    bev: np.ndarray,
    annotations: List[Dict],
    device: torch.device,
    gt_trajectory: Optional[np.ndarray] = None,
    sample_token: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Detailed visualization with GT comparison and obstacle context
    """
    # Prepare input
    bev_tensor = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(bev_tensor)
        best_traj, confidence = model.predict_best_trajectory(bev_tensor)
        
        all_trajs = outputs['trajectories'][0].cpu().numpy()
        mode_probs = outputs['mode_probs'][0].cpu().numpy()
        seg_pred = outputs['segmentation'][0].argmax(dim=0).cpu().numpy()
    
    best_traj = best_traj[0].cpu().numpy()
    confidence = confidence[0].cpu().numpy()
    
    # Create figure with dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12))
    
    # Grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)
    
    x_range = (-50, 50)
    y_range = (-50, 50)
    
    # ============ Plot 1: Main BEV with All Trajectories ============
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Show BEV with enhanced contrast
    bev_display = np.clip(bev[:, :, 2] * 2, 0, 1)
    ax1.imshow(bev_display, cmap='gray', origin='lower', 
               extent=[x_range[0], x_range[1], y_range[0], y_range[1]], alpha=0.7)
    
    # Draw lane boundaries (simulated based on road structure)
    for lane_y in [-3.5, 0, 3.5]:
        ax1.axhline(y=lane_y, color='white', linestyle='--', alpha=0.3, linewidth=1)
    
    # Draw all trajectory modes
    for m, (traj, prob) in enumerate(zip(all_trajs, mode_probs)):
        if prob > 0.1:  # Only show meaningful modes
            draw_curved_arrow(ax1, traj, color=MODE_COLORS[m], 
                            linewidth=2 + prob * 3, alpha=0.4 + 0.5 * prob)
            # Label endpoint
            ax1.scatter(traj[-1, 0], traj[-1, 1], c=MODE_COLORS[m], s=100, 
                       marker='o', edgecolors='white', linewidth=1.5, zorder=5)
            ax1.annotate(f'{prob:.0%}', (traj[-1, 0], traj[-1, 1]), 
                        fontsize=9, color='white', ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points')
    
    # Draw ground truth trajectory if available
    if gt_trajectory is not None and len(gt_trajectory) > 1:
        draw_curved_arrow(ax1, gt_trajectory, color='cyan', linewidth=4, alpha=0.9)
        ax1.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='cyan', s=150, 
                   marker='*', edgecolors='white', linewidth=2, zorder=6, label='GT End')
    
    # Draw annotations (obstacles)
    for ann in annotations:
        trans = ann.get('translation', [0, 0, 0])
        size = ann.get('size', [2, 2, 2])
        category = ann.get('category', '')
        
        x, y = trans[0], trans[1]
        w, l = size[0], size[1]
        
        # Skip if outside view
        if abs(x) > 45 or abs(y) > 45:
            continue
        
        # Determine color
        color = 'gray'
        if 'car' in category:
            color = '#FF4444'
        elif 'truck' in category or 'bus' in category:
            color = '#FFA500'
        elif 'pedestrian' in category:
            color = '#4488FF'
        elif 'bicycle' in category or 'motorcycle' in category:
            color = '#00FF88'
        
        rect = plt.Rectangle((x - w/2, y - l/2), w, l,
                             facecolor=color, edgecolor='white', 
                             alpha=0.6, linewidth=1)
        ax1.add_patch(rect)
    
    # Draw ego vehicle
    draw_vehicle(ax1, 0, 0, np.pi/2, color='#00AA00', alpha=0.9, label='EGO')
    
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-10, 50)
    ax1.set_xlabel('X (meters)', fontsize=10)
    ax1.set_ylabel('Y (meters)', fontsize=10)
    ax1.set_title('Multi-Modal Trajectory Prediction', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=MODE_COLORS[0], linewidth=3, label=f'Mode 1: {mode_probs[0]:.0%}'),
        plt.Line2D([0], [0], color=MODE_COLORS[1], linewidth=3, label=f'Mode 2: {mode_probs[1]:.0%}'),
        plt.Line2D([0], [0], color=MODE_COLORS[2], linewidth=3, label=f'Mode 3: {mode_probs[2]:.0%}'),
    ]
    if gt_trajectory is not None:
        legend_elements.append(plt.Line2D([0], [0], color='cyan', linewidth=3, label='Ground Truth'))
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # ============ Plot 2: Zoomed Best Trajectory ============
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.imshow(bev_display, cmap='gray', origin='lower',
               extent=[x_range[0], x_range[1], y_range[0], y_range[1]], alpha=0.5)
    
    # Draw best trajectory with waypoint markers
    best_mode_idx = np.argmax(mode_probs)
    best_mode_traj = all_trajs[best_mode_idx]
    
    # Smooth and draw
    smooth_best = smooth_trajectory(best_mode_traj, num_points=50)
    ax2.plot(smooth_best[:, 0], smooth_best[:, 1], 'g-', linewidth=4, alpha=0.8)
    
    # Draw waypoints with confidence coloring
    for i, (wp, conf) in enumerate(zip(best_traj, confidence)):
        color = plt.cm.RdYlGn(conf)
        ax2.scatter(wp[0], wp[1], c=[color], s=80, edgecolors='white', linewidth=1.5, zorder=5)
        
        # Add directional arrow between waypoints
        if i < len(best_traj) - 1:
            next_wp = best_traj[i + 1]
            dx = next_wp[0] - wp[0]
            dy = next_wp[1] - wp[1]
            ax2.arrow(wp[0], wp[1], dx * 0.7, dy * 0.7, 
                     head_width=1.0, head_length=0.5, fc='lime', ec='white', linewidth=0.5)
    
    # Compare with GT
    if gt_trajectory is not None and len(gt_trajectory) > 1:
        ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'c--', linewidth=2, alpha=0.8, label='GT')
        
        # Calculate error
        min_len = min(len(best_traj), len(gt_trajectory))
        if min_len > 0:
            ade = np.mean(np.linalg.norm(best_traj[:min_len] - gt_trajectory[:min_len], axis=1))
            fde = np.linalg.norm(best_traj[-1] - gt_trajectory[-1])
            ax2.set_title(f'Best Trajectory (ADE: {ade:.2f}m, FDE: {fde:.2f}m)', fontsize=12, fontweight='bold')
        else:
            ax2.set_title('Best Trajectory with Waypoints', fontsize=12, fontweight='bold')
    else:
        ax2.set_title('Best Trajectory with Waypoints', fontsize=12, fontweight='bold')
    
    draw_vehicle(ax2, 0, 0, np.pi/2, color='#00AA00', alpha=0.9)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-5, 35)
    ax2.set_xlabel('X (meters)', fontsize=10)
    ax2.set_ylabel('Y (meters)', fontsize=10)
    ax2.grid(True, alpha=0.2)
    
    # ============ Plot 3: Semantic Segmentation ============
    ax3 = fig.add_subplot(gs[0, 2])
    
    seg_colored = np.zeros((*seg_pred.shape, 3))
    for cid, color in CATEGORY_COLORS.items():
        seg_colored[seg_pred == cid] = color
    
    ax3.imshow(seg_colored, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    ax3.set_title('Semantic Segmentation', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (meters)', fontsize=10)
    ax3.set_ylabel('Y (meters)', fontsize=10)
    
    patches = [mpatches.Patch(color=CATEGORY_COLORS[i], label=NUSCENES_CATEGORIES[i]) 
               for i in range(NUM_CLASSES) if i < len(NUSCENES_CATEGORIES)]
    ax3.legend(handles=patches, loc='upper right', fontsize=7)
    
    # ============ Plot 4: Mode Probabilities ============
    ax4 = fig.add_subplot(gs[1, 0])
    
    bars = ax4.bar(range(len(mode_probs)), mode_probs, color=MODE_COLORS[:len(mode_probs)], 
                   edgecolor='white', linewidth=2)
    ax4.set_xlabel('Trajectory Mode', fontsize=10)
    ax4.set_ylabel('Probability', fontsize=10)
    ax4.set_title('Mode Selection Confidence', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.set_xticks(range(len(mode_probs)))
    ax4.set_xticklabels([f'Mode {i+1}' for i in range(len(mode_probs))])
    
    for i, (bar, prob) in enumerate(zip(bars, mode_probs)):
        ax4.annotate(f'{prob:.1%}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ============ Plot 5: Waypoint Confidence ============
    ax5 = fig.add_subplot(gs[1, 1])
    
    waypoint_ids = np.arange(len(confidence))
    colors = [plt.cm.RdYlGn(c) for c in confidence]
    bars = ax5.bar(waypoint_ids, confidence, color=colors, edgecolor='white', linewidth=1)
    
    ax5.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='Threshold')
    ax5.set_xlabel('Waypoint Index (Future Timesteps)', fontsize=10)
    ax5.set_ylabel('Confidence', fontsize=10)
    ax5.set_title('Waypoint Prediction Confidence', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.legend(loc='upper right')
    
    # ============ Plot 6: Trajectory Comparison Top-Down ============
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Draw all modes
    for m, (traj, prob) in enumerate(zip(all_trajs, mode_probs)):
        smooth = smooth_trajectory(traj, num_points=30)
        ax6.plot(smooth[:, 0], smooth[:, 1], '-', color=MODE_COLORS[m], 
                linewidth=1 + prob * 4, alpha=0.3 + prob * 0.6, label=f'Pred {m+1}')
        ax6.scatter(traj[-1, 0], traj[-1, 1], c=MODE_COLORS[m], s=50 + prob * 100, 
                   marker='o', edgecolors='white', zorder=5)
    
    if gt_trajectory is not None and len(gt_trajectory) > 1:
        ax6.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'c-', linewidth=3, 
                alpha=0.9, label='Ground Truth')
        ax6.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='cyan', s=150, 
                   marker='*', edgecolors='white', linewidth=2, zorder=6)
    
    ax6.scatter(0, 0, c='lime', s=200, marker='s', edgecolors='white', linewidth=2, label='Ego', zorder=7)
    ax6.set_xlabel('X (meters)', fontsize=10)
    ax6.set_ylabel('Y (meters)', fontsize=10)
    ax6.set_title('All Trajectories Comparison', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    ax6.set_xlim(-25, 25)
    ax6.set_ylim(-5, 45)
    
    # Add sample token as suptitle
    fig.suptitle(f'Sample: {sample_token}', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Saved visualization to {output_path}")
    
    return fig


def visualize_from_dataset(
    model_path: str,
    data_pkl: str,
    trajectory_sequences_path: str,
    nuscenes_root: str,
    output_dir: str,
    num_samples: int = 10,
):
    """Visualize predictions on dataset samples with GT comparison"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Load data
    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples from {data_pkl}")
    
    nuscenes_root = Path(nuscenes_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to samples with existing LiDAR files
    valid_samples = []
    for sample in data:
        lidar_path = nuscenes_root / sample['lidar_path']
        if lidar_path.exists():
            valid_samples.append(sample)
    
    print(f"Found {len(valid_samples)} samples with existing LiDAR files")
    print(f"\nVisualizing {min(num_samples, len(valid_samples))} samples...")
    
    success_count = 0
    for i, sample in enumerate(valid_samples[:num_samples]):
        try:
            lidar_path = nuscenes_root / sample['lidar_path']
            points = load_lidar_points(str(lidar_path))
            
            bev, _ = create_bev_representation(
                points,
                annotations=sample.get('annotations', []),
            )
            
            # Try to load GT trajectory
            gt_traj = load_ground_truth_trajectory(trajectory_sequences_path, sample['sample_token'])
            
            output_path = output_dir / f"viz_{i:03d}_{sample['sample_token'][:8]}.png"
            
            fig = visualize_detailed(
                model=model,
                bev=bev,
                annotations=sample.get('annotations', []),
                device=device,
                gt_trajectory=gt_traj,
                sample_token=sample['sample_token'],
                output_path=str(output_path),
            )
            plt.close(fig)
            
            gt_status = "with GT" if gt_traj is not None else "no GT"
            print(f"  [{i+1}/{num_samples}] {sample['sample_token'][:8]} - "
                  f"{len(sample.get('annotations', []))} annotations ({gt_status})")
            success_count += 1
            
        except Exception as e:
            print(f"  [{i+1}/{num_samples}] Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Successfully visualized {success_count}/{num_samples} samples")
    print(f"Visualizations saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    MODEL_PATH = './checkpoints/best_trajectory_model.pth'
    DATA_PKL = '/home/ec2-user/SageMaker/nuscenes/trajectory_output/training_data.pkl'
    TRAJECTORY_SEQ_PATH = '/home/ec2-user/SageMaker/nuscenes/trajectory_output/trajectory_sequences.pkl'
    NUSCENES_ROOT = '/home/ec2-user/SageMaker/nuscenes'
    OUTPUT_DIR = './visualizations'
    
    visualize_from_dataset(
        model_path=MODEL_PATH,
        data_pkl=DATA_PKL,
        trajectory_sequences_path=TRAJECTORY_SEQ_PATH,
        nuscenes_root=NUSCENES_ROOT,
        output_dir=OUTPUT_DIR,
        num_samples=15,
    )
