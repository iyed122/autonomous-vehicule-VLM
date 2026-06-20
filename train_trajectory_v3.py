"""
train_trajectory_v3.py
Improved training script with:
- Data augmentation (rotation, flip) to learn turns
- Diversity loss to separate modes (fixes mode collapse)
- Huber loss for stability
- Learning rate warmup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Import from other files
from trajectory_model import TrajectoryPlannerModel
from data_preparation_v2 import load_lidar_points, create_bev_representation


class NuscenesAugmentedDataset(Dataset):
    """Dataset with REAL ego trajectories + augmentation for learning turns"""

    def __init__(self, data_pkl_path, nuscenes_root, validate_files=True, augment=True):
        with open(data_pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        
        self.nuscenes_root = Path(nuscenes_root)
        self.augment = augment
        
        # Diagnostic info
        if len(all_data) > 0:
            print(f"Keys in first sample: {list(all_data[0].keys())}")
        
        if validate_files:
            print("Validating dataset samples...")
            valid_data = []
            missing_lidar = 0
            missing_gt = 0
            
            for sample in tqdm(all_data, desc="Checking files"):
                lidar_path = self.nuscenes_root / sample['lidar_path']
                if not lidar_path.exists():
                    missing_lidar += 1
                    continue
                if 'gt_trajectory' not in sample:
                    missing_gt += 1
                    continue
                valid_data.append(sample)
            
            self.data = valid_data
            print(f"Removed {missing_lidar} samples with missing LiDAR files")
            print(f"Removed {missing_gt} samples with missing gt_trajectory")
            
            if len(self.data) == 0:
                raise ValueError("No valid training samples found. Please regenerate training data.")
        else:
            self.data = all_data

        print(f"Loaded dataset with {len(self.data)} valid samples")
        print(f"Augmentation: {'ENABLED' if augment else 'DISABLED'}")
        
        if self.data:
            trajs = [s['gt_trajectory'] for s in self.data]
            all_trajs = np.array(trajs)
            print(f"Trajectory stats:")
            print(f"  X range: [{all_trajs[:,:,0].min():.2f}, {all_trajs[:,:,0].max():.2f}]")
            print(f"  Y range: [{all_trajs[:,:,1].min():.2f}, {all_trajs[:,:,1].max():.2f}]")

    def __len__(self):
        return len(self.data)

    def _augment_sample(self, bev, trajectory):
        """
        KEY FIX: Random rotation and flip to learn turns/maneuvers.
        Without this, model only learns "go straight" since most driving is straight.
        """
        # Random rotation (-25 to 25 degrees)
        if np.random.random() > 0.3:
            angle_deg = np.random.uniform(-25, 25)
            angle_rad = np.radians(angle_deg)
            
            # Rotate trajectory points
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            trajectory = trajectory @ rot_matrix.T
            
            # Rotate BEV image to match
            bev = TF.rotate(bev, angle_deg, fill=0)
        
        # Random horizontal flip (mirror left turns to right turns and vice versa)
        if np.random.random() > 0.5:
            trajectory[:, 1] *= -1  # Flip Y axis
            bev = TF.hflip(bev)
        
        return bev, trajectory

    def __getitem__(self, idx):
        sample = self.data[idx]

        try:
            lidar_path = self.nuscenes_root / sample['lidar_path']
            points = load_lidar_points(str(lidar_path))
            
            if points.size == 0:
                raise ValueError("Empty point cloud")
            
            bev = create_bev_representation(points)
            bev = torch.from_numpy(bev).permute(2, 0, 1).float()
            gt_trajectory = sample['gt_trajectory'].copy()

            # Apply augmentation during training
            if self.augment:
                bev, gt_trajectory = self._augment_sample(bev, gt_trajectory)
            
            gt_trajectory = torch.from_numpy(gt_trajectory).float()

            lane_mask = self._create_lane_mask()
            obstacle_mask = self._create_obstacle_mask(sample)

            return {
                'bev': bev,
                'gt_trajectory': gt_trajectory,
                'lane_mask': lane_mask,
                'obstacle_mask': obstacle_mask,
                'sample_token': sample['sample_token']
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return {
                'bev': torch.zeros(3, 512, 512),
                'gt_trajectory': torch.zeros(20, 2),
                'lane_mask': torch.ones(512, 512),
                'obstacle_mask': torch.zeros(512, 512),
                'sample_token': 'error'
            }

    def _create_lane_mask(self):
        mask = torch.zeros(512, 512)
        mask[256:512, 200:312] = 1.0
        mask[200:256, 220:292] = 1.0
        return mask

    def _create_obstacle_mask(self, sample):
        mask = torch.zeros(512, 512)
        grid_size = 512
        world_range = 100

        for ann in sample.get('annotations', []):
            category = ann.get('category', '')
            if any(prefix in category.lower() for prefix in ['vehicle', 'human', 'bicycle', 'motorcycle']):
                translation = ann.get('translation', [0, 0, 0])
                size = ann.get('size', [2, 2, 2])
                x, y = translation[0], translation[1]
                w, l = size[0], size[1]

                x_grid = int((x + 50) / world_range * grid_size)
                y_grid = int((y + 50) / world_range * grid_size)
                w_grid = max(int(w / world_range * grid_size), 3)
                l_grid = max(int(l / world_range * grid_size), 3)

                if 0 <= x_grid < grid_size and 0 <= y_grid < grid_size:
                    x_min = max(0, x_grid - l_grid // 2)
                    x_max = min(grid_size, x_grid + l_grid // 2)
                    y_min = max(0, y_grid - w_grid // 2)
                    y_max = min(grid_size, y_grid + w_grid // 2)
                    mask[x_min:x_max, y_min:y_max] = 1.0

        return mask


def custom_collate_fn(batch):
    valid_batch = [b for b in batch if b['sample_token'] != 'error']
    
    if len(valid_batch) == 0:
        return {
            'bev': torch.zeros(1, 3, 512, 512),
            'gt_trajectory': torch.zeros(1, 20, 2),
            'lane_mask': torch.ones(1, 512, 512),
            'obstacle_mask': torch.zeros(1, 512, 512),
            'sample_token': ['dummy']
        }
    
    return {
        'bev': torch.stack([b['bev'] for b in valid_batch]),
        'gt_trajectory': torch.stack([b['gt_trajectory'] for b in valid_batch]),
        'lane_mask': torch.stack([b['lane_mask'] for b in valid_batch]),
        'obstacle_mask': torch.stack([b['obstacle_mask'] for b in valid_batch]),
        'sample_token': [b['sample_token'] for b in valid_batch]
    }


class ImprovedTrajectoryLoss(nn.Module):
    """
    KEY FIX: Diversity loss pushes modes apart to prevent mode collapse.
    Also uses Huber loss (more robust than MSE).
    """
    
    def __init__(self, lambda_traj=1.0, lambda_mode=0.5, lambda_diversity=0.15, 
                 lambda_lane=0.2, lambda_collision=0.5):
        super().__init__()
        self.lambda_traj = lambda_traj
        self.lambda_mode = lambda_mode
        self.lambda_diversity = lambda_diversity
        self.lambda_lane = lambda_lane
        self.lambda_collision = lambda_collision
        self.huber = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, outputs, gt_traj, lane_mask, obstacle_mask):
        pred_trajs = outputs['trajectories']  # (B, K, T, 2)
        mode_probs = outputs['mode_probs']    # (B, K)
        
        batch_size, num_modes, num_waypoints, _ = pred_trajs.shape
        
        # Compute per-mode errors using Huber loss
        gt_expanded = gt_traj.unsqueeze(1).expand(-1, num_modes, -1, -1)
        per_point_error = self.huber(pred_trajs, gt_expanded).sum(dim=-1)
        traj_errors = per_point_error.mean(dim=-1)  # (B, K)
        
        # Winner-takes-all
        best_mode = traj_errors.argmin(dim=1)
        
        # Regression loss on best mode
        regression_loss = traj_errors[torch.arange(batch_size), best_mode].mean()
        
        # Mode classification with label smoothing
        smooth_targets = torch.zeros_like(mode_probs)
        smooth_targets.scatter_(1, best_mode.unsqueeze(1), 0.85)
        smooth_targets += 0.15 / num_modes
        mode_loss = F.cross_entropy(mode_probs, smooth_targets)
        
        # DIVERSITY LOSS - encourage different predictions per mode
        final_positions = pred_trajs[:, :, -1, :]  # (B, K, 2)
        diversity_loss = self._compute_diversity_loss(final_positions)
        
        # Lane and collision losses
        lane_loss = self._lane_loss(pred_trajs, best_mode, lane_mask)
        collision_loss = self._collision_loss(pred_trajs, best_mode, obstacle_mask)
        
        total_loss = (
            self.lambda_traj * regression_loss +
            self.lambda_mode * mode_loss +
            self.lambda_diversity * diversity_loss +
            self.lambda_lane * lane_loss +
            self.lambda_collision * collision_loss
        )
        
        return total_loss, {
            'regression': regression_loss.item(),
            'mode': mode_loss.item(),
            'diversity': diversity_loss.item(),
            'lane': lane_loss.item(),
            'collision': collision_loss.item(),
        }
    
    def _compute_diversity_loss(self, final_positions):
        """Encourage modes to predict different endpoints (target ~8-10m apart)"""
        batch_size, num_modes, _ = final_positions.shape
        
        total_dist = 0.0
        count = 0
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                dist = torch.norm(final_positions[:, i] - final_positions[:, j], dim=-1)
                total_dist += dist.mean()
                count += 1
        
        avg_dist = total_dist / max(count, 1)
        # Want high distance, return penalty when distance is low
        return torch.clamp(10.0 - avg_dist, min=0.0)
    
    def _lane_loss(self, pred_trajs, best_mode, lane_mask):
        batch_size = pred_trajs.size(0)
        grid_size = lane_mask.shape[-1]
        violations = 0.0
        count = 0
        
        for b in range(batch_size):
            traj = pred_trajs[b, best_mode[b]]
            for i in range(traj.size(0)):
                x, y = traj[i]
                x_idx = int(torch.clamp((x + 50) / 100 * grid_size, 0, grid_size-1))
                y_idx = int(torch.clamp((y + 50) / 100 * grid_size, 0, grid_size-1))
                violations += (1.0 - lane_mask[b, x_idx, y_idx])
                count += 1
        
        return violations / (count + 1e-6)
    
    def _collision_loss(self, pred_trajs, best_mode, obstacle_mask):
        batch_size = pred_trajs.size(0)
        grid_size = obstacle_mask.shape[-1]
        collisions = 0.0
        count = 0
        
        for b in range(batch_size):
            traj = pred_trajs[b, best_mode[b]]
            for i in range(traj.size(0)):
                x, y = traj[i]
                x_idx = int(torch.clamp((x + 50) / 100 * grid_size, 0, grid_size-1))
                y_idx = int(torch.clamp((y + 50) / 100 * grid_size, 0, grid_size-1))
                collisions += obstacle_mask[b, x_idx, y_idx]
                count += 1
        
        return collisions / (count + 1e-6)


def train_model(
    data_pkl: str,
    nuscenes_root: str,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 3e-4,
    warmup_epochs: int = 5,
    save_dir: str = './checkpoint_v3'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("Training V3 - With Augmentation + Diversity Loss")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"Warmup Epochs: {warmup_epochs}")
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset with augmentation
    print("\nLoading datasets...")
    train_dataset_full = NuscenesAugmentedDataset(data_pkl, nuscenes_root, augment=True)
    
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    
    # Split with fixed seed for reproducibility
    indices = list(range(len(train_dataset_full)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    
    # Validation without augmentation
    val_dataset_full = NuscenesAugmentedDataset(data_pkl, nuscenes_root, augment=False, validate_files=False)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    # Model
    model = TrajectoryPlannerModel().to(device)
    criterion = ImprovedTrajectoryLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    best_ade = float('inf')
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(epochs):
        # LR warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr * lr_scale
            current_lr = lr * lr_scale
        else:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        model.train()
        train_loss = 0.0
        train_losses = {'regression': 0, 'mode': 0, 'diversity': 0, 'lane': 0, 'collision': 0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            if batch['sample_token'] == ['dummy']:
                continue
            
            bev = batch['bev'].to(device)
            gt_traj = batch['gt_trajectory'].to(device)
            lane_mask = batch['lane_mask'].to(device)
            obstacle_mask = batch['obstacle_mask'].to(device)
            
            outputs = model(bev)
            loss, loss_dict = criterion(outputs, gt_traj, lane_mask, obstacle_mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            for k, v in loss_dict.items():
                train_losses[k] += v
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reg': f'{loss_dict["regression"]:.3f}',
                'div': f'{loss_dict["diversity"]:.3f}'
            })
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        total_ade = 0.0
        total_fde = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  "):
                if batch['sample_token'] == ['dummy']:
                    continue
                
                bev = batch['bev'].to(device)
                gt_traj = batch['gt_trajectory'].to(device)
                lane_mask = batch['lane_mask'].to(device)
                obstacle_mask = batch['obstacle_mask'].to(device)
                
                outputs = model(bev)
                loss, _ = criterion(outputs, gt_traj, lane_mask, obstacle_mask)
                
                # Calculate ADE/FDE
                pred_trajs = outputs['trajectories']
                errors = torch.norm(pred_trajs - gt_traj.unsqueeze(1), dim=-1)
                min_ade = errors.mean(dim=-1).min(dim=1)[0]
                min_fde = errors[:, :, -1].min(dim=1)[0]
                
                total_ade += min_ade.sum().item()
                total_fde += min_fde.sum().item()
                total_samples += bev.size(0)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train = train_loss / max(num_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        avg_ade = total_ade / max(total_samples, 1)
        avg_fde = total_fde / max(total_samples, 1)
        
        print(f"\nEpoch {epoch+1}/{epochs} | LR: {current_lr:.2e}")
        print(f"  Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        print(f"  Val ADE: {avg_ade:.2f}m | Val FDE: {avg_fde:.2f}m")
        print(f"  Diversity loss: {train_losses['diversity']/max(num_batches,1):.4f}")
        
        # Save best model based on ADE (more meaningful metric)
        if avg_ade < best_ade:
            best_ade = avg_ade
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val,
                'val_ade': avg_ade,
                'val_fde': avg_fde,
            }, Path(save_dir) / 'best_trajectory_model.pth')
            print(f"  [SAVED] Best model (ADE: {avg_ade:.2f}m, FDE: {avg_fde:.2f}m)")
        
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val,
            }, Path(save_dir) / f'checkpoint_epoch_{epoch+1}.pth')
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Best ADE: {best_ade:.2f}m")
    print("=" * 80)


if __name__ == '__main__':
    DATA_PKL_CHECKPOINT = '/home/ec2-user/SageMaker/nuscenes/trajectory_output/training_data_checkpoint.pkl'
    DATA_PKL_FINAL = '/home/ec2-user/SageMaker/nuscenes/trajectory_output/training_data_real.pkl'
    NUSCENES_ROOT = '/home/ec2-user/SageMaker/nuscenes'
    
    import os
    if os.path.exists(DATA_PKL_FINAL):
        data_pkl = DATA_PKL_FINAL
        print(f"Using final training data: {DATA_PKL_FINAL}")
    elif os.path.exists(DATA_PKL_CHECKPOINT):
        data_pkl = DATA_PKL_CHECKPOINT
        print(f"Using checkpoint training data: {DATA_PKL_CHECKPOINT}")
    else:
        raise FileNotFoundError(
            f"No training data found. Run data_preparation_v2.py first.\n"
            f"Expected: {DATA_PKL_FINAL} or {DATA_PKL_CHECKPOINT}"
        )
    
    train_model(
        data_pkl=data_pkl,
        nuscenes_root=NUSCENES_ROOT,
        epochs=80,
        batch_size=16,
        lr=3e-4,
        warmup_epochs=5,
        save_dir='./checkpoint_v3'
    )
