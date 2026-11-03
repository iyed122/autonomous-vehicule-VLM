"""
data_preparation_v2.py
CORRECTED data preparation that extracts REAL trajectories from nuScenes ego_pose
This replaces data_preparation.py with proper ground truth extraction
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import shutil


def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def quaternion_to_yaw(q: List[float]) -> float:
    """Extract yaw angle from quaternion [w, x, y, z]"""
    w, x, y, z = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def transform_to_ego_frame(global_pos: np.ndarray, ego_pos: np.ndarray, ego_rotation: List[float]) -> np.ndarray:
    """
    Transform global position to ego vehicle frame
    
    Args:
        global_pos: (N, 2) or (N, 3) global positions
        ego_pos: (3,) ego vehicle global position
        ego_rotation: quaternion [w, x, y, z]
    
    Returns:
        (N, 2) positions in ego frame (x forward, y left)
    """
    # Get yaw from quaternion
    yaw = quaternion_to_yaw(ego_rotation)
    
    # Rotation matrix (2D, rotate by -yaw to transform to ego frame)
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    rot_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])
    
    # Translate to ego origin
    if global_pos.ndim == 1:
        relative = global_pos[:2] - ego_pos[:2]
        return rot_matrix @ relative
    else:
        relative = global_pos[:, :2] - ego_pos[:2]
        return (rot_matrix @ relative.T).T


def load_nuscenes_metadata(metadata_dir: Path) -> Dict:
    """Load all nuScenes metadata JSON files"""
    metadata = {}
    
    json_files = [
        'sample.json',
        'sample_data.json', 
        'ego_pose.json',
        'sample_annotation.json',
        'instance.json',
        'category.json',
        'calibrated_sensor.json',
        'sensor.json',
        'scene.json',
    ]
    
    for filename in json_files:
        filepath = metadata_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                key = filename.replace('.json', '')
                metadata[key] = {item['token']: item for item in data}
                print(f"  Loaded {filename}: {len(data)} entries")
        else:
            print(f"  WARNING: {filename} not found!")
            metadata[filename.replace('.json', '')] = {}
    
    return metadata


def get_sensor_channel(metadata: Dict, calibrated_sensor_token: str) -> str:
    """Get sensor channel name from calibrated_sensor_token"""
    if calibrated_sensor_token not in metadata['calibrated_sensor']:
        return ''
    
    calib = metadata['calibrated_sensor'][calibrated_sensor_token]
    sensor_token = calib['sensor_token']
    
    if sensor_token not in metadata['sensor']:
        return ''
    
    return metadata['sensor'][sensor_token]['channel']


def get_future_ego_trajectory(
    metadata: Dict,
    current_sample_token: str,
    num_future_samples: int = 12,  # ~6 seconds at 2Hz
    num_waypoints: int = 20
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Extract REAL future ego trajectory from nuScenes
    
    Args:
        metadata: Loaded metadata dict
        current_sample_token: Current sample token
        num_future_samples: Number of future samples to look at
        num_waypoints: Number of waypoints to interpolate
    
    Returns:
        (trajectory, future_tokens) - trajectory is (num_waypoints, 2) in ego frame
    """
    samples = metadata['sample']
    sample_data = metadata['sample_data']
    ego_pose = metadata['ego_pose']
    
    if current_sample_token not in samples:
        return None, []
    
    current_sample = samples[current_sample_token]
    
    # Find LIDAR_TOP sample_data for current sample to get ego_pose
    current_ego_pose = None
    current_ego_pose_token = None
    
    for sd_token, sd in sample_data.items():
        if sd['sample_token'] == current_sample_token:
            channel = get_sensor_channel(metadata, sd['calibrated_sensor_token'])
            if channel == 'LIDAR_TOP' and sd['is_key_frame']:
                current_ego_pose_token = sd['ego_pose_token']
                if current_ego_pose_token in ego_pose:
                    current_ego_pose = ego_pose[current_ego_pose_token]
                break
    
    if current_ego_pose is None:
        return None, []
    
    ego_translation = np.array(current_ego_pose['translation'])
    ego_rotation = current_ego_pose['rotation']
    
    # Collect future ego positions
    future_positions = []
    future_tokens = []
    
    next_token = current_sample['next']
    while next_token and len(future_positions) < num_future_samples:
        if next_token not in samples:
            break
            
        future_sample = samples[next_token]
        future_tokens.append(next_token)
        
        # Find LIDAR_TOP sample_data for this future sample
        for sd_token, sd in sample_data.items():
            if sd['sample_token'] == next_token:
                channel = get_sensor_channel(metadata, sd['calibrated_sensor_token'])
                if channel == 'LIDAR_TOP' and sd['is_key_frame']:
                    future_ego_token = sd['ego_pose_token']
                    if future_ego_token in ego_pose:
                        future_ego = ego_pose[future_ego_token]
                        future_pos = np.array(future_ego['translation'])
                        # Transform to ego frame
                        ego_frame_pos = transform_to_ego_frame(
                            future_pos, ego_translation, ego_rotation
                        )
                        future_positions.append(ego_frame_pos)
                    break
        
        next_token = future_sample['next']
    
    if len(future_positions) < 2:
        return None, []
    
    # Interpolate to num_waypoints
    future_positions = np.array(future_positions)  # (N, 2)
    
    # Create parameter for interpolation (normalized time)
    t_original = np.linspace(0, 1, len(future_positions))
    t_interp = np.linspace(0, 1, num_waypoints)
    
    trajectory = np.zeros((num_waypoints, 2))
    trajectory[:, 0] = np.interp(t_interp, t_original, future_positions[:, 0])
    trajectory[:, 1] = np.interp(t_interp, t_original, future_positions[:, 1])
    
    return trajectory.astype(np.float32), future_tokens


def get_annotations_for_sample(metadata: Dict, sample_token: str) -> List[Dict]:
    """Get all annotations for a sample with category names"""
    annotations = []
    
    sample_annotations = metadata.get('sample_annotation', {})
    categories = metadata.get('category', {})
    instances = metadata.get('instance', {})
    
    for ann_token, ann in sample_annotations.items():
        if ann['sample_token'] == sample_token:
            # Get category name
            category_name = 'unknown'
            instance_token = ann.get('instance_token', '')
            
            if instance_token in instances:
                category_token = instances[instance_token].get('category_token', '')
                if category_token in categories:
                    category_name = categories[category_token]['name']
            
            annotations.append({
                'token': ann_token,
                'translation': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation'],
                'category': category_name,
                'instance_token': instance_token,
                'visibility_token': ann.get('visibility_token', ''),
            })
    
    return annotations


def get_lidar_path_for_sample(metadata: Dict, sample_token: str) -> Optional[str]:
    """Get LIDAR_TOP file path for a sample"""
    sample_data = metadata['sample_data']
    
    for sd_token, sd in sample_data.items():
        if sd['sample_token'] == sample_token:
            channel = get_sensor_channel(metadata, sd['calibrated_sensor_token'])
            if channel == 'LIDAR_TOP' and sd['is_key_frame']:
                return sd['filename']
    
    return None


def load_lidar_points(filepath: str) -> np.ndarray:
    """Load LiDAR points from .pcd.bin file"""
    points = np.fromfile(filepath, dtype=np.float32)
    return points.reshape(-1, 5)[:, :4]  # x, y, z, intensity


def create_bev_representation(
    points: np.ndarray,
    x_range: Tuple[float, float] = (-50, 50),
    y_range: Tuple[float, float] = (-50, 50),
    z_range: Tuple[float, float] = (-3, 3),
    resolution: float = 0.2,
) -> np.ndarray:
    """Create BEV representation from point cloud"""
    grid_size = int((x_range[1] - x_range[0]) / resolution)
    
    # Filter points
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1])
    )
    points = points[mask]
    
    if len(points) == 0:
        return np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    
    # Convert to grid indices
    x_idx = ((points[:, 0] - x_range[0]) / resolution).astype(np.int32)
    y_idx = ((points[:, 1] - y_range[0]) / resolution).astype(np.int32)
    
    x_idx = np.clip(x_idx, 0, grid_size - 1)
    y_idx = np.clip(y_idx, 0, grid_size - 1)
    
    # Create BEV channels
    bev = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    
    # Channel 0: Height (max normalised z per cell)
    height = ((points[:, 2].astype(np.float64) - z_range[0]) / (z_range[1] - z_range[0])).astype(np.float32)
    np.maximum.at(bev[:, :, 0], (x_idx, y_idx), height)
    
    # Channel 1: Intensity (max per cell)
    if points.shape[1] >= 4:
        intensity = np.minimum(points[:, 3] / 255.0, 1.0).astype(np.float32)
        np.maximum.at(bev[:, :, 1], (x_idx, y_idx), intensity)
    
    # Channel 2: Density
    density = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(density, (x_idx, y_idx), 1)
    bev[:, :, 2] = np.clip(density / 10.0, 0, 1)
    
    return bev


def prepare_training_data(
    nuscenes_root: str,
    output_dir: str,
    num_waypoints: int = 20,
    max_samples: int = None,  # Set to limit (e.g., 10000), None for all
    save_every: int = 500,    # Save checkpoint every N samples
):
    """
    Prepare training data with REAL trajectories from ego_pose
    """
    nuscenes_root = Path(nuscenes_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find metadata directory
    metadata_dir = None
    for candidate in ['v1.0-trainval', 'v1.0-mini', 'v1.0-test']:
        if (nuscenes_root / candidate).exists():
            metadata_dir = nuscenes_root / candidate
            break
    
    if metadata_dir is None:
        raise FileNotFoundError(f"No metadata directory found in {nuscenes_root}")
    
    print("=" * 80)
    print("Loading nuScenes Metadata (with REAL trajectories)")
    print("=" * 80)
    print(f"Metadata directory: {metadata_dir}")
    
    metadata = load_nuscenes_metadata(metadata_dir)
    
    # Process all samples
    print("\n" + "=" * 80)
    print("Processing Samples with REAL Future Trajectories")
    print("=" * 80)
    
    checkpoint_path = output_dir / 'training_data_checkpoint.pkl'
    if checkpoint_path.exists():
        print(f"Found checkpoint, loading...")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        training_data = checkpoint['training_data']
        processed_tokens = set(checkpoint['processed_tokens'])
        label_counts = checkpoint['label_counts']
        print(f"  Resuming from {len(training_data)} samples")
    else:
        training_data = []
        processed_tokens = set()
        label_counts = {}
    
    samples_with_trajectory = len(training_data)
    samples_without_trajectory = 0
    samples_missing_lidar = 0
    
    samples_list = list(metadata['sample'].values())
    
    if max_samples:
        print(f"Max samples limit: {max_samples}")
    
    for idx, sample in enumerate(tqdm(samples_list, desc="Processing samples")):
        if max_samples and samples_with_trajectory >= max_samples:
            print(f"\nReached max_samples limit ({max_samples}), stopping.")
            break
        
        sample_token = sample['token']
        
        if sample_token in processed_tokens:
            continue
        
        # Get LiDAR path
        lidar_rel_path = get_lidar_path_for_sample(metadata, sample_token)
        if lidar_rel_path is None:
            samples_missing_lidar += 1
            processed_tokens.add(sample_token)
            continue
        
        lidar_full_path = nuscenes_root / lidar_rel_path
        if not lidar_full_path.exists():
            samples_missing_lidar += 1
            processed_tokens.add(sample_token)
            continue
        
        # Get REAL future trajectory
        gt_trajectory, future_tokens = get_future_ego_trajectory(
            metadata, sample_token, 
            num_future_samples=12,
            num_waypoints=num_waypoints
        )
        
        if gt_trajectory is None:
            samples_without_trajectory += 1
            processed_tokens.add(sample_token)
            continue
        
        samples_with_trajectory += 1
        
        # Get annotations
        annotations = get_annotations_for_sample(metadata, sample_token)
        
        for ann in annotations:
            cat = ann['category']
            label_counts[cat] = label_counts.get(cat, 0) + 1
        
        training_data.append({
            'sample_token': sample_token,
            'lidar_path': lidar_rel_path,
            'gt_trajectory': gt_trajectory,  # REAL trajectory!
            'annotations': annotations,
            'scene_token': sample['scene_token'],
            'timestamp': sample['timestamp'],
            'future_sample_tokens': future_tokens,
        })
        
        processed_tokens.add(sample_token)
        
        if save_every and len(training_data) % save_every == 0:
            checkpoint = {
                'training_data': training_data,
                'processed_tokens': list(processed_tokens),
                'label_counts': label_counts,
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"\n  [Checkpoint] Saved {len(training_data)} samples")
    
    print(f"\nProcessing complete:")
    print(f"  - Samples with REAL trajectory: {samples_with_trajectory}")
    print(f"  - Samples without enough future: {samples_without_trajectory}")
    print(f"  - Samples with missing LiDAR: {samples_missing_lidar}")
    
    print(f"\nLabel distribution:")
    for cat, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    
    output_path = output_dir / 'training_data_real.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"\nSaved training data to {output_path}")
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint (completed successfully)")
    
    # Save a sample for inspection
    if training_data:
        sample_output = output_dir / 'sample_with_real_trajectory.json'
        sample_data = training_data[0].copy()
        sample_data['gt_trajectory'] = sample_data['gt_trajectory'].tolist()
        with open(sample_output, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Saved sample to {sample_output}")
    
    return training_data


if __name__ == '__main__':
    NUSCENES_ROOT = '/home/ec2-user/SageMaker/nuscenes'
    OUTPUT_DIR = '/home/ec2-user/SageMaker/nuscenes/trajectory_output'
    
    prepare_training_data(
        nuscenes_root=NUSCENES_ROOT,
        output_dir=OUTPUT_DIR,
        num_waypoints=20,
        max_samples=None,    # Set to 10000 to limit, or None for all
        save_every=500,      # Save checkpoint every 500 samples
    )
