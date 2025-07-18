#!/usr/bin/env python3

import h5py
import numpy as np
import isaacgym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pytorch_kinematics as pk
import torch
import argparse
import os
import cv2
from tqdm import tqdm
import subprocess
from termcolor import cprint
import scipy.spatial.transform

# Try to import ffmpeg
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    print("Warning: ffmpeg-python not available. Install with: pip install ffmpeg-python")
    FFMPEG_AVAILABLE = False

# Try to import scipy for smoothing
try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Install with: pip install scipy")
    SCIPY_AVAILABLE = False

"""
NOTE: 
- q (joint positions) is always in the order of joint_names (chain.get_joint_parameter_names()).
- Visualization and keypoint extraction should always use dexhand.body_names (link names), which may be more than the number of joints.
- Forward kinematics must output keypoints in the order of dexhand.body_names for correct visualization.
"""

def load_rollout_data(hdf5_path, rollout_indices=None):
    """Load multiple rollout data from HDF5 file"""
    rollouts_data = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        successful_group = f['rollouts/successful']
        
        if rollout_indices is None:
            rollout_indices = list(range(len(successful_group.keys())))
        
        for idx in rollout_indices:
            rollout_key = f'rollout_{idx}'
            if rollout_key in successful_group:
                rollout_group = successful_group[rollout_key]
                
                rollouts_data[idx] = {
                    'q': np.array(rollout_group['q'][:]),      # joint positions [T, 12]
                    'dq': np.array(rollout_group['dq'][:]),    # joint velocities [T, 12]
                    'actions': np.array(rollout_group['actions'][:]),  # [T-1, 18]
                    'base_state': np.array(rollout_group['base_state'][:])  # [T, 13]
                }
                
                print(f"Loaded rollout {idx}: {rollouts_data[idx]['q'].shape[0]} timesteps")
    
    return rollouts_data

def setup_inspire_hand_chain():
    """Setup the inspire hand kinematic chain"""
    import sys
    import os
    
    # Add lib directory to path for imports
    lib_dir = os.path.join(os.path.dirname(__file__), "lib")
    if lib_dir not in sys.path:
        sys.path.append(lib_dir)
    
    from maniptrans_envs.lib.envs.dexhands import DexHandFactory
    
    # DexHand factory 자동 등록
    hands_dir = os.path.join(os.path.dirname(__file__), "maniptrans_envs", "lib", "envs", "dexhands")
    DexHandFactory.auto_register_hands(hands_dir, "maniptrans_envs.lib.envs.dexhands")
    
    # Hand 생성
    dexhand = DexHandFactory.create_hand("inspire", side="right")
    
    # URDF에서 kinematics chain 직접 생성
    asset_root = os.path.split(dexhand.urdf_path)[0]
    asset_file = os.path.split(dexhand.urdf_path)[1]
    urdf_path = os.path.join(asset_root, asset_file)
    
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # PyTorch Kinematics chain 생성
    chain = pk.build_chain_from_urdf(urdf_content)
    chain = chain.to(dtype=torch.float32, device='cpu')
    
    print(f"✅ Successfully set up inspire hand")
    print(f"   Chain has {chain.n_joints} joints")
    print(f"   Joint names: {[j.name for j in chain.get_joints()]}")
    print(f"   DexHand body names: {dexhand.body_names}")
    
    return chain, dexhand

def forward_kinematics_batch(chain, joint_angles_batch, body_names):
    """
    Compute forward kinematics for a batch of joint configurations.
    Returns keypoints in the order of body_names (dexhand.body_names).
    Args:
        chain: pytorch_kinematics chain
        joint_angles_batch: [batch_size, n_joints] tensor of joint angles
        body_names: list of link names (dexhand.body_names)
    Returns:
        keypoints: [batch_size, n_bodies, 3] tensor of 3D positions
    """
    batch_size = joint_angles_batch.shape[0]
    transforms = chain.forward_kinematics(joint_angles_batch)
    keypoints = []
    for body_name in body_names:
        if body_name in transforms:
            position = transforms[body_name].get_matrix()[:, :3, 3]  # [batch_size, 3]
            keypoints.append(position)
        else:
            # If body not found, use zeros
            keypoints.append(torch.zeros(batch_size, 3, dtype=torch.float32))
    keypoints = torch.stack(keypoints, dim=1)
    return keypoints

def smooth_temporal_data(data_sequence, sigma=1.0, method='gaussian'):
    """
    Apply temporal smoothing to motion data
    """
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available, skipping smoothing")
        return data_sequence
    
    if len(data_sequence) < 3:
        print("Warning: Sequence too short for meaningful smoothing")
        return data_sequence
    
    print(f"Applying temporal smoothing with method='{method}', sigma={sigma}")
    
    # Convert to numpy array
    data_array = np.array(data_sequence)
    
    if method == 'gaussian':
        smooth_data = np.zeros_like(data_array)
        for i in range(data_array.shape[-1]):  # For each DOF/coordinate
            if len(data_array.shape) == 2:
                smooth_data[:, i] = gaussian_filter1d(data_array[:, i], sigma=sigma)
            else:  # 3D array
                for j in range(data_array.shape[1]):
                    smooth_data[:, j, i] = gaussian_filter1d(data_array[:, j, i], sigma=sigma)
    
    elif method == 'moving_average':
        window_size = max(3, int(sigma * 3))
        if window_size % 2 == 0:
            window_size += 1
        
        def moving_average_1d(data, window):
            return np.convolve(data, np.ones(window)/window, mode='same')
        
        smooth_data = np.zeros_like(data_array)
        for i in range(data_array.shape[-1]):
            if len(data_array.shape) == 2:
                smooth_data[:, i] = moving_average_1d(data_array[:, i], window_size)
            else:
                for j in range(data_array.shape[1]):
                    smooth_data[:, j, i] = moving_average_1d(data_array[:, j, i], window_size)
    
    return smooth_data

def get_finger_connections(joint_names):
    """
    Create finger connections based on joint names (from render_retarget_multi.py)
    """
    finger_connections = {
        'thumb': [],
        'index': [],
        'middle': [],
        'ring': [],
        'pinky': [],
        'wrist': []
    }
    
    # joint_names를 손가락별로 분류
    for i, name in enumerate(joint_names):
        if 'thumb' in name:
            finger_connections['thumb'].append(i)
        elif 'index' in name:
            finger_connections['index'].append(i)
        elif 'middle' in name:
            finger_connections['middle'].append(i)
        elif 'ring' in name:
            finger_connections['ring'].append(i)
        elif 'pinky' in name:
            finger_connections['pinky'].append(i)
        else:
            finger_connections['wrist'].append(i)
    
    return finger_connections

def get_finger_colors():
    """
    Get finger color scheme (from render_retarget_multi.py)
    """
    return {
        'thumb': 'red',
        'index': 'blue', 
        'middle': 'green',
        'ring': 'orange',
        'pinky': 'purple',
        'wrist': 'black'
    }

def quat_to_rotmat(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix"""
    # Accepts numpy array or torch tensor
    if isinstance(quat, torch.Tensor):
        quat = quat.detach().cpu().numpy()
    rot = scipy.spatial.transform.Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return rot.as_matrix()

# --- New: Visualize each rollout independently ---
def create_single_rollout_video(
    q, base_state, chain, body_names, output_path, max_length=None, fps=10, smooth=False, smooth_sigma=1.5):
    """
    Visualize a single rollout as a video, applying wrist position/rotation to FK results.
    """
    if max_length is not None:
        q = q[:max_length]
        base_state = base_state[:max_length]
    T = q.shape[0]
    if smooth and T > 3:
        q = smooth_temporal_data(q, sigma=smooth_sigma)
    q_tensor = torch.FloatTensor(q)
    # FK in local hand frame
    local_kps = forward_kinematics_batch(chain, q_tensor, body_names).numpy()  # [T, n_bodies, 3]
    # Apply wrist pose to get world coordinates
    world_kps = np.zeros_like(local_kps)
    for t in range(T):
        wrist_pos = base_state[t, :3]
        wrist_quat = base_state[t, 3:7]
        R = quat_to_rotmat(wrist_quat)  # [3,3]
        world_kps[t] = (R @ local_kps[t].T).T + wrist_pos  # [n_bodies, 3]
    # Visualization
    finger_connections = get_finger_connections(body_names)
    finger_colors = get_finger_colors()
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    for t in range(T):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        keypoints = world_kps[t]
        for finger, indices in finger_connections.items():
            if len(indices) > 0:
                color = finger_colors[finger]
                finger_points = keypoints[indices]
                ax.scatter(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2], c=color, s=50, alpha=0.8, label=finger)
                # Connect joints within finger
                if finger != 'wrist' and len(indices) > 1:
                    sorted_indices = []
                    for joint_type in ['proximal', 'intermediate', 'tip']:
                        for idx in indices:
                            if joint_type in body_names[idx]:
                                sorted_indices.append(idx)
                    if len(sorted_indices) > 1:
                        sorted_points = keypoints[sorted_indices]
                        ax.plot(sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2], c=color, linewidth=2, alpha=0.7)
                # Connect wrist to finger base
                if finger != 'wrist' and len(finger_connections['wrist']) > 0:
                    wrist_idx = finger_connections['wrist'][0]
                    first_joint_idx = indices[0]
                    for idx in indices:
                        if 'proximal' in body_names[idx]:
                            first_joint_idx = idx
                            break
                    wrist_pos = keypoints[wrist_idx]
                    first_joint_pos = keypoints[first_joint_idx]
                    ax.plot([wrist_pos[0], first_joint_pos[0]], [wrist_pos[1], first_joint_pos[1]], [wrist_pos[2], first_joint_pos[2]], c=color, linewidth=1.5, alpha=0.5, linestyle='--')
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"Frame {t+1}/{T}")
        # Set consistent bounds
        all_points = world_kps.reshape(-1, 3)
        margin = 0.05
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
        if t == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    # Video
    if FFMPEG_AVAILABLE:
        create_video_from_images_ffmpeg(temp_dir, output_path, fps)
    else:
        create_video_from_images_cv2(temp_dir, output_path, fps)
    import shutil
    shutil.rmtree(temp_dir)
    print(f"✅ Saved: {output_path}")

def create_video_from_images_ffmpeg(image_dir, output_path, fps=30):
    """Create video using ffmpeg-python"""
    if not FFMPEG_AVAILABLE:
        return False
    try:
        input_pattern = os.path.join(image_dir, 'frame_%04d.png')
        input_stream = ffmpeg.input(input_pattern, framerate=fps)
        scaled_stream = ffmpeg.filter(input_stream, 'scale', 'trunc(iw/2)*2', 'trunc(ih/2)*2')
        output_stream = ffmpeg.output(
            scaled_stream, 
            output_path,
            vcodec='libx264',
            pix_fmt='yuv420p',
            crf=18
        )
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
        return True
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False

def create_video_from_images_cv2(image_dir, output_path, fps=30):
    """Create video using OpenCV (fallback)"""
    try:
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if not image_files:
            return False
        first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
        height, width, layers = first_image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            out.write(frame)
        out.release()
        return True
    except Exception as e:
        print(f"OpenCV error: {e}")
        return False

# --- Main: generate per-rollout videos ---
def main():
    parser = argparse.ArgumentParser(description='Per-rollout robot hand visualization')
    parser.add_argument('--hdf5_file', type=str, required=True,
                       help='Path to HDF5 file containing rollout data')
    parser.add_argument('--rollout_indices', type=int, nargs='+', default=None,
                       help='Specific rollout indices to visualize (default: all)')
    parser.add_argument('--max_rollouts', type=int, default=10,
                       help='Maximum number of rollouts to include')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum number of timesteps to animate')
    parser.add_argument('--output_dir', type=str, default='robot_hand_videos/',
                       help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=10,
                       help='Video frame rate')
    parser.add_argument('--smooth', action='store_true',
                       help='Apply temporal smoothing to joint trajectories')
    parser.add_argument('--smooth_sigma', type=float, default=1.5,
                       help='Smoothing strength (higher = more smoothing)')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading rollout data from: {args.hdf5_file}")
    if args.rollout_indices is None:
        args.rollout_indices = list(range(min(args.max_rollouts, 10)))
    else:
        args.rollout_indices = args.rollout_indices[:args.max_rollouts]
    rollouts_data = load_rollout_data(args.hdf5_file, args.rollout_indices)
    if not rollouts_data:
        print("No rollout data found!"); return
    print(f"Loaded {len(rollouts_data)} rollouts")
    print("Setting up inspire hand kinematic chain...")
    chain, dexhand = setup_inspire_hand_chain()
    for idx, data in rollouts_data.items():
        print(f"\n--- Visualizing rollout {idx} ---")
        out_path = os.path.join(args.output_dir, f"rollout_{idx}_hand.mp4")
        create_single_rollout_video(
            q=data['q'],
            base_state=data['base_state'],
            chain=chain,
            body_names=dexhand.body_names,
            output_path=out_path,
            max_length=args.max_length,
            fps=args.fps,
            smooth=args.smooth,
            smooth_sigma=args.smooth_sigma
        )
    print("All rollouts visualized!")

if __name__ == '__main__':
    main() 