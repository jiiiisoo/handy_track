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
    
    return chain

def forward_kinematics_batch(chain, joint_angles_batch):
    """
    Compute forward kinematics for a batch of joint configurations
    Args:
        chain: pytorch_kinematics chain
        joint_angles_batch: [batch_size, n_joints] tensor of joint angles
    Returns:
        keypoints: [batch_size, n_keypoints, 3] tensor of 3D positions
    """
    batch_size = joint_angles_batch.shape[0]
    
    # Get all link transforms
    transforms = chain.forward_kinematics(joint_angles_batch)
    
    # Extract positions of all links
    keypoints = []
    link_names = list(transforms.keys())
    
    # Skip common root names that might not have meaningful positions
    skip_names = ['base_link', 'world', 'root', 'base']
    
    for link_name in link_names:
        if not any(skip in link_name.lower() for skip in skip_names):
            transform = transforms[link_name]
            # Extract translation part (last column, first 3 elements)
            position = transform.get_matrix()[:, :3, 3]  # [batch_size, 3]
            keypoints.append(position)
    
    if keypoints:
        # Stack all keypoints: [batch_size, n_links, 3]
        keypoints = torch.stack(keypoints, dim=1)
    else:
        # Fallback: use all links if none were selected
        print("Warning: No meaningful links found, using all links")
        for link_name in link_names:
            transform = transforms[link_name]
            position = transform.get_matrix()[:, :3, 3]  # [batch_size, 3]
            keypoints.append(position)
        keypoints = torch.stack(keypoints, dim=1)
    
    return keypoints

def create_comparison_animation(rollouts_data, chain, output_path, max_length=None):
    """Create animation comparing multiple rollouts side by side"""
    
    # Determine the number of rollouts and grid layout
    n_rollouts = len(rollouts_data)
    if n_rollouts <= 4:
        n_cols = 2
        n_rows = (n_rollouts + 1) // 2
    else:
        n_cols = 3
        n_rows = (n_rollouts + 2) // 3
    
    # Find the maximum timesteps across all rollouts
    if max_length is None:
        max_length = max(data['q'].shape[0] for data in rollouts_data.values())
    
    # Pre-compute keypoints for all rollouts
    all_keypoints = {}
    for rollout_idx, data in rollouts_data.items():
        q_data = data['q'][:max_length]  # Limit to max_length
        q_tensor = torch.FloatTensor(q_data)
        
        # Compute keypoints for this rollout
        keypoints = forward_kinematics_batch(chain, q_tensor)
        all_keypoints[rollout_idx] = keypoints.numpy()
        
        print(f"Rollout {rollout_idx}: computed {keypoints.shape[0]} frames with {keypoints.shape[1]} keypoints")
    
    # Setup the figure and subplots
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    axes = []
    scatters = []
    lines = []
    
    # Connection patterns for inspire hand visualization (simplified)
    # This is a simplified connection pattern - you might need to adjust based on actual hand structure
    connections = [
        # Basic connections between adjacent joints
        (0, 1), (1, 2), (2, 3),  # thumb chain
        (0, 4), (4, 5), (5, 6),  # index chain  
        (0, 7), (7, 8), (8, 9),  # middle chain
        (0, 10), (10, 11)        # simplified connections
    ]
    
    for i, (rollout_idx, keypoints) in enumerate(all_keypoints.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        axes.append(ax)
        
        # Set up the plot
        ax.set_title(f'Rollout {rollout_idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        
        # Determine plot limits from all keypoints
        all_points = keypoints.reshape(-1, 3)
        margin = 0.05
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
        
        # Initial scatter plot
        initial_frame = keypoints[0]  # [n_keypoints, 3]
        scatter = ax.scatter(initial_frame[:, 0], initial_frame[:, 1], initial_frame[:, 2], 
                           c='red', s=50, alpha=0.8)
        scatters.append(scatter)
        
        # Add lines for connections
        rollout_lines = []
        for connection in connections:
            if connection[0] < keypoints.shape[1] and connection[1] < keypoints.shape[1]:
                line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
                rollout_lines.append((line, connection))
        lines.append(rollout_lines)
    
    plt.tight_layout()
    
    def animate(frame):
        """Animation function"""
        for i, (rollout_idx, keypoints) in enumerate(all_keypoints.items()):
            if frame < keypoints.shape[0]:
                current_frame = keypoints[frame]  # [n_keypoints, 3]
                
                # Update scatter plot
                scatters[i]._offsets3d = (current_frame[:, 0], 
                                        current_frame[:, 1], 
                                        current_frame[:, 2])
                
                # Update lines
                for line, connection in lines[i]:
                    if connection[0] < current_frame.shape[0] and connection[1] < current_frame.shape[0]:
                        start_point = current_frame[connection[0]]
                        end_point = current_frame[connection[1]]
                        line.set_data_3d([start_point[0], end_point[0]],
                                       [start_point[1], end_point[1]], 
                                       [start_point[2], end_point[2]])
        
        return scatters + [line for rollout_lines in lines for line, _ in rollout_lines]
    
    # Create animation
    print(f"Creating animation with {max_length} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=max_length, 
                                 interval=100, blit=False, repeat=True)
    
    # Save animation
    print(f"Saving animation to {output_path}")
    anim.save(output_path, writer='pillow', fps=10)
    
    # Also save as mp4 if possible
    mp4_path = output_path.replace('.gif', '.mp4')
    try:
        anim.save(mp4_path, writer='ffmpeg', fps=10)
        print(f"Also saved as MP4: {mp4_path}")
    except:
        print("Could not save MP4 (ffmpeg not available)")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Render multiple robot hand rollouts for comparison')
    parser.add_argument('--hdf5_file', type=str, required=True,
                       help='Path to HDF5 file containing rollout data')
    parser.add_argument('--rollout_indices', type=int, nargs='+', default=None,
                       help='Specific rollout indices to visualize (default: all)')
    parser.add_argument('--max_rollouts', type=int, default=6,
                       help='Maximum number of rollouts to include')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum number of timesteps to animate')
    parser.add_argument('--output', type=str, default='robot_hand_videos/rollouts_comparison.gif',
                       help='Output video file path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Loading rollout data from: {args.hdf5_file}")
    
    # Load rollout data
    if args.rollout_indices is None:
        # Load first N rollouts
        args.rollout_indices = list(range(min(args.max_rollouts, 10)))  # Assume max 10 rollouts
    else:
        args.rollout_indices = args.rollout_indices[:args.max_rollouts]
    
    rollouts_data = load_rollout_data(args.hdf5_file, args.rollout_indices)
    
    if not rollouts_data:
        print("No rollout data found!")
        return
    
    print(f"Loaded {len(rollouts_data)} rollouts")
    
    # Setup kinematic chain
    print("Setting up inspire hand kinematic chain...")
    chain = setup_inspire_hand_chain()
    
    # Create comparison animation
    print("Creating comparison animation...")
    create_comparison_animation(rollouts_data, chain, args.output, args.max_length)
    
    print("✅ Comparison animation created successfully!")

if __name__ == '__main__':
    main() 