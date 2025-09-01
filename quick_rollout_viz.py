#!/usr/bin/env python3
"""
롤아웃 데이터 빠른 확인 및 시각화 스크립트
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os

def explore_hdf5_structure(filepath):
    """HDF5 파일 구조 확인"""
    print(f"🔍 Exploring HDF5 structure: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  📊 Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
            else:
                print(f"  📁 Group: {name}")
        
        f.visititems(print_structure)
        
        # 롤아웃 정보 확인
        if 'rollouts/successful' in f:
            successful_rollouts = list(f['rollouts/successful'].keys())
            print(f"\n✅ Found {len(successful_rollouts)} successful rollouts:")
            for rollout in successful_rollouts[:5]:  # 처음 5개만 표시
                print(f"  - {rollout}")
            if len(successful_rollouts) > 5:
                print(f"  ... and {len(successful_rollouts) - 5} more")
        
        if 'rollouts/failed' in f:
            failed_rollouts = list(f['rollouts/failed'].keys())
            print(f"\n❌ Found {len(failed_rollouts)} failed rollouts")

def quick_visualize(filepath, rollout_name=None):
    """빠른 시각화"""
    print(f"🎬 Creating quick visualization...")
    
    with h5py.File(filepath, 'r') as f:
        if 'rollouts/successful' not in f:
            print("❌ No successful rollouts found!")
            return
        
        # 롤아웃 선택
        available_rollouts = list(f['rollouts/successful'].keys())
        if rollout_name is None or rollout_name not in available_rollouts:
            rollout_name = available_rollouts[0]
        
        print(f"📋 Using rollout: {rollout_name}")
        
        rollout = f[f'rollouts/successful/{rollout_name}']
        
        # 데이터 로드
        q = rollout['q'][:]                    # 관절 각도
        base_state = rollout['base_state'][:]  # 손목 상태
        
        print(f"📊 Data info:")
        print(f"  - Timesteps: {len(q)}")
        print(f"  - Joint angles shape: {q.shape}")
        print(f"  - Base state shape: {base_state.shape}")
        
        # 간단한 플롯
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 관절 각도 변화
        axes[0, 0].plot(q)
        axes[0, 0].set_title('Joint Angles Over Time')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Angle (rad)')
        
        # 손목 위치
        wrist_pos = base_state[:, :3]
        axes[0, 1].plot(wrist_pos)
        axes[0, 1].set_title('Wrist Position')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Position (m)')
        axes[0, 1].legend(['X', 'Y', 'Z'])
        
        # 손목 속도
        wrist_vel = base_state[:, 7:10]
        axes[1, 0].plot(wrist_vel)
        axes[1, 0].set_title('Wrist Velocity')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Velocity (m/s)')
        axes[1, 0].legend(['X', 'Y', 'Z'])
        
        # 궤적 3D
        ax = axes[1, 1]
        ax.plot(wrist_pos[:, 0], wrist_pos[:, 1], 'b-', alpha=0.7)
        ax.scatter(wrist_pos[0, 0], wrist_pos[0, 1], c='green', s=100, label='Start')
        ax.scatter(wrist_pos[-1, 0], wrist_pos[-1, 1], c='red', s=100, label='End')
        ax.set_title('Wrist XY Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # 저장
        output_path = f"rollout_{rollout_name}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved analysis plot to: {output_path}")
        plt.show()

def create_simple_animation(filepath, rollout_name=None, save_video=False):
    """간단한 애니메이션 생성"""
    print(f"🎥 Creating animation...")
    
    with h5py.File(filepath, 'r') as f:
        if 'rollouts/successful' not in f:
            print("❌ No successful rollouts found!")
            return
        
        available_rollouts = list(f['rollouts/successful'].keys())
        if rollout_name is None or rollout_name not in available_rollouts:
            rollout_name = available_rollouts[0]
        
        rollout = f[f'rollouts/successful/{rollout_name}']
        
        q = rollout['q'][:]
        base_state = rollout['base_state'][:]
        
        wrist_pos = base_state[:, :3]
        
        # 애니메이션 설정
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 전체 궤적 표시
        ax.plot(wrist_pos[:, 0], wrist_pos[:, 1], 'lightblue', alpha=0.5, linewidth=1)
        
        # 현재 위치
        point, = ax.plot([], [], 'ro', markersize=8)
        trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
        
        ax.set_xlim(wrist_pos[:, 0].min() - 0.1, wrist_pos[:, 0].max() + 0.1)
        ax.set_ylim(wrist_pos[:, 1].min() - 0.1, wrist_pos[:, 1].max() + 0.1)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Wrist Movement Animation - {rollout_name}')
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            # 현재 위치
            point.set_data([wrist_pos[frame, 0]], [wrist_pos[frame, 1]])
            
            # 트레일 (최근 20프레임)
            start_idx = max(0, frame - 20)
            trail.set_data(wrist_pos[start_idx:frame+1, 0], wrist_pos[start_idx:frame+1, 1])
            
            return point, trail
        
        # 애니메이션 생성
        anim = animation.FuncAnimation(fig, animate, frames=len(wrist_pos), 
                                     interval=50, blit=True, repeat=True)
        
        if save_video:
            output_path = f"rollout_{rollout_name}_animation.mp4"
            anim.save(output_path, writer='ffmpeg', fps=20)
            print(f"💾 Saved animation to: {output_path}")
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick rollout data exploration and visualization')
    parser.add_argument('--hdf5_path', type=str, default='rollouts.hdf5',
                       help='Path to HDF5 rollout file')
    parser.add_argument('--rollout_name', type=str, default=None,
                       help='Specific rollout to visualize')
    parser.add_argument('--explore_only', action='store_true',
                       help='Only explore structure, no visualization')
    parser.add_argument('--animate', action='store_true',
                       help='Create animation instead of static plots')
    parser.add_argument('--save_video', action='store_true',
                       help='Save animation as video file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_path):
        print(f"❌ HDF5 file not found: {args.hdf5_path}")
        exit(1)
    
    print(f"🚀 Quick Rollout Visualization Tool")
    print(f"📁 File: {args.hdf5_path}")
    
    # 구조 탐색
    explore_hdf5_structure(args.hdf5_path)
    
    if not args.explore_only:
        if args.animate:
            create_simple_animation(args.hdf5_path, args.rollout_name, args.save_video)
        else:
            quick_visualize(args.hdf5_path, args.rollout_name) 