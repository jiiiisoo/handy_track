#!/usr/bin/env python3
"""
HDF5 rollout 데이터를 사용해서 로봇 손 렌더링 비디오를 생성하는 스크립트
"""

import h5py
import numpy as np
import isaacgym
import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# 기존 렌더링 함수들 import
from render_retarget_multi import render_multiview_keypoints_sequence, smooth_temporal_data
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.mano2dexhand import Mano2Dexhand

def quaternion_to_axis_angle(quat):
    """
    Quaternion을 axis-angle로 변환
    quat: [w, x, y, z] format
    """
    quat = quat / np.linalg.norm(quat)  # normalize
    w, x, y, z = quat
    
    # axis-angle 변환
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    
    if angle < 1e-6:  # 회전이 거의 없는 경우
        return np.array([0, 0, 0])
    
    axis = np.array([x, y, z]) / np.sin(angle / 2)
    axis_angle = axis * angle
    
    return axis_angle

def load_rollout_data(hdf5_path, rollout_name="rollout_0"):
    """HDF5에서 rollout 데이터 로드"""
    print(f"🔍 Loading rollout data: {rollout_name}")
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'rollouts/successful' not in f:
            raise ValueError("No successful rollouts found in HDF5 file!")
            
        if rollout_name not in f['rollouts/successful']:
            available = list(f['rollouts/successful'].keys())
            print(f"Available rollouts: {available}")
            rollout_name = available[0]  # 첫 번째 사용
            print(f"Using: {rollout_name}")
        
        rollout = f[f'rollouts/successful/{rollout_name}']
        
        # 데이터 로드
        q = rollout['q'][:]           # 관절 각도 (T, 12)
        base_state = rollout['base_state'][:]  # 손목 상태 (T, 13)
        
        print(f"📊 Data shapes:")
        print(f"   Joint angles (q): {q.shape}")
        print(f"   Base state: {base_state.shape}")
        print(f"   Episode length: {len(q)} steps")
        
        # base_state에서 위치와 회전 분리
        # base_state = [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        wrist_pos_seq = base_state[:, :3]  # 위치 (T, 3)
        wrist_quat_seq = base_state[:, 3:7]  # 쿼터니언 (T, 4)
        
        # 쿼터니언을 axis-angle로 변환
        wrist_rot_seq = []
        for quat in wrist_quat_seq:
            axis_angle = quaternion_to_axis_angle(quat)
            wrist_rot_seq.append(axis_angle)
        wrist_rot_seq = np.array(wrist_rot_seq)
        
        print(f"   Wrist position range: [{wrist_pos_seq.min():.3f}, {wrist_pos_seq.max():.3f}]")
        print(f"   Wrist rotation range: [{wrist_rot_seq.min():.3f}, {wrist_rot_seq.max():.3f}]")
        print(f"   Joint angle range: [{q.min():.3f}, {q.max():.3f}]")
        
        return {
            'wrist_pos_seq': wrist_pos_seq,    # (T, 3)
            'wrist_rot_seq': wrist_rot_seq,    # (T, 3)
            'dof_pos_seq': q,                  # (T, 12)
            'rollout_name': rollout_name,
            'episode_length': len(q)
        }

def expand_dof_to_inspire_hand(dof_12, dexhand_type='inspire'):
    """
    12차원 DOF를 inspire hand의 전체 DOF로 확장
    """
    if dexhand_type == 'inspire':
        # Inspire hand는 일반적으로 20 DOF
        # 12차원을 20차원으로 확장 (나머지는 0으로 채움)
        dof_20 = np.zeros(20)
        dof_20[:12] = dof_12  # 처음 12개만 사용
        return dof_20
    else:
        return dof_12

def create_video_from_rollout(hdf5_path, rollout_name="rollout_0", output_dir="rollout_videos", 
                            fps=30, smooth=True, dexhand_type='inspire'):
    """메인 비디오 생성 함수"""
    
    # 1. 데이터 로드
    data = load_rollout_data(hdf5_path, rollout_name)
    
    # 2. DexHand 설정
    print(f"🤖 Setting up {dexhand_type} hand...")
    dexhand = DexHandFactory.create_dexhand(dexhand_type)
    
    # 3. Mano2Dexhand 변환기 설정 (필요한 경우)
    try:
        mano2inspire = Mano2Dexhand(
            dexhand_name=dexhand_type,
            device='cuda:0',
            use_pca=False  # PCA 사용 안함
        )
    except Exception as e:
        print(f"Warning: Could not initialize Mano2Dexhand: {e}")
        mano2inspire = None
    
    # 4. DOF 확장 (12 -> 20)
    dof_pos_seq_expanded = []
    for dof_12 in data['dof_pos_seq']:
        dof_20 = expand_dof_to_inspire_hand(dof_12, dexhand_type)
        dof_pos_seq_expanded.append(dof_20)
    
    # 5. 스무딩 적용 (옵션)
    wrist_pos_seq = data['wrist_pos_seq']
    wrist_rot_seq = data['wrist_rot_seq']
    dof_pos_seq = np.array(dof_pos_seq_expanded)
    
    if smooth and len(wrist_pos_seq) > 3:
        print(f"🎬 Applying temporal smoothing...")
        try:
            wrist_pos_seq, wrist_rot_seq, dof_pos_seq = smooth_temporal_data(
                opt_wrist_pos_seq=wrist_pos_seq,
                opt_wrist_rot_seq=wrist_rot_seq,
                opt_dof_pos_seq=dof_pos_seq,
                sigma=1.0,
                method='gaussian'
            )
            print("   Smoothing completed!")
        except Exception as e:
            print(f"   Warning: Smoothing failed: {e}")
    
    # 6. 리스트로 변환 (렌더링 함수 요구사항)
    wrist_pos_list = [pos for pos in wrist_pos_seq]
    wrist_rot_list = [rot for rot in wrist_rot_seq]
    dof_pos_list = [dof for dof in dof_pos_seq]
    
    # 7. 출력 디렉토리 설정
    hdf5_name = Path(hdf5_path).stem
    video_save_dir = f"{output_dir}/{hdf5_name}_{rollout_name}"
    os.makedirs(video_save_dir, exist_ok=True)
    
    print(f"📹 Rendering video...")
    print(f"   Input: {len(wrist_pos_list)} frames")
    print(f"   Output: {video_save_dir}")
    print(f"   FPS: {fps}")
    
    # 8. 비디오 렌더링
    try:
        if mano2inspire and hasattr(mano2inspire, 'chain'):
            chain = mano2inspire.chain
        else:
            chain = None
            
        render_multiview_keypoints_sequence(
            opt_wrist_pos_seq=wrist_pos_list,
            opt_wrist_rot_seq=wrist_rot_list,
            opt_dof_pos_seq=dof_pos_list,
            chain=chain,
            joint_names=dexhand.body_names,
            save_dir=video_save_dir,
            prefix=f"{rollout_name}_render",
            fps=fps
        )
        print(f"✅ Video rendering completed!")
        print(f"   Check: {video_save_dir}")
        
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        print("   Trying simplified rendering...")
        
        # 간단한 fallback 렌더링
        render_simple_rollout_video(
            wrist_pos_seq=wrist_pos_seq,
            wrist_rot_seq=wrist_rot_seq,
            dof_pos_seq=dof_pos_seq,
            save_path=f"{video_save_dir}/{rollout_name}_simple.mp4",
            fps=fps
        )

def render_simple_rollout_video(wrist_pos_seq, wrist_rot_seq, dof_pos_seq, 
                               save_path="rollout_simple.mp4", fps=30):
    """간단한 matplotlib 기반 비디오 렌더링"""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    print(f"🎬 Creating simple visualization video...")
    
    fig = plt.figure(figsize=(12, 8))
    
    # 서브플롯 생성
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    def animate(frame_idx):
        fig.clear()
        
        # 3D 손목 궤적
        ax1 = plt.subplot(2, 2, 1, projection='3d')
        ax1.plot(wrist_pos_seq[:frame_idx+1, 0], 
                wrist_pos_seq[:frame_idx+1, 1], 
                wrist_pos_seq[:frame_idx+1, 2], 'b-', alpha=0.7)
        ax1.scatter(wrist_pos_seq[frame_idx, 0], 
                   wrist_pos_seq[frame_idx, 1], 
                   wrist_pos_seq[frame_idx, 2], 'ro', s=100)
        ax1.set_title(f'Wrist Trajectory (Frame {frame_idx+1}/{len(wrist_pos_seq)})')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 관절 각도 히트맵
        ax2 = plt.subplot(2, 2, 2)
        if frame_idx > 0:
            im = ax2.imshow(dof_pos_seq[:frame_idx+1].T, aspect='auto', cmap='viridis')
            ax2.axvline(frame_idx, color='red', linewidth=2)
        ax2.set_title('Joint Angles Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Joint Index')
        
        # 현재 프레임 관절 각도
        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(range(len(dof_pos_seq[frame_idx])), dof_pos_seq[frame_idx])
        ax3.set_title(f'Current Joint Angles (Frame {frame_idx+1})')
        ax3.set_xlabel('Joint Index')
        ax3.set_ylabel('Angle (rad)')
        
        # 손목 회전
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(wrist_rot_seq[:frame_idx+1, 0], 'r-', label='X')
        ax4.plot(wrist_rot_seq[:frame_idx+1, 1], 'g-', label='Y')
        ax4.plot(wrist_rot_seq[:frame_idx+1, 2], 'b-', label='Z')
        ax4.axvline(frame_idx, color='black', linestyle='--', alpha=0.7)
        ax4.set_title('Wrist Rotation (Axis-Angle)')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Rotation (rad)')
        ax4.legend()
        
        plt.tight_layout()
    
    # 애니메이션 생성
    anim = animation.FuncAnimation(fig, animate, frames=len(wrist_pos_seq), 
                                 interval=1000/fps, blit=False)
    
    # 비디오 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer='ffmpeg', fps=fps, bitrate=1800)
    print(f"✅ Simple video saved: {save_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Render robot hand videos from HDF5 rollout data')
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='Path to HDF5 rollout file')
    parser.add_argument('--rollout_name', type=str, default="rollout_0",
                       help='Name of rollout to render (default: rollout_0)')
    parser.add_argument('--output_dir', type=str, default="rollout_videos",
                       help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS (default: 30)')
    parser.add_argument('--smooth', action='store_true',
                       help='Apply temporal smoothing')
    parser.add_argument('--dexhand_type', type=str, default='inspire',
                       help='DexHand type (default: inspire)')
    parser.add_argument('--list_rollouts', action='store_true',
                       help='List available rollouts and exit')
    
    args = parser.parse_args()
    
    # HDF5 파일 존재 확인
    if not os.path.exists(args.hdf5_path):
        print(f"❌ HDF5 file not found: {args.hdf5_path}")
        return
    
    # 롤아웃 목록 출력만 하는 경우
    if args.list_rollouts:
        with h5py.File(args.hdf5_path, 'r') as f:
            if 'rollouts/successful' in f:
                rollouts = list(f['rollouts/successful'].keys())
                print(f"📋 Available successful rollouts in {args.hdf5_path}:")
                for i, rollout in enumerate(rollouts):
                    data = f[f'rollouts/successful/{rollout}']
                    length = data['q'].shape[0]
                    print(f"   {i+1}. {rollout} ({length} steps)")
            else:
                print("❌ No successful rollouts found!")
        return
    
    # 비디오 생성
    try:
        create_video_from_rollout(
            hdf5_path=args.hdf5_path,
            rollout_name=args.rollout_name,
            output_dir=args.output_dir,
            fps=args.fps,
            smooth=args.smooth,
            dexhand_type=args.dexhand_type
        )
    except Exception as e:
        print(f"❌ Failed to create video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 