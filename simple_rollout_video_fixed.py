#!/usr/bin/env python3
"""
HDF5 rollout 데이터를 matplotlib으로 간단하게 시각화하는 스크립트 (수정된 버전)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
from pathlib import Path

def quaternion_to_axis_angle(quat):
    """Quaternion을 axis-angle로 변환"""
    quat = quat / np.linalg.norm(quat)  # normalize
    w, x, y, z = quat
    
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    
    if angle < 1e-6:
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
            rollout_name = available[0]
            print(f"Using: {rollout_name}")
        
        rollout = f[f'rollouts/successful/{rollout_name}']
        
        # 데이터 로드
        q = rollout['q'][:]                    # 관절 각도 (T, 12)
        base_state = rollout['base_state'][:]  # 손목 상태 (T, 13)
        actions = rollout['actions'][:]        # 행동 (T-1, 18)
        
        print(f"📊 Data shapes:")
        print(f"   Joint angles (q): {q.shape}")
        print(f"   Base state: {base_state.shape}")
        print(f"   Actions: {actions.shape}")
        print(f"   Episode length: {len(q)} steps")
        
        # base_state 분해
        wrist_pos_seq = base_state[:, :3]      # 위치 (T, 3)
        wrist_quat_seq = base_state[:, 3:7]    # 쿼터니언 (T, 4)
        wrist_vel_seq = base_state[:, 7:10]    # 선속도 (T, 3)
        wrist_angvel_seq = base_state[:, 10:13] # 각속도 (T, 3)
        
        # 쿼터니언을 axis-angle로 변환
        wrist_rot_seq = []
        for quat in wrist_quat_seq:
            axis_angle = quaternion_to_axis_angle(quat)
            wrist_rot_seq.append(axis_angle)
        wrist_rot_seq = np.array(wrist_rot_seq)
        
        print(f"   Wrist position range: [{wrist_pos_seq.min():.3f}, {wrist_pos_seq.max():.3f}]")
        print(f"   Joint angle range: [{q.min():.3f}, {q.max():.3f}]")
        
        return {
            'q': q,
            'wrist_pos': wrist_pos_seq,
            'wrist_rot': wrist_rot_seq,
            'wrist_vel': wrist_vel_seq,
            'wrist_angvel': wrist_angvel_seq,
            'actions': actions,
            'rollout_name': rollout_name,
            'episode_length': len(q)
        }

def create_rollout_video(data, save_path="rollout_video.mp4", fps=10):
    """롤아웃 데이터로 애니메이션 비디오 생성"""
    
    q = data['q']
    wrist_pos = data['wrist_pos']
    wrist_rot = data['wrist_rot']
    wrist_vel = data['wrist_vel']
    actions = data['actions']
    episode_length = data['episode_length']
    
    print(f"🎬 Creating rollout video...")
    print(f"   Frames: {episode_length}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {episode_length/fps:.1f} seconds")
    
    # Figure 설정
    fig = plt.figure(figsize=(15, 10))
    
    def animate(frame_idx):
        fig.clear()
        fig.suptitle(f"Robot Hand Rollout: {data['rollout_name']} - Frame {frame_idx+1}/{episode_length}", fontsize=14)
        
        try:
            # 1. 3D 손목 궤적
            ax1 = plt.subplot(2, 3, 1, projection='3d')
            if frame_idx > 0:
                ax1.plot(wrist_pos[:frame_idx+1, 0], 
                        wrist_pos[:frame_idx+1, 1], 
                        wrist_pos[:frame_idx+1, 2], 'b-', alpha=0.7, linewidth=2)
            ax1.scatter(wrist_pos[frame_idx, 0], 
                       wrist_pos[frame_idx, 1], 
                       wrist_pos[frame_idx, 2], color='red', s=100, alpha=1.0)
            
            if frame_idx > 0:
                ax1.scatter(wrist_pos[0, 0], wrist_pos[0, 1], wrist_pos[0, 2], 
                           color='green', s=80, alpha=0.8)
            
            ax1.set_title('3D Wrist Trajectory')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            
            # 2. 손목 위치 (2D 투영)
            ax2 = plt.subplot(2, 3, 2)
            if frame_idx > 0:
                ax2.plot(wrist_pos[:frame_idx+1, 0], wrist_pos[:frame_idx+1, 1], 'b-', alpha=0.7, linewidth=2)
            ax2.scatter(wrist_pos[frame_idx, 0], wrist_pos[frame_idx, 1], color='red', s=100)
            if frame_idx > 0:
                ax2.scatter(wrist_pos[0, 0], wrist_pos[0, 1], color='green', s=80)
            ax2.set_title('Wrist Position (XY Plane)')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            
            # 3. 손목 높이 변화
            ax3 = plt.subplot(2, 3, 3)
            if frame_idx > 0:
                ax3.plot(range(frame_idx+1), wrist_pos[:frame_idx+1, 2], 'g-', linewidth=2)
            ax3.scatter(frame_idx, wrist_pos[frame_idx, 2], color='red', s=100)
            ax3.set_title('Wrist Height (Z)')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Z Position (m)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, episode_length-1)
            ax3.set_ylim(wrist_pos[:, 2].min() - 0.1, wrist_pos[:, 2].max() + 0.1)
            
            # 4. 관절 각도 히트맵
            ax4 = plt.subplot(2, 3, 4)
            if frame_idx > 0:
                im = ax4.imshow(q[:frame_idx+1].T, aspect='auto', cmap='viridis', interpolation='nearest')
                ax4.axvline(frame_idx, color='red', linewidth=2)
                try:
                    plt.colorbar(im, ax=ax4, label='Angle (rad)')
                except:
                    pass  # colorbar 에러 무시
            ax4.set_title('Joint Angles Over Time')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Joint Index')
            
            # 5. 현재 관절 각도
            ax5 = plt.subplot(2, 3, 5)
            joint_angles = q[frame_idx]
            colors = ['red' if abs(angle) > 0.1 else 'blue' for angle in joint_angles]
            ax5.bar(range(len(joint_angles)), joint_angles, color=colors)
            ax5.set_title(f'Current Joint Angles (Frame {frame_idx+1})')
            ax5.set_xlabel('Joint Index')
            ax5.set_ylabel('Angle (rad)')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim(q.min() - 0.1, q.max() + 0.1)
            
            # 6. 손목 속도
            ax6 = plt.subplot(2, 3, 6)
            if frame_idx > 0:
                speed = np.linalg.norm(wrist_vel[:frame_idx+1], axis=1)
                ax6.plot(range(frame_idx+1), speed, 'purple', linewidth=2)
                ax6.scatter(frame_idx, speed[frame_idx], color='red', s=100)
            ax6.set_title('Wrist Speed')
            ax6.set_xlabel('Time Step')
            ax6.set_ylabel('Speed (m/s)')
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim(0, episode_length-1)
            speed_all = np.linalg.norm(wrist_vel, axis=1)
            ax6.set_ylim(0, speed_all.max() + 0.1)
            
        except Exception as e:
            print(f"   Warning: Error in frame {frame_idx}: {e}")
        
        plt.tight_layout()
    
    # 애니메이션 생성
    print("   Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=episode_length, 
                                 interval=1000/fps, blit=False, repeat=True)
    
    # 비디오 저장
    print(f"   Saving video to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # 먼저 GIF로 시도 (더 호환성이 좋음)
        gif_path = save_path.replace('.mp4', '.gif')
        print(f"   Saving as GIF: {gif_path}")
        anim.save(gif_path, writer='pillow', fps=fps)
        print(f"✅ GIF saved successfully!")
        
        # MP4도 시도
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"✅ MP4 video also saved!")
        except:
            print("   MP4 failed, but GIF succeeded")
            
    except Exception as e:
        print(f"❌ Video creation failed: {e}")
        
        # 마지막 시도: 정적 이미지들 저장
        print("   Trying to save static frames...")
        frames_dir = save_path.replace('.mp4', '_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        for i in range(0, episode_length, max(1, episode_length//10)):  # 10개 프레임만
            animate(i)
            plt.savefig(f"{frames_dir}/frame_{i:03d}.png", dpi=150, bbox_inches='tight')
        print(f"✅ Static frames saved to: {frames_dir}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create simple rollout videos from HDF5 data')
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='Path to HDF5 rollout file')
    parser.add_argument('--rollout_name', type=str, default="rollout_0",
                       help='Name of rollout to render')
    parser.add_argument('--output_dir', type=str, default="simple_rollout_videos",
                       help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=5,
                       help='Video FPS')
    parser.add_argument('--list_rollouts', action='store_true',
                       help='List available rollouts and exit')
    
    args = parser.parse_args()
    
    # HDF5 파일 존재 확인
    if not os.path.exists(args.hdf5_path):
        print(f"❌ HDF5 file not found: {args.hdf5_path}")
        return
    
    # 롤아웃 목록 출력
    if args.list_rollouts:
        with h5py.File(args.hdf5_path, 'r') as f:
            if 'rollouts/successful' in f:
                rollouts = list(f['rollouts/successful'].keys())
                print(f"📋 Available successful rollouts:")
                for i, rollout in enumerate(rollouts):
                    data = f[f'rollouts/successful/{rollout}']
                    length = data['q'].shape[0]
                    print(f"   {i+1}. {rollout} ({length} steps)")
            else:
                print("❌ No successful rollouts found!")
        return
    
    # 데이터 로드
    try:
        data = load_rollout_data(args.hdf5_path, args.rollout_name)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # 비디오 생성
    try:
        hdf5_name = Path(args.hdf5_path).stem
        output_path = f"{args.output_dir}/{hdf5_name}_{args.rollout_name}.mp4"
        
        create_rollout_video(data, save_path=output_path, fps=args.fps)
        
    except Exception as e:
        print(f"❌ Failed to create video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 