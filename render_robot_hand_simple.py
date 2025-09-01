#!/usr/bin/env python3
"""
HDF5 rollout 데이터로 실제 로봇 손 3D 애니메이션을 생성하는 스크립트 (간단한 버전)
render_zero_pose_keypoints.py의 방식을 참고하여 작성
"""

# CRITICAL: Import isaacgym modules FIRST
import isaacgym
import torch
import h5py
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pytorch_kinematics as pk

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

def quaternion_to_axis_angle(quat):
    """Quaternion을 axis-angle로 변환"""
    quat = quat / np.linalg.norm(quat)
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
        
        print(f"📊 Data shapes:")
        print(f"   Joint angles (q): {q.shape}")
        print(f"   Base state: {base_state.shape}")
        print(f"   Episode length: {len(q)} steps")
        
        # base_state 분해: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        wrist_pos_seq = base_state[:, :3]      # 위치 (T, 3)
        wrist_quat_seq = base_state[:, 3:7]    # 쿼터니언 (T, 4)
        
        return {
            'q': q,                            # (T, 12)
            'wrist_pos': wrist_pos_seq,        # (T, 3)
            'wrist_quat': wrist_quat_seq,      # (T, 4)
            'rollout_name': rollout_name,
            'episode_length': len(q)
        }

def expand_dof_to_inspire_hand(dof_12, target_dofs):
    """12차원 DOF를 inspire hand의 실제 DOF 차원으로 확장"""
    if target_dofs <= 12:
        # 타겟이 12 이하면 그대로 사용하거나 잘라냄
        return dof_12[:target_dofs]
    else:
        # 타겟이 12보다 크면 0으로 채움
        expanded_dof = np.zeros(target_dofs)
        expanded_dof[:12] = dof_12
        return expanded_dof

def setup_hand_chain(dexhand_type='inspire'):
    """로봇 손 kinematics chain 설정 (간단한 방법)"""
    print(f"🤖 Setting up {dexhand_type} hand kinematics...")
    
    try:
        # DexHand factory 자동 등록
        hands_dir = os.path.join(os.path.dirname(__file__), "maniptrans_envs", "lib", "envs", "dexhands")
        DexHandFactory.auto_register_hands(hands_dir, "maniptrans_envs.lib.envs.dexhands")
        
        # Hand 생성
        dexhand = DexHandFactory.create_hand(dexhand_type, side="right")
        
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
        
        joint_names = dexhand.body_names
        
        # Chain의 실제 DOF 수 확인
        actual_dofs = chain.n_joints
        
        print(f"✅ Successfully set up {dexhand_type} hand")
        print(f"   DexHand DOFs: {dexhand.n_dofs}")
        print(f"   Chain DOFs: {actual_dofs}")
        print(f"   Joints: {len(joint_names)}")
        print(f"   URDF: {urdf_path}")
        
        return dexhand, chain, joint_names, actual_dofs
        
    except Exception as e:
        print(f"❌ Failed to setup hand chain: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def quat_to_rotmat(quat):
    """쿼터니언을 회전 행렬로 변환"""
    quat = quat / np.linalg.norm(quat)  # normalize
    w, x, y, z = quat
    
    # 회전 행렬 계산
    rotmat = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return rotmat

def compute_hand_keypoints(wrist_pos, wrist_quat, dof_pos, chain, joint_names, device='cpu'):
    """주어진 pose에서 손 키포인트 계산"""
    
    # 텐서로 변환
    wrist_pos = torch.tensor(wrist_pos, dtype=torch.float32, device=device)
    wrist_quat = torch.tensor(wrist_quat, dtype=torch.float32, device=device)
    dof_pos = torch.tensor(dof_pos, dtype=torch.float32, device=device)
    
    # 쿼터니언을 회전 행렬로 변환
    wrist_rotmat = torch.tensor(quat_to_rotmat(wrist_quat.cpu().numpy()), 
                               dtype=torch.float32, device=device)
    
    # DOF 차원 체크
    expected_dofs = chain.n_joints
    if len(dof_pos) != expected_dofs:
        print(f"   Warning: DOF mismatch. Expected {expected_dofs}, got {len(dof_pos)}")
        # 차원 맞추기
        if len(dof_pos) > expected_dofs:
            dof_pos = dof_pos[:expected_dofs]
        else:
            padded_dof = torch.zeros(expected_dofs, device=device)
            padded_dof[:len(dof_pos)] = dof_pos
            dof_pos = padded_dof
    
    # Forward Kinematics 계산
    with torch.no_grad():
        try:
            fk_result = chain.forward_kinematics(dof_pos.unsqueeze(0))
            
            # 모든 관절의 위치 추출
            joint_positions = []
            for joint_name in joint_names:
                if joint_name in fk_result:
                    # Transform matrix에서 위치 추출
                    transform_matrix = fk_result[joint_name].get_matrix()
                    position = transform_matrix[0, :3, 3]  # [x, y, z]
                    joint_positions.append(position)
                else:
                    # 해당 관절이 없으면 원점 사용
                    joint_positions.append(torch.zeros(3, device=device))
            
            joint_positions = torch.stack(joint_positions, dim=0)
            
            # 손목 변환 적용 (회전 + 이동)
            joint_positions = (wrist_rotmat @ joint_positions.T).T + wrist_pos
            
        except Exception as e:
            print(f"   FK error: {e}")
            # 에러 발생시 손목 위치만 반환
            joint_positions = wrist_pos.unsqueeze(0).repeat(len(joint_names), 1)
    
    return joint_positions.cpu().numpy()

def get_finger_color(joint_name):
    """손가락별 색상 정의"""
    joint_name = joint_name.lower()
    if "thumb" in joint_name:
        return "red"
    elif "index" in joint_name:
        return "blue" 
    elif "middle" in joint_name:
        return "green"
    elif "ring" in joint_name:
        return "orange"
    elif "pinky" in joint_name:
        return "purple"
    elif "wrist" in joint_name or "palm" in joint_name:
        return "black"
    else:
        return "gray"

def create_finger_connections(joint_names):
    """손가락별 관절 연결 정보 생성"""
    finger_groups = {}
    
    for i, name in enumerate(joint_names):
        for finger in ["thumb", "index", "middle", "ring", "pinky", "wrist", "palm"]:
            if finger in name.lower():
                if finger not in finger_groups:
                    finger_groups[finger] = []
                finger_groups[finger].append(i)
                break
    
    return finger_groups

def create_robot_hand_video(data, dexhand, chain, joint_names, actual_dofs, save_path="robot_hand.mp4", fps=8):
    """로봇 손 애니메이션 비디오 생성"""
    
    q = data['q']
    wrist_pos = data['wrist_pos']
    wrist_quat = data['wrist_quat']
    episode_length = data['episode_length']
    
    print(f"🎬 Creating robot hand video...")
    print(f"   Frames: {episode_length}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {episode_length/fps:.1f} seconds")
    print(f"   Using {actual_dofs} DOFs")
    
    # 손가락 연결 정보
    finger_groups = create_finger_connections(joint_names)
    
    # 모든 프레임의 키포인트 미리 계산
    print("   Computing keypoints for all frames...")
    all_keypoints = []
    
    device = 'cpu'
    for frame_idx in range(episode_length):
        # DOF 확장 (12 -> actual_dofs)
        dof_expanded = expand_dof_to_inspire_hand(q[frame_idx], actual_dofs)
        
        # 키포인트 계산
        try:
            keypoints = compute_hand_keypoints(
                wrist_pos[frame_idx], 
                wrist_quat[frame_idx], 
                dof_expanded, 
                chain, 
                joint_names, 
                device
            )
            all_keypoints.append(keypoints)
        except Exception as e:
            print(f"   Warning: Failed to compute keypoints for frame {frame_idx}: {e}")
            # 이전 프레임 복사 또는 zero pose 사용
            if len(all_keypoints) > 0:
                all_keypoints.append(all_keypoints[-1])
            else:
                # Zero pose 계산
                zero_dof = np.zeros(actual_dofs)
                try:
                    keypoints = compute_hand_keypoints(
                        wrist_pos[frame_idx], 
                        wrist_quat[frame_idx], 
                        zero_dof, 
                        chain, 
                        joint_names, 
                        device
                    )
                    all_keypoints.append(keypoints)
                except:
                    # 최후의 수단: 손목 위치만 사용
                    fallback_keypoints = np.tile(wrist_pos[frame_idx], (len(joint_names), 1))
                    all_keypoints.append(fallback_keypoints)
    
    all_keypoints = np.array(all_keypoints)
    
    # 전체 범위 계산 (일관된 축 범위를 위해)
    all_points = all_keypoints.reshape(-1, 3)
    max_range = np.ptp(all_points, axis=0).max() * 0.7
    center = all_points.mean(axis=0)
    
    print("   Creating animation...")
    
    # Figure 설정
    fig = plt.figure(figsize=(15, 6))
    
    def animate(frame_idx):
        fig.clear()
        
        # 2개 뷰 생성
        views = [(30, 45), (0, 90)]
        view_titles = ["Side View", "Top View"]
        
        for view_idx, ((elev, azim), title) in enumerate(zip(views, view_titles)):
            ax = fig.add_subplot(1, 2, view_idx+1, projection="3d")
            
            joint_positions = all_keypoints[frame_idx]
            
            # 모든 관절 점들 그리기
            for i, (keypoint, name) in enumerate(zip(joint_positions, joint_names)):
                color = get_finger_color(name)
                ax.scatter(keypoint[0], keypoint[1], keypoint[2], 
                          c=color, s=80, alpha=0.9, edgecolors='black', linewidths=0.5)
                
                # 관절 번호 표시 (일부만)
                if i % 3 == 0:  # 3개 중 1개만 표시
                    ax.text(keypoint[0], keypoint[1], keypoint[2], f'{i}', 
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
            
            # 손가락별 연결선 그리기
            for finger, indices in finger_groups.items():
                if len(indices) > 1:
                    color = get_finger_color(finger)
                    finger_points = joint_positions[indices]
                    
                    # 손가락 관절들을 순서대로 연결
                    for i in range(len(finger_points) - 1):
                        ax.plot([finger_points[i, 0], finger_points[i+1, 0]], 
                               [finger_points[i, 1], finger_points[i+1, 1]], 
                               [finger_points[i, 2], finger_points[i+1, 2]], 
                               c=color, linewidth=2, alpha=0.8)
            
            # 손목에서 각 손가락 첫 관절로 연결 (만약 손목이 있다면)
            if 'wrist' in finger_groups or 'palm' in finger_groups:
                wrist_indices = finger_groups.get('wrist', []) + finger_groups.get('palm', [])
                if len(wrist_indices) > 0:
                    wrist_pos_frame = joint_positions[wrist_indices[0]]
                    
                    for finger, indices in finger_groups.items():
                        if finger not in ['wrist', 'palm'] and len(indices) > 0:
                            first_joint_pos = joint_positions[indices[0]]
                            color = get_finger_color(finger)
                            ax.plot([wrist_pos_frame[0], first_joint_pos[0]], 
                                   [wrist_pos_frame[1], first_joint_pos[1]], 
                                   [wrist_pos_frame[2], first_joint_pos[2]], 
                                   c=color, linewidth=1.5, alpha=0.6, linestyle='--')
            
            # 축 설정
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(f"{data['rollout_name']} - {title}\nFrame {frame_idx+1}/{episode_length}")
            ax.view_init(elev=elev, azim=azim)
            
            # 일관된 축 범위 설정
            for axis, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
                axis([c - max_range, c + max_range])
        
        plt.tight_layout()
    
    # 애니메이션 생성
    anim = animation.FuncAnimation(fig, animate, frames=episode_length, 
                                 interval=1000/fps, blit=False, repeat=True)
    
    # 비디오 저장
    print(f"   Saving video to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # GIF 저장
        gif_path = save_path.replace('.mp4', '.gif')
        print(f"   Saving as GIF: {gif_path}")
        anim.save(gif_path, writer='pillow', fps=fps)
        print(f"✅ GIF saved successfully!")
        
        # MP4 저장 시도
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"✅ MP4 video also saved!")
        except:
            print("   MP4 failed, but GIF succeeded")
            
    except Exception as e:
        print(f"❌ Video creation failed: {e}")
        
        # 정적 이미지들 저장
        print("   Saving static frames...")
        frames_dir = save_path.replace('.mp4', '_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        for i in range(0, episode_length, max(1, episode_length//10)):
            animate(i)
            plt.savefig(f"{frames_dir}/frame_{i:03d}.png", dpi=150, bbox_inches='tight')
        print(f"✅ Static frames saved to: {frames_dir}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Render robot hand videos from HDF5 rollout data (simple version)')
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='Path to HDF5 rollout file')
    parser.add_argument('--rollout_name', type=str, default="rollout_0",
                       help='Name of rollout to render')
    parser.add_argument('--output_dir', type=str, default="robot_hand_videos",
                       help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=8,
                       help='Video FPS')
    parser.add_argument('--dexhand_type', type=str, default='inspire',
                       help='DexHand type')
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
    
    # 손 kinematics 설정
    try:
        dexhand, chain, joint_names, actual_dofs = setup_hand_chain(args.dexhand_type)
        if dexhand is None:
            print("❌ Failed to setup hand kinematics")
            return
    except Exception as e:
        print(f"❌ Failed to setup hand: {e}")
        return
    
    # 비디오 생성
    try:
        hdf5_name = Path(args.hdf5_path).stem
        output_path = f"{args.output_dir}/{hdf5_name}_{args.rollout_name}_robothand_simple.mp4"
        
        create_robot_hand_video(data, dexhand, chain, joint_names, actual_dofs,
                              save_path=output_path, fps=args.fps)
        
    except Exception as e:
        print(f"❌ Failed to create robot hand video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 