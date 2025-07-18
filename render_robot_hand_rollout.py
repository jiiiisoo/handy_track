#!/usr/bin/env python3
"""
HDF5 rollout 데이터로 실제 로봇 손 3D 애니메이션을 생성하는 스크립트
render_retarget_multi.py를 참고하여 작성
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
from main.dataset.mano2dexhand_gigahands import Mano2DexhandGigaHands
# 기존 모듈들 import
from main.dataset.mano2dexhand import Mano2Dexhand
from main.dataset.transform import rot6d_to_rotmat, aa_to_rotmat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from isaacgym import gymutil


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
        
        # base_state 분해
        wrist_pos_seq = base_state[:, :3]      # 위치 (T, 3)
        wrist_quat_seq = base_state[:, 3:7]    # 쿼터니언 (T, 4)
        
        # 쿼터니언을 axis-angle로 변환
        wrist_rot_seq = []
        for quat in wrist_quat_seq:
            axis_angle = quaternion_to_axis_angle(quat)
            wrist_rot_seq.append(axis_angle)
        wrist_rot_seq = np.array(wrist_rot_seq)
        
        return {
            'q': q,                            # (T, 12)
            'wrist_pos': wrist_pos_seq,        # (T, 3)
            'wrist_rot': wrist_rot_seq,        # (T, 3)
            'rollout_name': rollout_name,
            'episode_length': len(q)
        }

def expand_dof_to_inspire_hand(dof_12):
    """12차원 DOF를 inspire hand의 20차원으로 확장"""
    dof_20 = np.zeros(20)
    dof_20[:12] = dof_12
    return dof_20

def setup_hand_chain(dexhand_type='inspire'):
    """로봇 손 kinematics chain 설정"""
    print(f"🤖 Setting up {dexhand_type} hand kinematics...")
    _parser = gymutil.parse_arguments(
        description="GigaHands MANO to Dexhand",
        headless=True,
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 7000,
            },
            {
                "name": "--data_idx",
                "type": str,
                "default": "20aed@0",
                "help": "Single sequence ID (e.g., '20aed@0') or 'all' for batch processing"
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "inspire",
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
            },
            {
                "name": "--data_dir",
                "type": str,
                "default": "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands",
                "help": "Path to GigaHands dataset directory containing annotations_v2.jsonl and handpose/"
            },
            {
                "name": "--save_dir",
                "type": str,
                "default": "/workspace/ManipTrans/visualize_results",
                "help": "Directory to save retargeted data"
            },
            {
                "name": "--skip_existing",
                "action": "store_true",
                "help": "Skip sequences that already have retargeted files"
            },
            {
                "name": "--max_sequences",
                "type": int,
                "default": -1,
                "help": "Maximum number of sequences to process (-1 for all)"
            },
            {"name": "--num_envs",
                "type": int,
                "default": 1,
                "help": "Number of environments to render"
            },
            {"name": "--fps",
                "type": int,
                "default": 30,
                "help": "FPS for output videos"
            },
            {"name": "--retargeted_data_path",
                "type": str,
                "default": "/scratch2/jisoo6687/gigahands/retargeted/mano2inspire_rh/p001-folder/keypoints_3d_mano/000_retargeted.pkl",
                "help": "Path to retargeted data pickle file"
            },
            {"name": "--smooth",
                "action": "store_true",
                "help": "Apply temporal smoothing to the motion data"
            },
            {"name": "--smooth_sigma",
                "type": float,
                "default": 1.5,
                "help": "Smoothing strength (higher = more smoothing)"
            },
            {"name": "--smooth_method",
                "type": str,
                "default": "gaussian",
                "choices": ["gaussian", "moving_average", "savgol"],
                "help": "Smoothing method to use"
            }
        ],
    )

    
    # DexHand factory 등록
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
    dexhand_module_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    hands_dir = os.path.join(dexhand_module_dir, "maniptrans_envs", "lib", "envs", "dexhands")
    DexHandFactory.auto_register_hands(hands_dir, "maniptrans_envs.lib.envs.dexhands")
    
    # Hand 생성
    dexhand = DexHandFactory.create_hand(dexhand_type, side="right")
    
    # Mano2Dexhand 변환기 설정
    mano2inspire = Mano2DexhandGigaHands(
        dexhand_name=dexhand_type,
        device='cpu',  # CPU 사용 (더 안정적)
        use_pca=False
    )
    
    chain = mano2inspire.chain
    joint_names = dexhand.body_names
    
    print(f"✅ Successfully set up {dexhand_type} hand")
    print(f"   DOFs: {dexhand.n_dofs}")
    print(f"   Joints: {len(joint_names)}")
    
    return dexhand, chain, joint_names

def compute_hand_keypoints(wrist_pos, wrist_rot, dof_pos, chain, joint_names, device='cpu'):
    """주어진 pose에서 손 키포인트 계산"""
    
    # 텐서로 변환
    wrist_pos = torch.tensor(wrist_pos, dtype=torch.float32, device=device)
    wrist_rot = torch.tensor(wrist_rot, dtype=torch.float32, device=device)
    dof_pos = torch.tensor(dof_pos, dtype=torch.float32, device=device)
    
    # 회전 행렬 계산 (axis-angle -> rotation matrix)
    if wrist_rot.shape[-1] == 6:
        wrist_rotmat = rot6d_to_rotmat(wrist_rot.unsqueeze(0))[0]
    else:
        wrist_rotmat = aa_to_rotmat(wrist_rot.unsqueeze(0))[0]
    
    # Forward Kinematics 계산
    with torch.no_grad():
        fk_result = chain.forward_kinematics(dof_pos.unsqueeze(0))
        joint_positions = torch.stack(
            [fk_result[k].get_matrix()[0, :3, 3] for k in joint_names], dim=0
        )
        joint_positions = joint_positions.to(device)
        
        # 손목 변환 적용
        joint_positions = (wrist_rotmat @ joint_positions.T).T + wrist_pos
    
    return joint_positions.cpu().numpy()

def create_finger_connections(joint_names):
    """손가락별 관절 연결 정보 생성"""
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
        if 'thumb' in name.lower():
            finger_connections['thumb'].append(i)
        elif 'index' in name.lower():
            finger_connections['index'].append(i)
        elif 'middle' in name.lower():
            finger_connections['middle'].append(i)
        elif 'ring' in name.lower():
            finger_connections['ring'].append(i)
        elif 'pinky' in name.lower():
            finger_connections['pinky'].append(i)
        else:
            finger_connections['wrist'].append(i)
    
    return finger_connections

def get_finger_colors():
    """손가락별 색상 정의"""
    return {
        'thumb': 'red',
        'index': 'blue', 
        'middle': 'green',
        'ring': 'orange',
        'pinky': 'purple',
        'wrist': 'black'
    }

def render_hand_frame(joint_positions, finger_connections, finger_colors, joint_names, 
                     elev=30, azim=45, title="Robot Hand"):
    """단일 프레임 손 렌더링"""
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # 손가락별로 관절 연결
    for finger, indices in finger_connections.items():
        if len(indices) > 0:
            color = finger_colors[finger]
            
            # 관절 점들 그리기
            finger_points = joint_positions[indices]
            ax.scatter(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2], 
                      c=color, s=80, label=finger, alpha=0.9, edgecolors='black', linewidths=0.5)
            
            # 손가락 관절들을 순서대로 연결 (wrist 제외)
            if finger != 'wrist' and len(indices) > 1:
                # 관절명에 따라 순서 정렬
                sorted_indices = []
                for joint_type in ['proximal', 'intermediate', 'tip', 'mcp', 'pip', 'dip']:
                    for idx in indices:
                        if joint_type in joint_names[idx].lower():
                            sorted_indices.append(idx)
                
                # 정렬된 순서대로 연결
                if len(sorted_indices) > 1:
                    sorted_points = joint_positions[sorted_indices]
                    ax.plot(sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2], 
                           c=color, linewidth=3, alpha=0.8)
                else:
                    # 정렬 실패시 순서대로 연결
                    finger_points = joint_positions[indices]
                    ax.plot(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2], 
                           c=color, linewidth=3, alpha=0.8)
            
            # 손목에서 각 손가락 첫 관절로 연결
            if finger != 'wrist' and len(finger_connections['wrist']) > 0:
                wrist_idx = finger_connections['wrist'][0]
                if len(indices) > 0:
                    first_joint_idx = indices[0]
                    # proximal 관절 찾기
                    for idx in indices:
                        if 'proximal' in joint_names[idx].lower() or 'mcp' in joint_names[idx].lower():
                            first_joint_idx = idx
                            break
                    
                    wrist_pos = joint_positions[wrist_idx]
                    first_joint_pos = joint_positions[first_joint_idx]
                    ax.plot([wrist_pos[0], first_joint_pos[0]], 
                           [wrist_pos[1], first_joint_pos[1]], 
                           [wrist_pos[2], first_joint_pos[2]], 
                           c=color, linewidth=2, alpha=0.6, linestyle='--')
    
    # 축 설정
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    
    # 렌더링 범위 자동 설정
    max_range = np.ptp(joint_positions, axis=0).max() * 0.6
    center = joint_positions.mean(axis=0)
    for axis, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        axis([c - max_range, c + max_range])
    
    # 범례 추가
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return fig, ax

def create_robot_hand_video(data, dexhand, chain, joint_names, save_path="robot_hand.mp4", fps=10):
    """로봇 손 애니메이션 비디오 생성"""
    
    q = data['q']
    wrist_pos = data['wrist_pos']
    wrist_rot = data['wrist_rot']
    episode_length = data['episode_length']
    
    print(f"🎬 Creating robot hand video...")
    print(f"   Frames: {episode_length}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {episode_length/fps:.1f} seconds")
    
    # 손가락 연결 정보
    finger_connections = create_finger_connections(joint_names)
    finger_colors = get_finger_colors()
    
    # 모든 프레임의 키포인트 미리 계산
    print("   Computing keypoints for all frames...")
    all_keypoints = []
    
    device = 'cpu'
    for frame_idx in range(episode_length):
        # DOF 확장 (12 -> 20)
        dof_20 = expand_dof_to_inspire_hand(q[frame_idx])
        
        # 키포인트 계산
        keypoints = compute_hand_keypoints(
            wrist_pos[frame_idx], 
            wrist_rot[frame_idx], 
            dof_20, 
            chain, 
            joint_names, 
            device
        )
        all_keypoints.append(keypoints)
    
    all_keypoints = np.array(all_keypoints)
    
    # 전체 범위 계산 (일관된 축 범위를 위해)
    all_points = all_keypoints.reshape(-1, 3)
    max_range = np.ptp(all_points, axis=0).max() * 0.7
    center = all_points.mean(axis=0)
    
    print("   Creating animation...")
    
    # Figure 설정
    fig = plt.figure(figsize=(12, 10))
    
    def animate(frame_idx):
        fig.clear()
        
        # 2개 뷰 생성
        views = [(30, 45), (30, -45)]
        
        for view_idx, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(1, 2, view_idx+1, projection="3d")
            
            joint_positions = all_keypoints[frame_idx]
            
            # 손가락별로 관절 연결
            for finger, indices in finger_connections.items():
                if len(indices) > 0:
                    color = finger_colors[finger]
                    
                    # 관절 점들 그리기
                    finger_points = joint_positions[indices]
                    ax.scatter(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2], 
                              c=color, s=80, alpha=0.9, edgecolors='black', linewidths=0.5)
                    
                    # 손가락 관절들을 순서대로 연결
                    if finger != 'wrist' and len(indices) > 1:
                        sorted_indices = []
                        for joint_type in ['proximal', 'intermediate', 'tip', 'mcp', 'pip', 'dip']:
                            for idx in indices:
                                if joint_type in joint_names[idx].lower():
                                    sorted_indices.append(idx)
                        
                        if len(sorted_indices) > 1:
                            sorted_points = joint_positions[sorted_indices]
                            ax.plot(sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2], 
                                   c=color, linewidth=3, alpha=0.8)
                        else:
                            finger_points = joint_positions[indices]
                            ax.plot(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2], 
                                   c=color, linewidth=3, alpha=0.8)
                    
                    # 손목에서 각 손가락으로 연결
                    if finger != 'wrist' and len(finger_connections['wrist']) > 0:
                        wrist_idx = finger_connections['wrist'][0]
                        if len(indices) > 0:
                            first_joint_idx = indices[0]
                            for idx in indices:
                                if 'proximal' in joint_names[idx].lower() or 'mcp' in joint_names[idx].lower():
                                    first_joint_idx = idx
                                    break
                            
                            wrist_pos_frame = joint_positions[wrist_idx]
                            first_joint_pos = joint_positions[first_joint_idx]
                            ax.plot([wrist_pos_frame[0], first_joint_pos[0]], 
                                   [wrist_pos_frame[1], first_joint_pos[1]], 
                                   [wrist_pos_frame[2], first_joint_pos[2]], 
                                   c=color, linewidth=2, alpha=0.6, linestyle='--')
            
            # 축 설정
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(f"{data['rollout_name']} - View {view_idx+1}\nFrame {frame_idx+1}/{episode_length}")
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
    parser = argparse.ArgumentParser(description='Render robot hand videos from HDF5 rollout data')
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
    dexhand, chain, joint_names = setup_hand_chain(args.dexhand_type)
    
    # 비디오 생성
    try:
        hdf5_name = Path(args.hdf5_path).stem
        output_path = f"{args.output_dir}/{hdf5_name}_{args.rollout_name}_robothand.mp4"
        
        create_robot_hand_video(data, dexhand, chain, joint_names, 
                              save_path=output_path, fps=args.fps)
        
    except Exception as e:
        print(f"❌ Failed to create robot hand video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 