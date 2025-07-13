#!/usr/bin/env python3
"""
MANO Multi-viewpoint Keypoint Renderer
입력으로 들어온 키포인트를 다양한 시점(위, 아래, 왼쪽, 오른쪽, 앞, 뒤)에서 렌더링
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from termcolor import cprint
import os
import torch
import argparse
import pickle
import glob

cprint("✅ Modules imported successfully", "green")


def get_mano_keypoint_names():
    """MANO 키포인트 이름 정의 (21개 키포인트)"""
    return [
        "wrist",           # 0
        "thumb_mcp",       # 1
        "thumb_pip",       # 2  
        "thumb_dip",       # 3
        "thumb_tip",       # 4
        "index_mcp",       # 5
        "index_pip",       # 6
        "index_dip",       # 7
        "index_tip",       # 8
        "middle_mcp",      # 9
        "middle_pip",      # 10
        "middle_dip",      # 11
        "middle_tip",      # 12
        "ring_mcp",        # 13
        "ring_pip",        # 14
        "ring_dip",        # 15
        "ring_tip",        # 16
        "pinky_mcp",       # 17
        "pinky_pip",       # 18
        "pinky_dip",       # 19
        "pinky_tip"        # 20
    ]


def get_finger_color(keypoint_name):
    """키포인트 이름으로부터 손가락별 색상 반환"""
    colors = {
        "wrist": "black",
        "thumb": "red",
        "index": "green", 
        "middle": "blue",
        "ring": "purple",
        "pinky": "orange"
    }
    
    keypoint_name = keypoint_name.lower()
    if "wrist" in keypoint_name or "palm" in keypoint_name:
        return colors["wrist"]
    elif "thumb" in keypoint_name:
        return colors["thumb"]
    elif "index" in keypoint_name:
        return colors["index"]
    elif "middle" in keypoint_name:
        return colors["middle"]
    elif "ring" in keypoint_name:
        return colors["ring"]
    elif "pinky" in keypoint_name:
        return colors["pinky"]
    else:
        return colors["wrist"]  # default


def get_mano_skeleton_connections():
    """MANO 키포인트 연결 구조 정의"""
    # MANO 21 keypoints 연결 구조
    connections = [
        # 손목에서 각 손가락 기저부로
        (0, 1),   # wrist -> thumb_mcp
        (0, 5),   # wrist -> index_mcp
        (0, 9),   # wrist -> middle_mcp
        (0, 13),  # wrist -> ring_mcp
        (0, 17),  # wrist -> pinky_mcp
        
        # 엄지손가락
        (1, 2),   # thumb_mcp -> thumb_pip
        (2, 3),   # thumb_pip -> thumb_dip
        (3, 4),   # thumb_dip -> thumb_tip
        
        # 검지
        (5, 6),   # index_mcp -> index_pip
        (6, 7),   # index_pip -> index_dip
        (7, 8),   # index_dip -> index_tip
        
        # 중지
        (9, 10),  # middle_mcp -> middle_pip
        (10, 11), # middle_pip -> middle_dip
        (11, 12), # middle_dip -> middle_tip
        
        # 약지
        (13, 14), # ring_mcp -> ring_pip
        (14, 15), # ring_pip -> ring_dip
        (15, 16), # ring_dip -> ring_tip
        
        # 새끼손가락
        (17, 18), # pinky_mcp -> pinky_pip
        (18, 19), # pinky_pip -> pinky_dip
        (19, 20), # pinky_dip -> pinky_tip
    ]
    return connections


def load_keypoints_from_file(file_path):
    """파일에서 키포인트 로드"""
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # pkl 파일 구조에 따라 키포인트 추출
            if isinstance(data, dict):
                if 'keypoints' in data:
                    keypoints = data['keypoints']
                elif 'keypoints_3d' in data:
                    keypoints = data['keypoints_3d']
                else:
                    # 첫 번째 키를 사용
                    keypoints = list(data.values())[0]
            else:
                keypoints = data
    elif file_path.endswith('.npy'):
        keypoints = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # 키포인트를 numpy array로 변환하고 shape 확인
    keypoints = np.array(keypoints)
    if len(keypoints.shape) == 3:  # [batch, num_keypoints, 3]
        keypoints = keypoints[0]  # 첫 번째 배치 사용
    elif len(keypoints.shape) == 2:  # [num_keypoints, 3]
        pass
    else:
        raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")
    
    return keypoints


def load_sequence_keypoints_from_file(file_path):
    """시퀀스 형태의 pkl 파일에서 키포인트 로드"""
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(data['opt_joints_pos'].shape)
            
            # 시퀀스 데이터 처리
            if isinstance(data, dict):
                if 'keypoints' in data:
                    keypoints = data['keypoints']
                elif 'keypoints_3d' in data:
                    keypoints = data['keypoints_3d']
                else:
                    # 모든 값을 시퀀스로 처리
                    keypoints = list(data.values())
            else:
                keypoints = data
            
            # numpy array로 변환
            keypoints = np.array(keypoints)
            
            # shape 확인 및 조정
            if len(keypoints.shape) == 3:  # [sequence_length, num_keypoints, 3]
                return keypoints
            elif len(keypoints.shape) == 4:  # [batch, sequence_length, num_keypoints, 3]
                return keypoints[0]  # 첫 번째 배치 사용
            else:
                raise ValueError(f"Unexpected sequence keypoint shape: {keypoints.shape}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def render_keypoints_multi_viewpoint(keypoints, save_path=None, title="Hand Keypoints"):
    """
    입력으로 받은 키포인트를 다양한 시점에서 렌더링
    
    Args:
        keypoints: numpy array of keypoint positions [N, 3]
        save_path: 저장할 파일 경로 (None이면 기본 경로 사용)
        title: 그래프 제목
    
    Returns:
        keypoint_names: list of keypoint names
    """
    
    cprint("🎥 Rendering keypoints multi-viewpoint...", "cyan")
    
    keypoint_names = get_mano_keypoint_names()
    
    # 키포인트 개수 확인 및 조정
    if len(keypoints) != len(keypoint_names):
        cprint(f"⚠️  Warning: Expected {len(keypoint_names)} keypoints, got {len(keypoints)}", "yellow")
        # 키포인트 개수에 맞게 이름 조정
        if len(keypoints) < len(keypoint_names):
            keypoint_names = keypoint_names[:len(keypoints)]
        else:
            # 추가 키포인트에 대해 이름 생성
            for i in range(len(keypoint_names), len(keypoints)):
                keypoint_names.append(f"keypoint_{i}")
    
    # 다양한 시점 정의 (elevation, azimuth)
    viewpoints = {
        "Front": (0, 0),      # 정면
        "Back": (0, 180),     # 후면
        "Left": (0, 90),      # 왼쪽
        "Right": (0, -90),    # 오른쪽
        "Top": (90, 0),       # 위쪽
        "Bottom": (-90, 0),   # 아래쪽
    }
    
    # 6개 시점으로 시각화
    fig = plt.figure(figsize=(18, 12))
    
    for i, (view_name, (elev, azim)) in enumerate(viewpoints.items()):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.set_title(f'{title} - {view_name} View\n(elev={elev}°, azim={azim}°)', 
                    fontsize=12, fontweight='bold')
        
        # 키포인트 강조 표시
        for j, (keypoint, name) in enumerate(zip(keypoints, keypoint_names)):
            color = get_finger_color(name)
            ax.scatter(keypoint[0], keypoint[1], keypoint[2], 
                      c=color, s=100, alpha=1.0, edgecolors='black', linewidth=1)
            
            # 텍스트 라벨 (키포인트 번호)
            ax.text(keypoint[0], keypoint[1], keypoint[2], f'{j}', 
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
        
        # MANO 키포인트 연결선 추가
        connections = get_mano_skeleton_connections()
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_pos = keypoints[start_idx]
                end_pos = keypoints[end_idx]
                start_finger_color = get_finger_color(keypoint_names[start_idx])
                ax.plot([start_pos[0], end_pos[0]], 
                       [start_pos[1], end_pos[1]], 
                       [start_pos[2], end_pos[2]], 
                       start_finger_color, alpha=0.8, linewidth=2)
        
        # 시점 설정
        ax.view_init(elev=elev, azim=azim)
        
        # 축 범위 동일하게 설정
        if len(keypoints) > 0:
            all_coords = keypoints.flatten()
            coord_range = max(all_coords.max() - all_coords.min(), 0.1)
            center = [(keypoints[:, i].max() + keypoints[:, i].min()) / 2 for i in range(3)]
            
            for axis, c in zip(['x', 'y', 'z'], center):
                getattr(ax, f'set_{axis}lim')(c - coord_range/2, c + coord_range/2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    if save_path is None:
        save_path = "/workspace/ManipTrans/mano_multi_viewpoint_keypoints.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    cprint(f"✅ MANO multi-viewpoint visualization saved: {save_path}", "green")
    
    plt.close()
    
    return keypoints, keypoint_names


def render_mano_different_poses_multiview(save_path=None):
    """
    여러 다른 포즈의 MANO 손을 멀티뷰포인트로 렌더링
    
    Args:
        save_path: 저장할 파일 경로 (None이면 기본 경로 사용)
    
    Returns:
        fig: matplotlib figure 객체
    """
    
    cprint("🎥 Rendering MANO different poses in multi-viewpoint...", "cyan")
    
    # 다양한 포즈 정의
    poses = {
        "Zero Pose": torch.zeros(1, 9),  # 모든 관절 0도
        "Semi Fist": torch.ones(1, 9) * 0.5,  # 반 주먹
        "Full Fist": torch.ones(1, 9) * 1.0,  # 완전 주먹
        "Peace Sign": torch.tensor([[0, 0, 0, -0.5, -0.5, 0.8, 0.8, 0.8, 0.8]]),  # 브이 사인
    }
    
    # 다양한 시점 정의
    viewpoints = {
        "Front": (0, 0),      # 정면
        "Right": (0, -90),    # 오른쪽
        "Top": (90, 0),       # 위쪽
    }
    
    # 포즈 x 시점으로 시각화
    fig = plt.figure(figsize=(15, 20))
    
    for i, (pose_name, pose_tensor) in enumerate(poses.items()):
        # 각 포즈에 대해 MANO 키포인트 계산
        # vertices, keypoints, mano_layer = get_mano_keypoints(pose=pose_tensor)
        keypoints = load_sequence_keypoints_from_file(args.pose)
        keypoint_names = get_mano_keypoint_names()
        
        for j, (view_name, (elev, azim)) in enumerate(viewpoints.items()):
            subplot_idx = i * 3 + j + 1
            ax = fig.add_subplot(4, 3, subplot_idx, projection='3d')
            ax.set_title(f'MANO Hand - {pose_name}\n{view_name} View (elev={elev}°, azim={azim}°)', 
                        fontsize=10, fontweight='bold')
            
            # 키포인트만 렌더링 (메쉬 제거)
            
            # 키포인트 강조 표시
            for k, (keypoint, name) in enumerate(zip(keypoints, keypoint_names)):
                color = get_finger_color(name)
                ax.scatter(keypoint[0], keypoint[1], keypoint[2], 
                          c=color, s=60, alpha=1.0, edgecolors='black', linewidth=0.5)
            
            # 연결선 추가
            connections = get_mano_skeleton_connections()
            for start_idx, end_idx in connections:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_pos = keypoints[start_idx]
                    end_pos = keypoints[end_idx]
                    start_finger_color = get_finger_color(keypoint_names[start_idx])
                    ax.plot([start_pos[0], end_pos[0]], 
                           [start_pos[1], end_pos[1]], 
                           [start_pos[2], end_pos[2]], 
                           start_finger_color, alpha=0.7, linewidth=1.5)
            
            # 시점 설정
            ax.view_init(elev=elev, azim=azim)
            
            # 축 범위 설정
            if len(keypoints) > 0:
                all_coords = keypoints.flatten()
                coord_range = max(all_coords.max() - all_coords.min(), 0.1)
                center = [(keypoints[:, i].max() + keypoints[:, i].min()) / 2 for i in range(3)]
                
                for axis, c in zip(['x', 'y', 'z'], center):
                    getattr(ax, f'set_{axis}lim')(c - coord_range/2, c + coord_range/2)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    if save_path is None:
        save_path = "/workspace/ManipTrans/mano_different_poses_multiview_2.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    cprint(f"✅ MANO different poses multi-viewpoint saved: {save_path}", "green")
    
    plt.close()
    
    return fig


def main():
    """메인 실행 함수"""
    
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="MANO Multi-viewpoint Keypoint Renderer")
    parser.add_argument("--mode", type=str, default="single", 
                       choices=["single", "poses"],
                       help="Rendering mode: single pose or multiple poses")
    parser.add_argument("--save_dir", type=str, default="/workspace/ManipTrans",
                       help="Directory to save output images")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for computation")
    parser.add_argument("--pose", type=str, default="/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/retargeted/mano2inspire_rh/p001-folder/keypoints_3d/001/retargeted.pkl")
    
    args = parser.parse_args()
    
    cprint("🚀 Starting MANO Multi-viewpoint Keypoint Rendering...", "cyan", attrs=['bold'])
    
    if args.mode == "single":
        # 단일 포즈 멀티뷰포인트 렌더링
        cprint("🎯 Rendering single pose multi-viewpoint...", "cyan", attrs=['bold'])
        
        save_path = os.path.join(args.save_dir, "mano_single_pose_multiview.png")
        
        keypoints = load_sequence_keypoints_from_file(args.pose)
        print(len(keypoints))
        1/0
        render_keypoints_multi_viewpoint(keypoints, save_path=save_path)
        
        cprint(f"✅ Single pose multi-viewpoint rendering completed!", "green", attrs=['bold'])
        cprint(f"📊 Generated {len(keypoints)} keypoints from 6 different views", "white")
    
    elif args.mode == "poses":
        # 여러 포즈 멀티뷰포인트 렌더링
        cprint("🎯 Rendering multiple poses multi-viewpoint...", "cyan", attrs=['bold'])
        
        save_path = os.path.join(args.save_dir, "mano_multiple_poses_multiview.png")
        
        fig = render_mano_different_poses_multiview(save_path=save_path)
        
        cprint(f"✅ Multiple poses multi-viewpoint rendering completed!", "green", attrs=['bold'])
        cprint(f"📊 Generated 4 different poses from 3 different views each", "white")


if __name__ == "__main__":
    main() 