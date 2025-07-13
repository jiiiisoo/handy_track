#!/usr/bin/env python3
"""
MANO Multi-viewpoint Keypoint Renderer
MANO 손의 키포인트를 다양한 시점(위, 아래, 왼쪽, 오른쪽, 앞, 뒤)에서 렌더링
"""

import sys
from isaacgym import gymapi, gymtorch, gymutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from termcolor import cprint
import os
import torch
import argparse

# MANO 관련 import
from manopth.manolayer import ManoLayer
from manopth import demo
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

cprint("✅ MANO modules imported successfully", "green")


def get_mano_keypoints(pose=None, shape=None, use_pca=True, ncomps=6):
    """MANO 모델에서 키포인트들과 메쉬 추출"""
    
    cprint("🤚 Extracting MANO keypoints and mesh...", "cyan")
    
    # MANO 모델 로드
    mano_layer = ManoLayer(
        mano_root='/workspace/manopth/mano/models', 
        use_pca=use_pca, 
        ncomps=ncomps, 
        flat_hand_mean=True
    )
    cprint("✅ MANO model loaded", "green")
    
    # 기본 파라미터 설정
    batch_size = 1
    if pose is None:
        pose = torch.zeros(batch_size, ncomps + 3)  # 모든 pose 파라미터 0
    if shape is None:
        shape = torch.zeros(batch_size, 10)  # 모든 shape 파라미터 0
    
    # Forward pass
    with torch.no_grad():
        hand_verts, hand_joints, transform_abs = mano_layer(pose, shape)
    
    # NumPy로 변환
    vertices = hand_verts[0].numpy()  # [778, 3]
    keypoints = hand_joints[0].numpy()  # [21, 3]  
    wrist_pos = keypoints[0]
    middle_pos = keypoints[9]
    wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # ? hack for wrist position
    dexhand = DexHandFactory.create_hand(dexhand_type="inspire", side="right")
    wrist_pos += np.array(dexhand.relative_translation)
    mano_rot_offset = dexhand.relative_rotation
    wrist_rot = transform_abs[:, 0, :3, :3].detach() @ np.repeat(mano_rot_offset[None], transform_abs.shape[0], axis=0)

    # joints_rel = keypoints - keypoints[0]  # wrist 기준 상대 좌표
    # joints_in_dexhand = (wrist_rot[0] @ joints_rel.T).T + wrist_pos  # (21, 3)
    print(keypoints.shape, wrist_pos.shape)
    keypoints[0] = wrist_pos
    torch.save(transform_abs, "/workspace/ManipTrans/transform_abs.pt")
    cprint("✅ transform_abs saved to /workspace/ManipTrans/transform_abs.pt", "yellow")
    
    
    cprint(f"✅ MANO: {vertices.shape[0]} vertices, {keypoints.shape[0]} keypoints", "green")
    
    return vertices, keypoints, mano_layer


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


def render_mano_multi_viewpoint_keypoints(pose=None, shape=None, save_path=None, device="cpu"):
    """
    MANO 키포인트를 다양한 시점(위, 아래, 왼쪽, 오른쪽, 앞, 뒤)에서 렌더링
    
    Args:
        pose: MANO pose 파라미터 (None이면 zero pose)
        shape: MANO shape 파라미터 (None이면 기본 shape)
        save_path: 저장할 파일 경로 (None이면 기본 경로 사용)
        device: 계산에 사용할 device
    
    Returns:
        vertices: numpy array of mesh vertices [N, 3]
        keypoints: numpy array of keypoint positions [21, 3]
        keypoint_names: list of keypoint names
    """
    
    cprint("🎥 Rendering MANO multi-viewpoint keypoints...", "cyan")
    
    # MANO 키포인트 및 메쉬 추출
    vertices, keypoints, mano_layer = get_mano_keypoints(pose, shape)
    keypoint_names = get_mano_keypoint_names()
    
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
        ax.set_title(f'MANO Hand - {view_name} View\n(elev={elev}°, azim={azim}°)', 
                    fontsize=12, fontweight='bold')
        
        # 키포인트만 렌더링 (메쉬 제거)
        
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
    
    return vertices, keypoints, keypoint_names


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
        vertices, keypoints, mano_layer = get_mano_keypoints(pose=pose_tensor)
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
    
    args = parser.parse_args()
    
    cprint("🚀 Starting MANO Multi-viewpoint Keypoint Rendering...", "cyan", attrs=['bold'])
    
    if args.mode == "single":
        # 단일 포즈 멀티뷰포인트 렌더링
        cprint("🎯 Rendering single pose multi-viewpoint...", "cyan", attrs=['bold'])
        
        save_path = os.path.join(args.save_dir, "mano_single_pose_multiview.png")
        
        vertices, keypoints, keypoint_names = render_mano_multi_viewpoint_keypoints(
            save_path=save_path,
            device=args.device
        )
        
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