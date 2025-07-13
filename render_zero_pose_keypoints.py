#!/usr/bin/env python3
"""
Zero Pose Keypoint Renderer
DOF를 모두 0으로 설정한 zero pose에서 forward kinematics로 키포인트를 계산하고 렌더링
"""

# CRITICAL: Import isaacgym modules FIRST before any other imports
from isaacgym import gymapi, gymtorch, gymutil
import logging
logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import os
import numpy as np
import torch
import pytorch_kinematics as pk
from termcolor import cprint
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for rendering

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def get_finger_color(joint_name):
    """손가락별 색상 정의"""
    joint_name = joint_name.lower()
    if "thumb" in joint_name:
        return "red"
    elif "index" in joint_name:
        return "green" 
    elif "middle" in joint_name:
        return "blue"
    elif "ring" in joint_name:
        return "purple"
    elif "pinky" in joint_name:
        return "orange"
    elif "wrist" in joint_name:
        return "black"
    elif "palm" in joint_name:
        return "cyan"
    else:
        return "gray"


def render_zero_pose_keypoints(dexhand, save_path=None, device="cpu"):
    """
    DOF를 모두 0으로 설정한 zero pose에서 forward kinematics로 키포인트를 계산하고 렌더링
    
    Args:
        dexhand: DexHand 객체
        save_path: 저장할 파일 경로 (None이면 기본 경로 사용)
        device: 계산에 사용할 device
    
    Returns:
        keypoints: numpy array of keypoint positions [N, 3]
        keypoint_names: list of keypoint names
    """
    
    cprint("🤖 Rendering zero pose keypoints using forward kinematics...", "cyan")
    
    # URDF에서 kinematic chain 생성
    asset_root = os.path.split(dexhand.urdf_path)[0]
    asset_file = os.path.split(dexhand.urdf_path)[1]
    urdf_path = os.path.join(asset_root, asset_file)
    
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # PyTorch Kinematics chain 생성
    chain = pk.build_chain_from_urdf(urdf_content)
    chain = chain.to(dtype=torch.float32, device=device)
    
    cprint(f"✅ Built kinematic chain for {dexhand.name} hand", "green")
    cprint(f"📊 DOFs: {dexhand.n_dofs}", "cyan")
    cprint(f"📊 Bodies: {len(dexhand.body_names)}", "cyan")
    
    # Zero pose (모든 DOF = 0)
    batch_size = 1
    zero_dof_pos = torch.zeros(batch_size, dexhand.n_dofs, dtype=torch.float32, device=device)
    
    cprint(f"🔢 Zero DOF pose shape: {zero_dof_pos.shape}", "blue")
    cprint(f"🔢 Zero DOF values: {zero_dof_pos.flatten()[:10].tolist()}{'...' if dexhand.n_dofs > 10 else ''}", "blue")
    
    # Forward kinematics 계산
    with torch.no_grad():
        ret = chain.forward_kinematics(zero_dof_pos)
    
    cprint(f"✅ Forward kinematics completed, found {len(ret)} bodies", "green")
    
    # 키포인트 추출
    keypoints = []
    keypoint_names = []
    
    for i, body_name in enumerate(dexhand.body_names):
        hand_joint_name = dexhand.to_hand(body_name)[0]
        keypoint_names.append(hand_joint_name)
        
        if body_name in ret:
            # Transform matrix에서 위치 추출 [batch, 4, 4] -> [3]
            transform_matrix = ret[body_name].get_matrix()
            position = transform_matrix[0, :3, 3].cpu().numpy()  # 첫 번째 배치의 위치
            keypoints.append(position)
            cprint(f"  {i:2d}. {body_name:25} -> {hand_joint_name:20} [{position[0]:7.4f}, {position[1]:7.4f}, {position[2]:7.4f}]", "white")
        else:
            print('!!!!!!!!!!!!!!!!!!! body_name not in ret')
            # 해당 body가 없으면 원점 사용
            cprint(f"⚠️ Body '{body_name}' not found in kinematic chain", "yellow")
            position = np.array([0.0, 0.0, 0.0])
            keypoints.append(position)
    
    keypoints = np.array(keypoints)
    cprint(f"✅ Extracted {len(keypoints)} keypoints from zero pose", "green")
    
    # 3D 시각화
    fig = plt.figure(figsize=(20, 6))
    
    # 1. 키포인트만 표시
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title(f'{dexhand.name.title()} Hand - Zero Pose\nKeypoints Only', fontsize=12, fontweight='bold')
    
    for i, (keypoint, name) in enumerate(zip(keypoints, keypoint_names)):
        color = get_finger_color(name)
        ax1.scatter(keypoint[0], keypoint[1], keypoint[2], 
                   c=color, s=100, alpha=0.8, edgecolors='black', linewidth=1)
        ax1.text(keypoint[0], keypoint[1], keypoint[2], f'{i}', 
                fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.grid(True)
    
    # 2. 키포인트 + 연결선
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f'{dexhand.name.title()} Hand - Zero Pose\nWith Connections', fontsize=12, fontweight='bold')
    
    # 키포인트 표시
    for i, (keypoint, name) in enumerate(zip(keypoints, keypoint_names)):
        color = get_finger_color(name)
        ax2.scatter(keypoint[0], keypoint[1], keypoint[2], 
                   c=color, s=100, alpha=0.8, edgecolors='black', linewidth=1)
        ax2.text(keypoint[0], keypoint[1], keypoint[2], f'{i}', 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # 간단한 연결선 (손목에서 각 손가락 기저부로)
    if len(keypoints) > 0:
        wrist_pos = keypoints[0]  # 첫 번째가 보통 손목
        for i, (keypoint, name) in enumerate(zip(keypoints[1:], keypoint_names[1:]), 1):
            if "proximal" in name.lower() or "mcp" in name.lower():
                ax2.plot([wrist_pos[0], keypoint[0]], 
                        [wrist_pos[1], keypoint[1]], 
                        [wrist_pos[2], keypoint[2]], 
                        'gray', alpha=0.6, linewidth=3)
    
    # 각 손가락 내부 연결
    finger_groups = {}
    for i, name in enumerate(keypoint_names):
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            if finger in name.lower():
                if finger not in finger_groups:
                    finger_groups[finger] = []
                finger_groups[finger].append(i)
    
    for finger, indices in finger_groups.items():
        if len(indices) > 1:
            # 각 손가락 내에서 순서대로 연결
            sorted_indices = sorted(indices, key=lambda x: keypoint_names[x])
            for i in range(len(sorted_indices) - 1):
                start_idx = sorted_indices[i]
                end_idx = sorted_indices[i + 1]
                start_pos = keypoints[start_idx]
                end_pos = keypoints[end_idx]
                ax2.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_pos[2], end_pos[2]], 
                        get_finger_color(keypoint_names[start_idx]), alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.grid(True)
    
    # 3. 정보 표시
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    ax3.set_title(f'{dexhand.name.title()} Hand - Keypoint Details', fontsize=12, fontweight='bold')
    
    # 키포인트 정보 텍스트
    info_text = f"Hand Type: {dexhand.name.upper()}\n"
    info_text += f"DOFs: {dexhand.n_dofs}\n"
    info_text += f"Bodies: {len(dexhand.body_names)}\n"
    info_text += f"Keypoints: {len(keypoints)}\n"
    info_text += f"Device: {device}\n\n"
    info_text += "Keypoint List:\n"
    info_text += "Idx | Joint Name           | Position [X, Y, Z]\n"
    info_text += "-" * 55 + "\n"
    
    for i, (pos, name) in enumerate(zip(keypoints, keypoint_names)):
        finger = get_finger_color(name)
        info_text += f"{i:2d}. | {name:20} | [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] ({finger})\n"
    
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=8, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # 저장
    if save_path is None:
        save_path = f"/workspace/ManipTrans/{dexhand.name}_zero_pose_keypoints.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    cprint(f"✅ Zero pose keypoints visualization saved: {save_path}", "green")
    
    # 키포인트 데이터 저장 (JSON)
    keypoints_data = {
        "hand_type": dexhand.name,
        "side": dexhand.side,
        "dofs": dexhand.n_dofs,
        "pose": "zero",
        "device": device,
        "urdf_path": dexhand.urdf_path,
        "body_names": dexhand.body_names,
        "dof_names": dexhand.dof_names if hasattr(dexhand, 'dof_names') else [],
        "keypoints": {
            name: pos.tolist() for name, pos in zip(keypoint_names, keypoints)
        },
        "keypoint_summary": {
            "total_keypoints": len(keypoints),
            "finger_distribution": {
                finger: len([name for name in keypoint_names if finger in name.lower()])
                for finger in ["thumb", "index", "middle", "ring", "pinky", "wrist", "palm"]
            },
            "bounds": {
                "x_min": float(keypoints[:, 0].min()),
                "x_max": float(keypoints[:, 0].max()),
                "y_min": float(keypoints[:, 1].min()),
                "y_max": float(keypoints[:, 1].max()),
                "z_min": float(keypoints[:, 2].min()),
                "z_max": float(keypoints[:, 2].max()),
            }
        }
    }
    
    json_save_path = save_path.replace('.png', '_data.json')
    with open(json_save_path, 'w') as f:
        json.dump(keypoints_data, f, indent=2)
    cprint(f"✅ Keypoint data saved: {json_save_path}", "green")
    
    plt.close()
    
    return keypoints, keypoint_names


def compare_poses(dexhand, dof_poses_dict, save_path=None, device="cpu"):
    """
    여러 DOF 포즈를 비교하여 시각화
    
    Args:
        dexhand: DexHand 객체
        dof_poses_dict: {"pose_name": dof_tensor} 형태의 딕셔너리
        save_path: 저장할 파일 경로
        device: 계산에 사용할 device
    
    Returns:
        all_keypoints: dict of pose_name -> keypoints array
    """
    
    cprint(f"🔄 Comparing {len(dof_poses_dict)} different poses...", "cyan")
    
    # URDF에서 kinematic chain 생성
    asset_root = os.path.split(dexhand.urdf_path)[0]
    asset_file = os.path.split(dexhand.urdf_path)[1]
    urdf_path = os.path.join(asset_root, asset_file)
    
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    chain = pk.build_chain_from_urdf(urdf_content)
    chain = chain.to(dtype=torch.float32, device=device)
    
    # 각 포즈에 대해 키포인트 계산
    all_keypoints = {}
    
    for pose_name, dof_pos in dof_poses_dict.items():
        cprint(f"  🔢 Processing pose: {pose_name}", "blue")
        
        if not isinstance(dof_pos, torch.Tensor):
            dof_pos = torch.tensor(dof_pos, dtype=torch.float32, device=device)
        
        if dof_pos.dim() == 1:
            dof_pos = dof_pos.unsqueeze(0)  # batch dimension 추가
        
        with torch.no_grad():
            ret = chain.forward_kinematics(dof_pos)
        
        keypoints = []
        for body_name in dexhand.body_names:
            if body_name in ret:
                transform_matrix = ret[body_name].get_matrix()
                position = transform_matrix[0, :3, 3].cpu().numpy()
                keypoints.append(position)
            else:
                keypoints.append(np.array([0.0, 0.0, 0.0]))
        
        all_keypoints[pose_name] = np.array(keypoints)
        cprint(f"    ✅ {pose_name}: {len(keypoints)} keypoints", "green")
    
    # 시각화
    num_poses = len(dof_poses_dict)
    cols = min(3, num_poses)
    rows = (num_poses + cols - 1) // cols
    
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    
    for i, (pose_name, keypoints) in enumerate(all_keypoints.items()):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax.set_title(f'{dexhand.name.title()} Hand - {pose_name}', fontsize=12, fontweight='bold')
        
        # 키포인트 표시
        keypoint_names = [dexhand.to_hand(name)[0] for name in dexhand.body_names]
        for j, (keypoint, joint_name) in enumerate(zip(keypoints, keypoint_names)):
            color = get_finger_color(joint_name)
            ax.scatter(keypoint[0], keypoint[1], keypoint[2], 
                      c=color, s=80, alpha=0.8, edgecolors='black', linewidth=1)
            ax.text(keypoint[0], keypoint[1], keypoint[2], f'{j}', fontsize=8)
        
        # 연결선 추가
        if len(keypoints) > 0:
            # 손목에서 각 손가락으로
            wrist_pos = keypoints[0]
            for k, (keypoint, name) in enumerate(zip(keypoints[1:], keypoint_names[1:]), 1):
                if "proximal" in name.lower() or "mcp" in name.lower():
                    ax.plot([wrist_pos[0], keypoint[0]], 
                           [wrist_pos[1], keypoint[1]], 
                           [wrist_pos[2], keypoint[2]], 
                           'gray', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"/workspace/ManipTrans/{dexhand.name}_pose_comparison.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    cprint(f"✅ Pose comparison saved: {save_path}", "green")
    
    plt.close()
    
    return all_keypoints


def render_multi_viewpoint_keypoints(dexhand, save_path=None, device="cpu"):
    """
    다양한 시점(위, 아래, 왼쪽, 오른쪽, 앞, 뒤)에서 zero pose 키포인트 렌더링
    
    Args:
        dexhand: DexHand 객체
        save_path: 저장할 파일 경로 (None이면 기본 경로 사용)
        device: 계산에 사용할 device
    
    Returns:
        keypoints: numpy array of keypoint positions [N, 3]
        keypoint_names: list of keypoint names
    """
    
    cprint("🎥 Rendering multi-viewpoint keypoints...", "cyan")
    
    # 먼저 키포인트 계산
    keypoints, keypoint_names = compute_zero_pose_keypoints(dexhand, device)
    
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
        ax.set_title(f'{dexhand.name.title()} Hand - {view_name} View\n(elev={elev}°, azim={azim}°)', 
                    fontsize=12, fontweight='bold')
        
        # 키포인트 표시
        for j, (keypoint, name) in enumerate(zip(keypoints, keypoint_names)):
            color = get_finger_color(name)
            ax.scatter(keypoint[0], keypoint[1], keypoint[2], 
                      c=color, s=100, alpha=0.8, edgecolors='black', linewidth=1)
            
            # 텍스트 라벨 (너무 많으면 생략)
            if len(keypoints) <= 20:  # 20개 이하일 때만 번호 표시
                ax.text(keypoint[0], keypoint[1], keypoint[2], f'{j}', 
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.6))
        
        # 연결선 추가
        if len(keypoints) > 0:
            # 손목에서 각 손가락 기저부로
            wrist_pos = keypoints[0]
            for k, (keypoint, name) in enumerate(zip(keypoints[1:], keypoint_names[1:]), 1):
                if "proximal" in name.lower() or "mcp" in name.lower():
                    ax.plot([wrist_pos[0], keypoint[0]], 
                           [wrist_pos[1], keypoint[1]], 
                           [wrist_pos[2], keypoint[2]], 
                           'gray', alpha=0.6, linewidth=2)
        
        # 각 손가락 내부 연결
        finger_groups = {}
        for k, name in enumerate(keypoint_names):
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                if finger in name.lower():
                    if finger not in finger_groups:
                        finger_groups[finger] = []
                    finger_groups[finger].append(k)
        
        for finger, indices in finger_groups.items():
            if len(indices) > 1:
                sorted_indices = sorted(indices, key=lambda x: keypoint_names[x])
                for k in range(len(sorted_indices) - 1):
                    start_idx = sorted_indices[k]
                    end_idx = sorted_indices[k + 1]
                    start_pos = keypoints[start_idx]
                    end_pos = keypoints[end_idx]
                    ax.plot([start_pos[0], end_pos[0]], 
                           [start_pos[1], end_pos[1]], 
                           [start_pos[2], end_pos[2]], 
                           get_finger_color(keypoint_names[start_idx]), alpha=0.7, linewidth=1.5)
        
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
        save_path = f"/workspace/ManipTrans/{dexhand.name}_{dexhand.side}_multi_viewpoint.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    cprint(f"✅ Multi-viewpoint visualization saved: {save_path}", "green")
    
    plt.close()
    
    return keypoints, keypoint_names


def compute_zero_pose_keypoints(dexhand, device="cpu"):
    """
    Zero pose에서 키포인트만 계산 (시각화 제외)
    
    Args:
        dexhand: DexHand 객체
        device: 계산에 사용할 device
    
    Returns:
        keypoints: numpy array of keypoint positions [N, 3]
        keypoint_names: list of keypoint names
    """
    
    # URDF에서 kinematic chain 생성
    asset_root = os.path.split(dexhand.urdf_path)[0]
    asset_file = os.path.split(dexhand.urdf_path)[1]
    urdf_path = os.path.join(asset_root, asset_file)
    
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # PyTorch Kinematics chain 생성
    chain = pk.build_chain_from_urdf(urdf_content)
    chain = chain.to(dtype=torch.float32, device=device)
    
    # Zero pose (모든 DOF = 0)
    batch_size = 1
    zero_dof_pos = torch.zeros(batch_size, dexhand.n_dofs, dtype=torch.float32, device=device)
    
    # Forward kinematics 계산
    with torch.no_grad():
        ret = chain.forward_kinematics(zero_dof_pos)
    
    # 키포인트 추출
    keypoints = []
    keypoint_names = []
    
    for i, body_name in enumerate(dexhand.body_names):
        hand_joint_name = dexhand.to_hand(body_name)[0]
        keypoint_names.append(hand_joint_name)
        
        if body_name in ret:
            # Transform matrix에서 위치 추출 [batch, 4, 4] -> [3]
            transform_matrix = ret[body_name].get_matrix()
            position = transform_matrix[0, :3, 3].cpu().numpy()  # 첫 번째 배치의 위치
            keypoints.append(position)
        else:
            print('!!!!!!!!!!!!!!!!!!! body_name not in ret')
            # 해당 body가 없으면 원점 사용
            cprint(f"⚠️ Body '{body_name}' not found in kinematic chain", "yellow")
            position = np.array([0.0, 0.0, 0.0])
            keypoints.append(position)
    
    keypoints = np.array(keypoints)
    
    return keypoints, keypoint_names


def main():
    """메인 실행 함수"""
    
    # Command line arguments
    parser = gymutil.parse_arguments(
        description="Zero Pose Keypoint Renderer",
        headless=True,
        custom_parameters=[
            {
                "name": "--dexhand",
                "type": str,
                "default": "inspire",
                "help": "Hand type (inspire, allegro, etc.)"
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
                "help": "Hand side (left or right)"
            },
            {
                "name": "--render_zero_pose",
                "action": "store_true",
                "help": "Render zero pose keypoints"
            },
            {
                "name": "--render_pose_comparison",
                "action": "store_true", 
                "help": "Render comparison of different poses"
            },
            {
                "name": "--render_multi_viewpoint",
                "action": "store_true",
                "help": "Render multi-viewpoint keypoints"
            },
            {
                "name": "--save_dir",
                "type": str,
                "default": "/workspace/ManipTrans",
                "help": "Directory to save output files"
            },
            {
                "name": "--device",
                "type": str,
                "default": "cpu",
                "help": "Device for computation (cpu or cuda)"
            }
        ],
    )
    
    # DexHand 생성
    dexhand = DexHandFactory.create_hand(parser.dexhand, parser.side)
    cprint(f"🤖 Created {dexhand.name} {dexhand.side} hand", "cyan")
    
    if parser.render_zero_pose or (not parser.render_pose_comparison and not parser.render_multi_viewpoint):
        # Zero pose 렌더링 (기본 동작)
        cprint("🎯 Rendering zero pose keypoints...", "cyan", attrs=['bold'])
        save_path = os.path.join(parser.save_dir, f"{dexhand.name}_{dexhand.side}_zero_pose_keypoints.png")
        
        keypoints, keypoint_names = render_zero_pose_keypoints(
            dexhand, 
            save_path=save_path,
            device=parser.device
        )
        
        cprint(f"✅ Zero pose rendering completed for {dexhand.name} {dexhand.side} hand", "green", attrs=['bold'])
        cprint(f"📊 Generated {len(keypoints)} keypoints", "white")
        
    if parser.render_pose_comparison:
        # 다양한 포즈 비교 렌더링
        cprint("🎯 Rendering pose comparison...", "cyan", attrs=['bold'])
        
        # 다양한 포즈 정의
        poses = {
            "zero_pose": torch.zeros(dexhand.n_dofs),
            "small_pose": torch.tensor(np.array([np.pi / 50] * dexhand.n_dofs)),
            "medium_pose": torch.tensor(np.array([np.pi / 10] * dexhand.n_dofs)),
        }
        
        # Inspire hand 특별 포즈 추가
        if dexhand.name == "inspire":
            inspire_special = torch.zeros(dexhand.n_dofs)
            if dexhand.n_dofs > 9:
                inspire_special[8] = 0.8
                inspire_special[9] = 0.05
            poses["inspire_special"] = inspire_special
        
        save_path = os.path.join(parser.save_dir, f"{dexhand.name}_{dexhand.side}_pose_comparison.png")
        
        all_keypoints = compare_poses(
            dexhand,
            poses,
            save_path=save_path,
            device=parser.device
        )
        
        cprint(f"✅ Pose comparison completed for {dexhand.name} {dexhand.side} hand", "green", attrs=['bold'])
        cprint(f"📊 Compared {len(poses)} different poses", "white")

    if parser.render_multi_viewpoint:
        # 다양한 시점 렌더링
        cprint("🎯 Rendering multi-viewpoint keypoints...", "cyan", attrs=['bold'])
        
        save_path = os.path.join(parser.save_dir, f"{dexhand.name}_{dexhand.side}_multi_viewpoint.png")
        
        keypoints, keypoint_names = render_multi_viewpoint_keypoints(
            dexhand,
            save_path=save_path,
            device=parser.device
        )
        
        cprint(f"✅ Multi-viewpoint rendering completed for {dexhand.name} {dexhand.side} hand", "green", attrs=['bold'])
        cprint(f"📊 Generated {len(keypoints)} keypoints", "white")


if __name__ == "__main__":
    main() 