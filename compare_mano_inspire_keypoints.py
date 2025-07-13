#!/usr/bin/env python3
"""
MANO vs Inspire Hand Keypoints Comparison
MANO 기본 상태 (pose=0, shape=0)와 Inspire hand의 기본 상태 keypoint들을 비교 시각화
"""

# Import isaacgym FIRST to avoid import order issues
from isaacgym import gymapi, gymtorch, gymutil
import logging
logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from termcolor import cprint
import os

# Now safe to import torch and other modules
import torch
import pytorch_kinematics as pk

# Import other modules
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from manopth.manolayer import ManoLayer
from manopth import demo

cprint("✅ MANO and Inspire Hand modules imported successfully", "green")


def get_mano_keypoints(pose=None, shape=None, use_pca=True, ncomps=6):
    """MANO 모델에서 기본 상태 keypoint들과 메쉬 추출"""
    
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
        hand_verts, hand_joints = mano_layer(pose, shape)
    
    # NumPy로 변환
    vertices = hand_verts[0].numpy()  # [778, 3]
    keypoints = hand_joints[0].numpy()  # [21, 3]  
    
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


def get_inspire_hand_keypoints():
    """Inspire hand의 기본 상태 keypoint들 추출 (실제 Forward Kinematics 사용)"""
    
    cprint("🤖 Extracting Inspire Hand keypoints using FK...", "cyan")
    
    # DexHand Factory로 hand 정보 생성  
    dexhand = DexHandFactory.create_hand("inspire", "right")
    cprint(f"✅ Created {dexhand.name} hand with {dexhand.n_dofs} DOFs", "green")
    
    # URDF에서 kinematic chain 생성
    asset_root = os.path.split(dexhand.urdf_path)[0]
    asset_file = os.path.split(dexhand.urdf_path)[1]
    urdf_path = os.path.join(asset_root, asset_file)
    
    cprint(f"📄 Loading URDF: {urdf_path}", "cyan")
    
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # PyTorch Kinematics chain 생성
    chain = pk.build_chain_from_urdf(urdf_content)
    chain = chain.to(dtype=torch.float32, device='cpu')
    
    cprint(f"✅ Built kinematic chain with {len(chain.get_joint_parameter_names())} joints", "green")
    
    # 기본 관절 각도 설정 (모든 관절 0도)
    batch_size = 1
    dof_pos = torch.zeros(batch_size, dexhand.n_dofs)
    
    # Forward kinematics 계산
    with torch.no_grad():
        ret = chain.forward_kinematics(dof_pos)
    
    # Body names에 해당하는 키포인트들 추출
    keypoints = []
    keypoint_names = []
    
    for body_name in dexhand.body_names:
        hand_joint_name = dexhand.to_hand(body_name)[0]
        keypoint_names.append(hand_joint_name)
        
        if body_name in ret:
            print(body_name)
            # Transform matrix에서 위치 추출 [batch, 4, 4] -> [batch, 3]
            transform_matrix = ret[body_name].get_matrix()
            position = transform_matrix[0, :3, 3].numpy()  # 첫 번째 배치의 위치
            keypoints.append(position)
        else:
            # 해당 body가 없으면 원점 사용
            cprint(f"⚠️ Body '{body_name}' not found in kinematic chain", "yellow")
            keypoints.append(np.array([0.0, 0.0, 0.0]))
    
    keypoints = np.array(keypoints)
    
    cprint(f"✅ Extracted {len(keypoints)} keypoints from Inspire Hand FK", "green")
    
    return keypoints, keypoint_names


def get_finger_colors():
    """손가락별 색상 정의"""
    return {
        "wrist": "black",
        "thumb": "red",
        "index": "green", 
        "middle": "blue",
        "ring": "purple",
        "pinky": "orange"
    }


def classify_keypoint_by_finger(keypoint_name):
    """키포인트 이름으로부터 손가락 분류"""
    keypoint_name = keypoint_name.lower()
    if "wrist" in keypoint_name or "palm" in keypoint_name:
        return "wrist"
    elif "thumb" in keypoint_name:
        return "thumb"
    elif "index" in keypoint_name or "ff" in keypoint_name:
        return "index"
    elif "middle" in keypoint_name or "mf" in keypoint_name:
        return "middle"
    elif "ring" in keypoint_name or "rf" in keypoint_name:
        return "ring"
    elif "pinky" in keypoint_name or "lf" in keypoint_name:
        return "pinky"
    else:
        return "wrist"  # default


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


def get_inspire_skeleton_connections(inspire_names):
    """Inspire Hand 키포인트 연결 구조 정의"""
    connections = []
    
    # 손가락별로 키포인트들을 그룹화
    finger_joints = {"thumb": [], "index": [], "middle": [], "ring": [], "pinky": [], "wrist": []}
    
    for i, name in enumerate(inspire_names):
        finger = classify_keypoint_by_finger(name)
        finger_joints[finger].append((i, name))
    
    # 손목에서 각 손가락 기저부로 연결
    wrist_idx = 0  # 첫 번째가 보통 손목
    for finger, joints in finger_joints.items():
        if finger != "wrist" and joints:
            # 각 손가락의 첫 번째 관절(proximal)로 연결
            proximal_joints = [j for j in joints if "proximal" in j[1].lower()]
            if proximal_joints:
                connections.append((wrist_idx, proximal_joints[0][0]))
    
    # 각 손가락 내부 연결
    for finger, joints in finger_joints.items():
        if finger != "wrist" and len(joints) > 1:
            # 관절 순서: proximal -> intermediate -> distal -> tip
            joint_order = ["proximal", "intermediate", "distal", "tip"]
            ordered_joints = []
            
            for order in joint_order:
                for idx, joint_name in joints:
                    if order in joint_name.lower():
                        ordered_joints.append((idx, joint_name))
            
            # 순서대로 연결
            for i in range(len(ordered_joints) - 1):
                start_idx = ordered_joints[i][0]
                end_idx = ordered_joints[i + 1][0]
                connections.append((start_idx, end_idx))
    
    return connections


def visualize_keypoints_comparison(mano_vertices, mano_keypoints, mano_names, mano_layer, 
                                   inspire_keypoints, inspire_names, inspire_mesh_vertices=None, inspire_mesh_faces=None, 
                                   save_path=None):
    """MANO와 Inspire hand keypoint들을 비교 시각화 (메쉬 포함)"""
    
    fig = plt.figure(figsize=(20, 8))
    colors = get_finger_colors()
    
    # 1. MANO 키포인트 및 메쉬 시각화
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('MANO Keypoints & Mesh\n(Default Pose & Shape)', fontsize=14, fontweight='bold')
    
    if mano_keypoints is not None and mano_vertices is not None:
        # MANO 메쉬 렌더링
        hand_verts = torch.from_numpy(mano_vertices).unsqueeze(0)  # [1, 778, 3]
        hand_joints = torch.from_numpy(mano_keypoints).unsqueeze(0)  # [1, 21, 3]
        
        # demo.display_hand를 사용해서 메쉬 렌더링
        demo.display_hand({
            'verts': hand_verts,
            'joints': hand_joints
        }, mano_faces=mano_layer.th_faces, ax=ax1, show=False)
        
        # 키포인트 강조 표시
        for i, (keypoint, name) in enumerate(zip(mano_keypoints, mano_names)):
            finger = classify_keypoint_by_finger(name)
            color = colors.get(finger, 'gray')
            ax1.scatter(keypoint[0], keypoint[1], keypoint[2], 
                       c=color, s=80, alpha=1.0, edgecolors='black', linewidth=1)
            ax1.text(keypoint[0], keypoint[1], keypoint[2], f'{i}', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('Z')
    ax1.grid(True)
    
    # 2. Inspire Hand 키포인트 및 메쉬 시각화
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Inspire Hand Robot\n(IsaacGym Mesh + Keypoints)', fontsize=14, fontweight='bold')
    
    if inspire_keypoints is not None:
        # Inspire Hand 메쉬 시각화 (간단한 wireframe)
        if inspire_mesh_vertices is not None and len(inspire_mesh_vertices) > 1:
            # 키포인트들을 연결한 간단한 wireframe 표시
            connections = get_inspire_skeleton_connections(inspire_names)
            for start_idx, end_idx in connections:
                if start_idx < len(inspire_keypoints) and end_idx < len(inspire_keypoints):
                    start_pos = inspire_keypoints[start_idx]
                    end_pos = inspire_keypoints[end_idx]
                    ax2.plot([start_pos[0], end_pos[0]], 
                            [start_pos[1], end_pos[1]], 
                            [start_pos[2], end_pos[2]], 
                            'gray', alpha=0.7, linewidth=2)
            
            # 메쉬 body들을 구로 표시 (로봇 body 표현)
            for i, vertex in enumerate(inspire_mesh_vertices):
                finger = classify_keypoint_by_finger(inspire_names[i] if i < len(inspire_names) else "wrist")
                color = colors.get(finger, 'gray')
                ax2.scatter(vertex[0], vertex[1], vertex[2], 
                           c=color, s=100, alpha=0.8, marker='s', edgecolors='black', linewidth=1)
        
        # Inspire Hand 키포인트 강조
        for i, (keypoint, name) in enumerate(zip(inspire_keypoints, inspire_names)):
            finger = classify_keypoint_by_finger(name)
            color = colors.get(finger, 'gray')
            ax2.scatter(keypoint[0], keypoint[1], keypoint[2], 
                       c=color, s=60, alpha=1.0, marker='^', edgecolors='black', linewidth=1)
            ax2.text(keypoint[0], keypoint[1], keypoint[2], f'{i}', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(True)
    
    # 3. 오버레이 비교 (MANO 메쉬 + Inspire 로봇)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('MANO Mesh vs Inspire Robot\nComparison', fontsize=14, fontweight='bold')
    
    # MANO 메쉬 (반투명)
    if mano_keypoints is not None and mano_vertices is not None:
        hand_verts = torch.from_numpy(mano_vertices).unsqueeze(0)
        hand_joints = torch.from_numpy(mano_keypoints).unsqueeze(0)
        
        # 메쉬를 더 투명하게 렌더링
        demo.display_hand({
            'verts': hand_verts,
            'joints': hand_joints
        }, mano_faces=mano_layer.th_faces, ax=ax3, show=False, alpha=0.3)
        
        # MANO 키포인트 (원형)
        for i, (keypoint, name) in enumerate(zip(mano_keypoints, mano_names)):
            finger = classify_keypoint_by_finger(name)
            color = colors.get(finger, 'gray')
            ax3.scatter(keypoint[0], keypoint[1], keypoint[2], 
                       c=color, s=60, alpha=0.9, marker='o', edgecolors='black', linewidth=1,
                       label=f'MANO' if i == 0 else "")
    
    # Inspire Robot (사각형 - 로봇 body)
    if inspire_keypoints is not None and inspire_mesh_vertices is not None:
        # 로봇 body들
        for i, vertex in enumerate(inspire_mesh_vertices):
            if i < len(inspire_names):
                finger = classify_keypoint_by_finger(inspire_names[i])
                color = colors.get(finger, 'gray')
                ax3.scatter(vertex[0], vertex[1], vertex[2], 
                           c=color, s=80, alpha=0.7, marker='s', edgecolors='black', linewidth=1,
                           label=f'Inspire Robot' if i == 0 else "")
        
        # 로봇 연결선
        connections = get_inspire_skeleton_connections(inspire_names)
        for start_idx, end_idx in connections:
            if (start_idx < len(inspire_keypoints) and end_idx < len(inspire_keypoints) and
                start_idx < len(inspire_mesh_vertices) and end_idx < len(inspire_mesh_vertices)):
                start_pos = inspire_mesh_vertices[start_idx]
                end_pos = inspire_mesh_vertices[end_idx]
                ax3.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_pos[2], end_pos[2]], 
                        'gray', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.grid(True)
    
    # 범례 추가
    handles, labels = ax3.get_legend_handles_labels()
    if len(handles) >= 2:
        ax3.legend(handles[:2], ['MANO (○)', 'Inspire Robot (□)'], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        cprint(f"✅ Saved comparison plot: {save_path}", "green")
    
    return fig


def compute_keypoint_distances(mano_keypoints, inspire_keypoints, mano_names, inspire_names):
    """키포인트들 간의 거리 계산 및 분석"""
    
    cprint("📊 Computing keypoint distances...", "cyan")
    
    distances = []
    comparisons = []
    
    # 비슷한 이름의 키포인트들 매칭해서 비교
    for i, mano_name in enumerate(mano_names):
        mano_finger = classify_keypoint_by_finger(mano_name)
        mano_pos = mano_keypoints[i]
        
        best_match_dist = float('inf')
        best_match_name = ""
        best_match_pos = None
        
        for j, inspire_name in enumerate(inspire_names):
            inspire_finger = classify_keypoint_by_finger(inspire_name)
            inspire_pos = inspire_keypoints[j]
            
            # 같은 손가락에서 유사한 관절 찾기
            if mano_finger == inspire_finger:
                # 관절 타입 매칭 (mcp, pip, dip, tip)
                mano_joint_type = ""
                inspire_joint_type = ""
                
                for joint_type in ["mcp", "pip", "dip", "tip", "proximal", "intermediate", "distal"]:
                    if joint_type in mano_name.lower():
                        mano_joint_type = joint_type
                    if joint_type in inspire_name.lower():
                        inspire_joint_type = joint_type
                
                # 유사한 관절 타입인지 확인
                is_similar = False
                if mano_joint_type == inspire_joint_type:
                    is_similar = True
                elif (mano_joint_type == "mcp" and inspire_joint_type == "proximal") or \
                     (mano_joint_type == "proximal" and inspire_joint_type == "mcp"):
                    is_similar = True
                elif (mano_joint_type == "pip" and inspire_joint_type == "intermediate") or \
                     (mano_joint_type == "intermediate" and inspire_joint_type == "pip"):
                    is_similar = True
                elif (mano_joint_type == "dip" and inspire_joint_type == "distal") or \
                     (mano_joint_type == "distal" and inspire_joint_type == "dip"):
                    is_similar = True
                
                if is_similar:
                    dist = np.linalg.norm(mano_pos - inspire_pos)
                    if dist < best_match_dist:
                        best_match_dist = dist
                        best_match_name = inspire_name
                        best_match_pos = inspire_pos
        
        if best_match_pos is not None:
            distances.append(best_match_dist)
            comparisons.append({
                'mano_name': mano_name,
                'inspire_name': best_match_name,
                'mano_pos': mano_pos,
                'inspire_pos': best_match_pos,
                'distance': best_match_dist,
                'finger': mano_finger
            })
    
    return distances, comparisons


def print_distance_analysis(distances, comparisons):
    """거리 분석 결과 출력"""
    
    if not distances:
        cprint("❌ No comparable keypoints found", "red")
        return
    
    cprint("\n" + "="*60, "cyan")
    cprint("📊 MANO vs Inspire Hand Keypoint Distance Analysis", "cyan", attrs=['bold'])
    cprint("="*60, "cyan")
    
    # 전체 통계
    distances = np.array(distances)
    cprint(f"📏 Total comparable keypoints: {len(distances)}", "white")
    cprint(f"📏 Average distance: {distances.mean():.4f} m", "white")
    cprint(f"📏 Std deviation: {distances.std():.4f} m", "white")
    cprint(f"📏 Min distance: {distances.min():.4f} m", "white")
    cprint(f"📏 Max distance: {distances.max():.4f} m", "white")
    
    # 손가락별 통계
    finger_stats = {}
    for comp in comparisons:
        finger = comp['finger']
        dist = comp['distance']
        
        if finger not in finger_stats:
            finger_stats[finger] = []
        finger_stats[finger].append(dist)
    
    cprint(f"\n🖐️ Finger-wise Analysis:", "yellow", attrs=['bold'])
    for finger, dists in finger_stats.items():
        dists = np.array(dists)
        cprint(f"  {finger:>8}: avg={dists.mean():.4f}m, std={dists.std():.4f}m, count={len(dists)}", "white")
    
    # 상세 비교 (거리 순으로 정렬)
    cprint(f"\n🔍 Detailed Comparisons (sorted by distance):", "yellow", attrs=['bold'])
    comparisons_sorted = sorted(comparisons, key=lambda x: x['distance'])
    
    for i, comp in enumerate(comparisons_sorted[:15]):  # 상위 15개만 출력
        cprint(f"  {i+1:2d}. {comp['mano_name']:15} <-> {comp['inspire_name']:20} | "
               f"dist: {comp['distance']:.4f}m | finger: {comp['finger']}", "white")


def print_keypoint_details(mano_keypoints, mano_names, inspire_keypoints, inspire_names):
    """키포인트 상세 정보 출력"""
    
    cprint("\n" + "="*80, "cyan")
    cprint("📋 Keypoint Details", "cyan", attrs=['bold'])
    cprint("="*80, "cyan")
    
    cprint(f"\n🤚 MANO Keypoints ({len(mano_keypoints)}):", "yellow", attrs=['bold'])
    for i, (pos, name) in enumerate(zip(mano_keypoints, mano_names)):
        finger = classify_keypoint_by_finger(name)
        cprint(f"  {i:2d}. {name:15} | pos: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}] | finger: {finger}", "white")
    
    cprint(f"\n🤖 Inspire Hand Keypoints ({len(inspire_keypoints)}):", "yellow", attrs=['bold'])
    for i, (pos, name) in enumerate(zip(inspire_keypoints, inspire_names)):
        finger = classify_keypoint_by_finger(name)
        cprint(f"  {i:2d}. {name:20} | pos: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}] | finger: {finger}", "white")


def create_isaacgym_inspire_hand():
    """IsaacGym에서 실제 Inspire Hand 환경 생성"""
    
    cprint("🚀 Creating IsaacGym simulation for Inspire Hand...", "cyan")
    
    # Gym 초기화
    gym = gymapi.acquire_gym()
    
    # 시뮬레이션 파라미터 설정
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    
    # Physics engine 설정
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 0
    sim_params.physx.use_gpu = True
    
    # 시뮬레이션 생성
    compute_device_id = 0
    graphics_device_id = 0
    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    
    if sim is None:
        raise Exception("Failed to create sim")
    
    # 환경 생성
    env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
    env_upper = gymapi.Vec3(1.0, 1.0, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    return gym, sim, env


def load_inspire_hand_asset(gym, sim, dexhand):
    """Inspire Hand URDF 에셋 로드"""
    
    cprint(f"📄 Loading Inspire Hand URDF: {dexhand.urdf_path}", "cyan")
    
    # Asset 옵션 설정
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params = gymapi.VhacdParams()
    asset_options.vhacd_params.resolution = 300000
    
    # URDF 로드
    if not os.path.exists(dexhand.urdf_path):
        raise FileNotFoundError(f"URDF file not found: {dexhand.urdf_path}")
    
    asset = gym.load_asset(sim, os.path.dirname(dexhand.urdf_path), 
                          os.path.basename(dexhand.urdf_path), asset_options)
    
    if asset is None:
        raise Exception("Failed to load hand asset")
    
    return asset


def get_inspire_hand_mesh_data(pose="open"):
    """IsaacGym으로 Inspire Hand 메쉬 데이터와 키포인트 추출 (개선된 에러 처리)"""
    
    cprint("🤖 Extracting Inspire Hand mesh data using IsaacGym...", "cyan")
    
    try:
        # 1. DexHand Factory로 hand 정보 생성
        dexhand = DexHandFactory.create_hand("inspire", "right")
        cprint(f"✅ Created {dexhand.name} hand with {dexhand.n_dofs} DOFs", "green")
        
        # 2. IsaacGym 환경 생성
        gym, sim, env = create_isaacgym_inspire_hand()
        cprint("✅ Created IsaacGym simulation", "green")
        
        # 3. Hand asset 로드
        asset = load_inspire_hand_asset(gym, sim, dexhand)
        cprint("✅ Loaded hand asset", "green")
        
        # 4. Actor 생성
        pose_initial = gymapi.Transform()
        pose_initial.p = gymapi.Vec3(0.0, 0.0, 0.5)  # 0.5m 높이
        pose_initial.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        actor_handle = gym.create_actor(env, asset, pose_initial, "inspire_hand", 0, 1)
        
        if actor_handle is None:
            raise Exception("Failed to create hand actor")
        
        cprint("✅ Created hand actor", "green")
        
        # 5. DOF 속성 설정
        dof_props = gym.get_actor_dof_properties(env, actor_handle)
        
        # 모든 관절을 position control로 설정
        for i in range(len(dof_props)):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = 1000.0
            dof_props['damping'][i] = 100.0
            dof_props['effort'][i] = 100.0
        
        gym.set_actor_dof_properties(env, actor_handle, dof_props)
        
        # 6. 포즈 설정
        if pose == "open":
            joint_angles = torch.zeros(dof_props.shape[0])
        elif pose == "fist":
            joint_angles = torch.ones(dof_props.shape[0]) * 0.8
        else:  # default
            joint_angles = torch.zeros(dof_props.shape[0])
            if len(joint_angles) > 10:
                joint_angles[2:8] = 0.3  # 손가락들 약간 구부림
        
        # 7. 관절 각도 적용
        gym.set_actor_dof_position_targets(env, actor_handle, joint_angles.numpy())
        
        # 8. 시뮬레이션 단계 실행 (포즈 안정화)
        for _ in range(100):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        
        # 9. Body 정보 가져오기
        num_bodies = gym.get_asset_rigid_body_count(asset)
        body_names = []
        for i in range(num_bodies):
            body_name = gym.get_asset_rigid_body_name(asset, i)
            body_names.append(body_name)
        
        cprint(f"✅ Got {num_bodies} bodies", "green")
        
        # 10. Body states 가져오기 (개선된 처리)
        try:
            body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
            cprint(f"📊 Body states shape: {body_states.shape if hasattr(body_states, 'shape') else 'unknown'}", "cyan")
            cprint(f"📊 Body states dtype: {body_states.dtype if hasattr(body_states, 'dtype') else 'unknown'}", "cyan")
            
        except Exception as e:
            cprint(f"❌ Failed to get body states: {e}", "red")
            raise Exception("Cannot retrieve body states from IsaacGym")
        
        # 11. 키포인트 추출 (더 안전한 방법)
        keypoints = []
        keypoint_names = []
        
        for i in range(min(num_bodies, len(body_states) if body_states is not None else 0)):
            body_name = body_names[i]
            
            # IsaacGym body states는 structured numpy array
            if hasattr(body_states, 'dtype') and body_states.dtype.names:
                # Structured array인 경우
                state = body_states[i]
                
                # pose 필드 확인
                if 'pose' in state.dtype.names:
                    pose_data = state['pose']
                    
                    # pose 안에 p (position) 필드가 있는지 확인
                    if hasattr(pose_data, 'dtype') and 'p' in pose_data.dtype.names:
                        pos_data = pose_data['p']
                        
                        # position 데이터 추출
                        if hasattr(pos_data, 'dtype') and pos_data.dtype.names:

                            if all(field in pos_data.dtype.names for field in ['x', 'y', 'z']):
                                position = np.array([float(pos_data['x']), float(pos_data['y']), float(pos_data['z'])])
            
            keypoints.append(position)
            keypoint_names.append(body_name)
        
        # 12. 메쉬 데이터 생성 (키포인트 기반)
        mesh_vertices = np.array(keypoints) if keypoints else np.array([[0, 0, 0]])
        mesh_faces = []
        
        # 간단한 연결 구조 생성
        if len(keypoints) > 1:
            for i in range(1, min(len(keypoints), 4)):  # 첫 번째 몇 개만 연결
                mesh_faces.append([0, i, min(i+1, len(keypoints)-1)])
        
        # 13. 정리
        gym.destroy_sim(sim)
        
        keypoints = np.array(keypoints) if keypoints else np.array([[0, 0, 0]])
        cprint(f"✅ Extracted {len(keypoints)} keypoints from Inspire Hand IsaacGym", "green")
        
        return keypoints, keypoint_names, mesh_vertices, mesh_faces, dexhand
        
    except Exception as e:
        cprint(f"❌ Failed to load Inspire Hand with IsaacGym: {e}", "red")
        cprint("🔄 Falling back to Forward Kinematics method...", "yellow")
        
        # Fallback to FK method
        keypoints, keypoint_names = get_inspire_hand_keypoints()
        return keypoints, keypoint_names, np.array(keypoints), [], None


def simple_comparison():
    """간단한 MANO vs Inspire 비교 (IsaacGym 없이)"""
    
    cprint("🚀 Starting Simple MANO vs Inspire Hand Comparison (FK only)...", "cyan", attrs=['bold'])
    
    try:
        # 1. MANO 키포인트 추출
        mano_vertices, mano_keypoints, mano_layer = get_mano_keypoints()
        mano_names = get_mano_keypoint_names()
        cprint(f"✅ MANO: {len(mano_keypoints)} keypoints", "green")
        
        # 2. Inspire Hand 키포인트 추출 (FK만 사용)
        inspire_keypoints, inspire_names = get_inspire_hand_keypoints()
        cprint(f"✅ Inspire Hand: {len(inspire_keypoints)} keypoints", "green")
        
        # 3. 간단한 시각화
        fig = plt.figure(figsize=(15, 5))
        colors = get_finger_colors()
        
        # MANO 시각화
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_title('MANO Hand\n(Mesh + Keypoints)', fontsize=12, fontweight='bold')
        
        # MANO 메쉬 렌더링
        hand_verts = torch.from_numpy(mano_vertices).unsqueeze(0)
        hand_joints = torch.from_numpy(mano_keypoints).unsqueeze(0)
        
        demo.display_hand({
            'verts': hand_verts,
            'joints': hand_joints
        }, mano_faces=mano_layer.th_faces, ax=ax1, show=False, alpha=0.7)
        
        # MANO 키포인트
        for i, (keypoint, name) in enumerate(zip(mano_keypoints, mano_names)):
            finger = classify_keypoint_by_finger(name)
            color = colors.get(finger, 'gray')
            ax1.scatter(keypoint[0], keypoint[1], keypoint[2], c=color, s=60, alpha=1.0)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Inspire Hand 시각화
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_title('Inspire Hand\n(FK Keypoints)', fontsize=12, fontweight='bold')
        
        # Inspire 키포인트
        for i, (keypoint, name) in enumerate(zip(inspire_keypoints, inspire_names)):
            finger = classify_keypoint_by_finger(name)
            color = colors.get(finger, 'gray')
            ax2.scatter(keypoint[0], keypoint[1], keypoint[2], c=color, s=60, alpha=1.0, marker='^')
            ax2.text(keypoint[0], keypoint[1], keypoint[2], f'{i}', fontsize=8)
        
        # Inspire 골격 연결
        connections = get_inspire_skeleton_connections(inspire_names)
        for start_idx, end_idx in connections:
            if start_idx < len(inspire_keypoints) and end_idx < len(inspire_keypoints):
                start_pos = inspire_keypoints[start_idx]
                end_pos = inspire_keypoints[end_idx]
                ax2.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_pos[2], end_pos[2]], 
                        'gray', alpha=0.6, linewidth=1)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 비교 오버레이
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_title('Comparison\n(MANO vs Inspire)', fontsize=12, fontweight='bold')
        
        # MANO 키포인트 (원형, 파란색)
        for keypoint in mano_keypoints:
            ax3.scatter(keypoint[0], keypoint[1], keypoint[2], c='blue', s=50, alpha=0.7, marker='o')
        
        # Inspire 키포인트 (삼각형, 빨간색)
        for keypoint in inspire_keypoints:
            ax3.scatter(keypoint[0], keypoint[1], keypoint[2], c='red', s=50, alpha=0.7, marker='^')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend(['MANO', 'Inspire'], loc='upper right')
        
        plt.tight_layout()
        
        # 저장
        save_path = "/workspace/ManipTrans/mano_inspire_simple_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        cprint(f"✅ Simple comparison saved: {save_path}", "green")
        
        # 4. 간단한 통계
        distances, comparisons = compute_keypoint_distances(mano_keypoints, inspire_keypoints, 
                                                           mano_names, inspire_names)
        if distances:
            cprint(f"\n📊 Quick Statistics:", "cyan", attrs=['bold'])
            cprint(f"  🤚 MANO keypoints: {len(mano_keypoints)}", "white")
            cprint(f"  🤖 Inspire keypoints: {len(inspire_keypoints)}", "white")
            cprint(f"  📏 Average distance: {np.mean(distances):.4f}m", "white")
            cprint(f"  📊 Comparable pairs: {len(distances)}", "white")
        
        cprint(f"✅ Simple comparison completed successfully!", "green", attrs=['bold'])
        return fig
        
    except Exception as e:
        cprint(f"❌ Simple comparison failed: {e}", "red")
        import traceback
        traceback.print_exc()


def main():
    """메인 실행 함수 (개선된 에러 처리)"""
    
    cprint("🔄 Starting MANO vs Inspire Hand Robot Comparison...", "cyan", attrs=['bold'])
    
    # 1. MANO 키포인트 추출
    try:
        mano_vertices, mano_keypoints, mano_layer = get_mano_keypoints()
        mano_names = get_mano_keypoint_names()
        cprint(f"✅ MANO data loaded successfully: {len(mano_keypoints)} keypoints", "green")
    except Exception as e:
        cprint(f"❌ Failed to load MANO data: {e}", "red")
        return
    
    # 2. Inspire Hand 로봇 메쉬 및 키포인트 추출
    inspire_keypoints = None
    inspire_names = None
    inspire_mesh_vertices = None
    inspire_mesh_faces = None
    inspire_dexhand = None
    
    # 첫 번째 시도: IsaacGym 사용
    cprint("🤖 Attempting IsaacGym method...", "cyan")
    try:
        inspire_keypoints, inspire_names, inspire_mesh_vertices, inspire_mesh_faces, inspire_dexhand = get_inspire_hand_mesh_data("open")
        
        if inspire_dexhand is not None:
            cprint(f"✅ Successfully loaded Inspire Hand robot with IsaacGym", "green")
            cprint(f"📊 Robot info: {inspire_dexhand.name}, {inspire_dexhand.n_dofs} DOFs", "cyan")
        else:
            cprint("⚠️ IsaacGym method returned None, trying fallback...", "yellow")
            raise Exception("IsaacGym method failed")
    
    except Exception as e:
        cprint(f"❌ IsaacGym method failed: {e}", "red")
        
        # 두 번째 시도: Forward Kinematics 방법
        cprint("🔄 Using fallback Forward Kinematics method...", "yellow")
        try:
            inspire_keypoints, inspire_names = get_inspire_hand_keypoints()
            inspire_mesh_vertices = inspire_keypoints  # 키포인트를 메쉬 버텍스로 사용
            inspire_mesh_faces = []
            inspire_dexhand = None
            cprint(f"✅ Forward Kinematics method successful: {len(inspire_keypoints)} keypoints", "green")
            
        except Exception as e2:
            cprint(f"❌ Both methods failed. Forward Kinematics error: {e2}", "red")
            return
    
    # 3. 키포인트 상세 정보 출력
    try:
        print_keypoint_details(mano_keypoints, mano_names, inspire_keypoints, inspire_names)
    except Exception as e:
        cprint(f"⚠️ Error printing keypoint details: {e}", "yellow")
    
    # 4. 시각화 (메쉬 포함)
    save_path = "/workspace/ManipTrans/mano_inspire_robot_comparison.png"
    try:
        fig = visualize_keypoints_comparison(mano_vertices, mano_keypoints, mano_names, mano_layer, 
                                           inspire_keypoints, inspire_names, 
                                           inspire_mesh_vertices, inspire_mesh_faces, 
                                           save_path)
        cprint(f"✅ Visualization saved to: {save_path}", "green")
    except Exception as e:
        cprint(f"❌ Failed to create visualization: {e}", "red")
        return
    
    # 5. 거리 분석
    try:
        distances, comparisons = compute_keypoint_distances(mano_keypoints, inspire_keypoints, 
                                                           mano_names, inspire_names)
        print_distance_analysis(distances, comparisons)
    except Exception as e:
        cprint(f"⚠️ Error in distance analysis: {e}", "yellow")
    
    # 6. 추가 정보 출력
    if inspire_dexhand is not None:
        try:
            cprint(f"\n🤖 Inspire Hand Robot Details:", "cyan", attrs=['bold'])
            cprint(f"  📄 URDF Path: {inspire_dexhand.urdf_path}", "white")
            cprint(f"  🔧 DOFs: {inspire_dexhand.n_dofs}", "white")
            cprint(f"  🏷️ Body Names: {len(inspire_dexhand.body_names)} bodies", "white")
            if hasattr(inspire_dexhand, 'body_names') and len(inspire_dexhand.body_names) > 0:
                cprint(f"  📋 Bodies: {inspire_dexhand.body_names[:5]}{'...' if len(inspire_dexhand.body_names) > 5 else ''}", "white")
        except Exception as e:
            cprint(f"⚠️ Error displaying robot details: {e}", "yellow")
    
    if inspire_mesh_vertices is not None:
        try:
            cprint(f"\n📊 Inspire Hand Mesh Data:", "cyan", attrs=['bold'])
            cprint(f"  🔺 Mesh vertices: {len(inspire_mesh_vertices)}", "white")
            cprint(f"  🔗 Mesh faces: {len(inspire_mesh_faces) if inspire_mesh_faces else 0}", "white")
        except Exception as e:
            cprint(f"⚠️ Error displaying mesh info: {e}", "yellow")
    
    # 7. 최종 요약
    cprint(f"\n✅ Robot comparison completed!", "green", attrs=['bold'])
    cprint(f"🎯 Method used: {'IsaacGym' if inspire_dexhand else 'Forward Kinematics'}", "yellow", attrs=['bold'])
    cprint(f"📊 Results: {len(mano_keypoints)} MANO vs {len(inspire_keypoints)} Inspire keypoints", "white")
    
    if inspire_dexhand:
        cprint(f"🤖 Robot simulation successful with full mesh data", "white")
    else:
        cprint(f"⚡ Used lightweight kinematics-only method", "white")
    
    cprint(f"💾 Visualization saved to: {save_path}", "white")


if __name__ == "__main__":
    import sys
    
    # 사용자가 simple 옵션을 원하는 경우
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        simple_comparison()
    else:
        main() 