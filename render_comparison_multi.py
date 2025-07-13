from isaacgym import gymapi, gymtorch, gymutil
from main.dataset.mano2dexhand_gigahands import load_gigahands_sequence, pack_gigahands_data
from main.dataset.transform import rot6d_to_rotmat, aa_to_rotmat, rotmat_to_aa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch
import os
import numpy as np
import pickle
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import pytorch_kinematics as pk
from termcolor import cprint


def load_original_mano_keypoints(data_dir, seq_id, side="right", frame_idx=0):
    """GigaHands 원본 MANO 키포인트 로드 및 좌표계 변환"""
    
    # 원본 시퀀스 데이터 로드
    motion_data, scene_name, sequence_name = load_gigahands_sequence(data_dir, seq_id, side)
    
    # 특정 프레임 선택
    if frame_idx >= motion_data.shape[0]:
        frame_idx = 0
        cprint(f"Frame index {frame_idx} out of range, using frame 0", "yellow")
    
    frame_data = motion_data[frame_idx:frame_idx+1]  # [1, 42, 3]
    
    # dexhand 생성 (좌표계 변환용)
    dexhand = DexHandFactory.create_hand("inspire", side)
    
    # pack_gigahands_data와 동일한 변환 적용
    demo_data = pack_gigahands_data(frame_data, dexhand, side)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    demo_data["wrist_pos"] = demo_data["wrist_pos"].to(device)
    demo_data["wrist_rot"] = demo_data["wrist_rot"].to(device)
    for joint_name in demo_data["mano_joints"]:
        demo_data["mano_joints"][joint_name] = demo_data["mano_joints"][joint_name].to(device)
    # print(demo_data['mano_joints']['wrist'].shape)
    # 1/0
    
    # 좌표계 변환 (mano2dexhand_gigahands.py와 동일)
    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
        np.array([np.pi / 2, 0, 0])
    )
    mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.5])
    mujoco2gym_transf = torch.tensor(mujoco2gym_transf, dtype=torch.float32, device=device)
    
    # 손목 변환
    wrist_pos = demo_data["wrist_pos"][0]  # [3]
    wrist_rot = demo_data["wrist_rot"][0]  # [3] axis-angle
    wrist_rot_matrix = aa_to_rotmat(wrist_rot.unsqueeze(0))[0]  # [3, 3]
    
    # 좌표계 변환 적용
    wrist_pos_transformed = (mujoco2gym_transf[:3, :3] @ wrist_pos.T).T + mujoco2gym_transf[:3, 3]
    wrist_rot_transformed = mujoco2gym_transf[:3, :3] @ wrist_rot_matrix
    
    # MANO 조인트들을 하나의 텐서로 결합
    # mano_joints_list = []
    # for j_name in dexhand.body_names:
    #     hand_joint_name = dexhand.to_hand(j_name)[0]
    #     if hand_joint_name != "wrist":
    #         mano_joints_list.append(demo_data["mano_joints"][hand_joint_name][0])
    mano_joints = torch.cat([
        demo_data["mano_joints"][dexhand.to_hand(j_name)[0]]
        for j_name in dexhand.body_names
        if dexhand.to_hand(j_name)[0] != "wrist"
    ], dim=-1).view(1, -1, 3)
    
    
    # mano_joints = torch.stack(mano_joints_list)  # [n_joints, 3]
    
    # MANO 조인트들도 좌표계 변환
    mano_joints = mano_joints.view(-1, 3)
    print((wrist_rot_transformed @ mano_joints.T).T.shape)
    # mano_joints_world = (mano_joints @ wrist_rot_transformed.T) + wrist_pos_transformed[:,None]
    # mano_joints_world = (wrist_rot_transformed @ mano_joints.T).T + wrist_pos_transformed[None,:]
    mano_joints_world = mano_joints.view(-1, 3)
    mano_joints_transformed = (mujoco2gym_transf[:3, :3] @ mano_joints_world.T).T + mujoco2gym_transf[:3, 3]
    
    # 손목 포함한 전체 조인트
    all_joints = torch.cat([wrist_pos_transformed.unsqueeze(0), mano_joints_transformed], dim=0).cpu()
    
    return all_joints.numpy(), dexhand.body_names


def compute_joint_errors(original_joints, retargeted_joints, joint_names):
    """원본과 retargeted 조인트 간의 오차 계산"""
    
    if original_joints.shape != retargeted_joints.shape:
        cprint(f"Shape mismatch: original {original_joints.shape} vs retargeted {retargeted_joints.shape}", "red")
        return None
    
    # 각 조인트별 L2 distance 계산
    joint_errors = np.linalg.norm(original_joints - retargeted_joints, axis=-1)
    
    # 통계 계산
    mean_error = np.mean(joint_errors)
    max_error = np.max(joint_errors)
    min_error = np.min(joint_errors)
    
    error_stats = {
        'joint_errors': joint_errors,
        'mean_error': mean_error,
        'max_error': max_error,
        'min_error': min_error,
        'joint_names': joint_names
    }
    
    return error_stats


def render_comparison_multiview(original_joints, retargeted_joints, joint_names, save_dir="comparison_renders", prefix="comparison"):
    """
    원본과 retargeted 키포인트를 같은 3D 공간에서 다중 뷰로 비교 시각화
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 에러 계산
    error_stats = compute_joint_errors(original_joints, retargeted_joints, joint_names)
    
    if error_stats is not None:
        cprint(f"Joint Error Statistics:", "cyan")
        cprint(f"  Mean error: {error_stats['mean_error']:.4f}", "white")
        cprint(f"  Max error: {error_stats['max_error']:.4f}", "white") 
        cprint(f"  Min error: {error_stats['min_error']:.4f}", "white")
    
    # 손가락별 연결 정의
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
        if 'thumb' in name:
            finger_connections['thumb'].append(i)
        elif 'index' in name:
            finger_connections['index'].append(i)
        elif 'middle' in name:
            finger_connections['middle'].append(i)
        elif 'ring' in name:
            finger_connections['ring'].append(i)
        elif 'pinky' in name:
            finger_connections['pinky'].append(i)
        else:
            finger_connections['wrist'].append(i)
    
    # 손가락별 색상 정의
    finger_colors = {
        'thumb': 'red',
        'index': 'blue', 
        'middle': 'green',
        'ring': 'orange',
        'pinky': 'purple',
        'wrist': 'black'
    }

    # 다양한 시점 설정: (elev, azim)
    views = [
        (30, 45),
        (30, -45),
        (0, 90),
        (90, 0),
        (45, 180),
        (20, 270)
    ]

    for i, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 손가락별로 원본과 retargeted 모두 표시
        for finger, indices in finger_connections.items():
            if len(indices) > 0:
                color = finger_colors[finger]
                
                # === 원본 MANO 키포인트 ===
                orig_points = original_joints[indices]
                ax.scatter(orig_points[:, 0], orig_points[:, 1], orig_points[:, 2], 
                          c=color, s=80, marker='o', alpha=0.7, label=f'{finger} (original)', 
                          edgecolors='black', linewidth=1)
                
                # === Retargeted 키포인트 ===
                ret_points = retargeted_joints[indices]
                ax.scatter(ret_points[:, 0], ret_points[:, 1], ret_points[:, 2], 
                          c=color, s=60, marker='^', alpha=0.9, label=f'{finger} (retargeted)',
                          edgecolors='white', linewidth=1)
                
                # 손가락 관절들을 순서대로 연결 (원본)
                if finger != 'wrist' and len(indices) > 1:
                    sorted_indices = []
                    for joint_type in ['proximal', 'intermediate', 'tip']:
                        for idx in indices:
                            if joint_type in joint_names[idx]:
                                sorted_indices.append(idx)
                    
                    if len(sorted_indices) > 1:
                        # 원본 연결 (실선)
                        sorted_orig_points = original_joints[sorted_indices]
                        ax.plot(sorted_orig_points[:, 0], sorted_orig_points[:, 1], sorted_orig_points[:, 2], 
                               c=color, linewidth=2, alpha=0.5, linestyle='-')
                        
                        # Retargeted 연결 (점선)
                        sorted_ret_points = retargeted_joints[sorted_indices]
                        ax.plot(sorted_ret_points[:, 0], sorted_ret_points[:, 1], sorted_ret_points[:, 2], 
                               c=color, linewidth=2, alpha=0.8, linestyle='--')
                
                # 손목에서 각 손가락으로 연결
                if finger != 'wrist' and len(finger_connections['wrist']) > 0:
                    wrist_idx = finger_connections['wrist'][0]
                    if len(indices) > 0:
                        first_joint_idx = indices[0]
                        for idx in indices:
                            if 'proximal' in joint_names[idx]:
                                first_joint_idx = idx
                                break
                        
                        # 원본 손목 연결
                        orig_wrist_pos = original_joints[wrist_idx]
                        orig_first_joint_pos = original_joints[first_joint_idx]
                        ax.plot([orig_wrist_pos[0], orig_first_joint_pos[0]], 
                               [orig_wrist_pos[1], orig_first_joint_pos[1]], 
                               [orig_wrist_pos[2], orig_first_joint_pos[2]], 
                               c=color, linewidth=1, alpha=0.3, linestyle='-')
                        
                        # Retargeted 손목 연결
                        ret_wrist_pos = retargeted_joints[wrist_idx]
                        ret_first_joint_pos = retargeted_joints[first_joint_idx]
                        ax.plot([ret_wrist_pos[0], ret_first_joint_pos[0]], 
                               [ret_wrist_pos[1], ret_first_joint_pos[1]], 
                               [ret_wrist_pos[2], ret_first_joint_pos[2]], 
                               c=color, linewidth=1, alpha=0.6, linestyle='--')
        
        # 대응 조인트 간 연결선 (에러 시각화)
        if error_stats is not None:
            for j in range(len(joint_names)):
                error_val = error_stats['joint_errors'][j]
                # 에러가 큰 조인트만 연결선 표시
                if error_val > error_stats['mean_error']:
                    orig_pos = original_joints[j]
                    ret_pos = retargeted_joints[j]
                    ax.plot([orig_pos[0], ret_pos[0]], 
                           [orig_pos[1], ret_pos[1]], 
                           [orig_pos[2], ret_pos[2]], 
                           c='red', linewidth=1, alpha=0.4, linestyle=':')
        
        # 범례 설정 (중복 제거)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Original vs Retargeted Comparison\nView {i+1} (elev={elev}, azim={azim})")
        ax.view_init(elev=elev, azim=azim)

        # 렌더링 범위 자동 설정 (두 데이터 모두 고려)
        all_points = np.concatenate([original_joints, retargeted_joints], axis=0)
        max_range = np.ptp(all_points, axis=0).max() * 0.6
        center = all_points.mean(axis=0)
        for axis, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            axis([c - max_range, c + max_range])

        # 에러 통계를 텍스트로 표시
        if error_stats is not None:
            ax.text2D(0.02, 0.98, f"Mean Error: {error_stats['mean_error']:.4f}\nMax Error: {error_stats['max_error']:.4f}", 
                     transform=ax.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        save_path = os.path.join(save_dir, f"{prefix}_view{i+1}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved comparison view {i+1} to {save_path}")

    # 에러 분석 결과 출력
    if error_stats is not None:
        print(f"\n=== Joint Error Analysis ===")
        for i, (joint_name, error) in enumerate(zip(joint_names, error_stats['joint_errors'])):
            print(f"  {joint_name}: {error:.4f}")


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="Compare Original vs Retargeted Hand Keypoints",
        headless=True,
        custom_parameters=[
            {
                "name": "--original_path",
                "type": str,
                "default": "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands",
                "help": "Path to GigaHands dataset directory"
            },
            {
                "name": "--seq_id",
                "type": str,
                "default": "p001-folder/000",
                "help": "Sequence ID (e.g., 'p001-folder/000')"
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
                "help": "Hand side (right/left)"
            },
            {
                "name": "--frame_idx",
                "type": int,
                "default": 0,
                "help": "Frame index to visualize"
            },
            {
                "name": "--retargeted_path",
                "type": str,
                "default": "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/retargeted/mano2inspire_rh/p001-folder/keypoints_3d_mano/000_retargeted.pkl",
                "help": "Path to retargeted data pickle file"
            },
            {
                "name": "--save_dir",
                "type": str,
                "default": "comparison_renders",
                "help": "Directory to save comparison renders"
            }
        ],
    )

    # 원본 MANO 키포인트 로드
    cprint(f"Loading original MANO keypoints for sequence: {_parser.original_path}", "blue")
    original_joints, joint_names = load_original_mano_keypoints(
        _parser.original_path, _parser.seq_id, _parser.side, _parser.frame_idx
    )
    cprint(f"Original joints shape: {original_joints.shape}", "green")

    # Retargeted 결과 로드
    cprint(f"Loading retargeted data from: {_parser.retargeted_path}", "blue")
    to_dump = pickle.load(open(_parser.retargeted_path, "rb"))
    
    # Retargeted 키포인트 복원 (Forward Kinematics 사용)
    dexhand = DexHandFactory.create_hand("inspire", _parser.side)
    
    # URDF에서 chain 생성
    asset_root = os.path.split(dexhand.urdf_path)[0]
    asset_file = os.path.split(dexhand.urdf_path)[1]
    chain = pk.build_chain_from_urdf(open(os.path.join(asset_root, asset_file)).read())
    chain = chain.to(dtype=torch.float32, device='cpu')
    
    # Retargeted 데이터에서 키포인트 복원
    opt_wrist_pos = torch.tensor(to_dump["opt_wrist_pos"][_parser.frame_idx])
    opt_wrist_rot = torch.tensor(to_dump["opt_wrist_rot"][_parser.frame_idx])  # axis-angle
    opt_dof_pos = torch.tensor(to_dump["opt_dof_pos"][_parser.frame_idx])
    
    # Isaac Gym DOF 순서를 pytorch_kinematics 순서로 변환
    isaac2chain_order = []
    for j in chain.get_joint_parameter_names():
        # 임시로 순서 매핑 (실제로는 mano2dexhand_gigahands.py에서 가져와야 함)
        isaac2chain_order.append(0)  # 임시값
    
    # Forward Kinematics 계산
    with torch.no_grad():
        # 손목 rotation matrix 변환
        wrist_rotmat = aa_to_rotmat(opt_wrist_rot.unsqueeze(0))[0]
        
        # FK 계산
        fk_result = chain.forward_kinematics(opt_dof_pos.unsqueeze(0))
        joint_positions_local = torch.stack(
            [fk_result[k].get_matrix()[0, :3, 3] for k in joint_names], dim=0
        )
        
        # 손목 좌표계로 변환
        joint_positions_world = (wrist_rotmat @ joint_positions_local.T).T + opt_wrist_pos
        retargeted_joints = joint_positions_world.numpy()

    cprint(f"Retargeted joints shape: {retargeted_joints.shape}", "green")

    # 비교 시각화
    render_comparison_multiview(
        original_joints=original_joints,
        retargeted_joints=retargeted_joints,
        joint_names=joint_names,
        save_dir=_parser.save_dir,
        prefix=f"{_parser.seq_id.replace('/', '_')}_frame{_parser.frame_idx}"
    ) 