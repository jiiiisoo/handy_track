# CRITICAL: Import isaacgym modules FIRST before any other imports
# This prevents the "PyTorch was imported before isaacgym modules" error
from isaacgym import gymapi, gymtorch, gymutil

import math
import os
import pickle
import torch
import logging

logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import numpy as np
import pytorch_kinematics as pk
from termcolor import cprint
import json

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rotmat_to_rot6d,
)
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from manopth.manolayer import ManoLayer


def pack_gigahands_data(data_item, dexhand, side="right"):
    """GigaHands 데이터를 ManipTrans 형식으로 변환 (원본 mano2dexhand.py 스타일)"""
    
    # GigaHands 키포인트 구조: [T, 42, 3] where 0-20=right, 21-41=left
    motion_data = data_item
    T = motion_data.shape[0]
    
    # 텐서가 torch.Tensor가 아닌 경우 변환
    if not isinstance(motion_data, torch.Tensor):
        motion_data = torch.tensor(motion_data, dtype=torch.float32)
    
    # 손 구분에 따른 키포인트 추출
    if side == "left":
        # 오른손: 0-20번 키포인트 (21개)
        hand_keypoints = motion_data[:, :21, :]  # [T, 21, 3]
        cprint(f"Using RIGHT hand keypoints (indices 0-20)", "yellow")
    elif side == "right":
        # 왼손: 21-41번 키포인트 (21개)  
        hand_keypoints = motion_data[:, 21:42, :]  # [T, 21, 3]
        cprint(f"Using LEFT hand keypoints (indices 21-41)", "yellow")
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'right' or 'left'")
    
    # 손목 위치 (첫 번째 키포인트가 손목)
    wrist_pos = hand_keypoints[:, 0, :]  # [T, 3]
    
        # 손목 회전 추정 (index finger 방향 기반)
    # if hand_keypoints.shape[1] >= 9:  # index_tip이 있다면
    #     # 손목에서 index tip으로의 벡터로 회전 추정
    #     wrist_to_index = hand_keypoints[:, 8, :] - hand_keypoints[:, 0, :]  # index_tip - wrist
    #     wrist_to_index = wrist_to_index / (torch.norm(wrist_to_index, dim=-1, keepdim=True) + 1e-8)
        
    #     # 기본 forward 방향 [1, 0, 0]에서 wrist_to_index로의 회전 계산
    #     forward = torch.tensor([1.0, 0.0, 0.0], dtype=wrist_to_index.dtype, device=wrist_to_index.device).expand_as(wrist_to_index)
        
    #     # Cross product로 회전축 계산
    #     rotation_axis = torch.cross(forward, wrist_to_index, dim=-1)
    #     rotation_axis = rotation_axis / (torch.norm(rotation_axis, dim=-1, keepdim=True) + 1e-8)
        
    #     # Dot product로 회전각 계산
    #     cos_angle = torch.sum(forward * wrist_to_index, dim=-1, keepdim=True)
    #     cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    #     angle = torch.acos(cos_angle)
        
    #     # Axis-angle 표현
    #     wrist_rot = rotation_axis * angle  # [T, 3]
    
    # GigaHands MANO 키포인트 순서 정의 (21개)
    
    with open("/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose/p001-folder/params/000.json", "r") as f:
        params = json.load(f)
    pose = torch.tensor(params["right"]["poses"], dtype=torch.float32)
    T = pose.shape[0]
    shape = torch.tensor(params["right"]["shapes"], dtype=torch.float32).repeat(T, 1)
    mano_layer = ManoLayer(
        mano_root='/workspace/manopth/mano/models', 
        use_pca=True, 
        ncomps=6, 
        flat_hand_mean=True
    )
    with torch.no_grad():
        hand_verts, _, transform_abs = mano_layer(pose, shape)
    # print(hand_verts.shape, hand_keypoints.shape, transform_abs.shape)
    # print(hand_keypoints.shape)

    # with torch.no_grad():
    #     hand_verts, hand_joints, transform_abs = mano_layer(hand_keypoints, shape)

    # print(hand_verts.shape, hand_joints.shape, transform_abs.shape)
    # 1/0

    # transform_abs = torch.load("/workspace/ManipTrans/transform_abs.pt")
    keypoints = hand_keypoints.numpy()  # [21, 3]  
    wrist_pos = keypoints[:,0,:]
    middle_pos = keypoints[:,9,:]
    wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # ? hack for wrist position
    dexhand = DexHandFactory.create_hand(dexhand_type="inspire", side="right")
    wrist_pos += np.array(dexhand.relative_translation)
    mano_rot_offset = dexhand.relative_rotation
    # wrist_rot = transform_abs[:, 0, :3, :3].detach() @ np.repeat(mano_rot_offset[None], transform_abs.shape[0], axis=0)
    # print(wrist_pos.shape, wrist_rot.shape)
    wrist_rot = transform_abs[:, 0, :3, :3].detach() @ np.repeat(mano_rot_offset[None], transform_abs.shape[0], axis=0)
    # rotation matrix를 angle-axis로 변환 (원본 mano2dexhand.py와 일관성 유지)
    wrist_rot = torch.tensor(wrist_rot, dtype=torch.float32)
    wrist_rot = rotmat_to_aa(wrist_rot).detach()  # [T, 3] angle-axis 형태

    # wrist_pos_expanded = hand_keypoints[:, 0, :].unsqueeze(1).expand(-1, 21, -1)  # [T, 21, 3]
    # joints_rel = hand_keypoints - wrist_pos_expanded  # [T, 21, 3]
    
    # angle-axis를 rotation matrix로 변환
    # wrist_rot_matrix = aa_to_rotmat(wrist_rot)  # [T, 3, 3]
    
    # batch matrix multiplication으로 회전 적용
    # joints_rel_rotated = torch.bmm(wrist_rot_matrix, joints_rel.transpose(-2, -1))  # [T, 3, 21]
    # joints_rel_rotated = joints_rel_rotated.transpose(-2, -1)  # [T, 21, 3]
    
    # 손목 위치 추가 (wrist_pos를 tensor로 변환)
    # wrist_pos_tensor = torch.tensor(wrist_pos, dtype=torch.float32)
    # joints_in_dexhand = joints_rel_rotated + wrist_pos_tensor.unsqueeze(1)  # [T, 21, 3]

    gigahands_mano_joints = {
        "wrist": hand_keypoints[:, 0, :].detach(),           # 0
        "thumb_proximal": hand_keypoints[:, 1, :].detach(),  # 1 (mcp)
        "thumb_intermediate": hand_keypoints[:, 2, :].detach(), # 2 (pip)
        "thumb_distal": hand_keypoints[:, 3, :].detach(),    # 3 (dip)
        "thumb_tip": hand_keypoints[:, 4, :].detach(),       # 4
        "index_proximal": hand_keypoints[:, 5, :].detach(),  # 5 (mcp)
        "index_intermediate": hand_keypoints[:, 6, :].detach(), # 6 (pip)
        "index_distal": hand_keypoints[:, 7, :].detach(),    # 7 (dip)
        "index_tip": hand_keypoints[:, 8, :].detach(),       # 8
        "middle_proximal": hand_keypoints[:, 9, :].detach(), # 9 (mcp)
        "middle_intermediate": hand_keypoints[:, 10, :].detach(), # 10 (pip)
        "middle_distal": hand_keypoints[:, 11, :].detach(),  # 11 (dip)
        "middle_tip": hand_keypoints[:, 12, :].detach(),
        "ring_proximal": hand_keypoints[:, 13, :].detach(),  # 13 (mcp)
        "ring_intermediate": hand_keypoints[:, 14, :].detach(), # 14 (pip)
        "ring_distal": hand_keypoints[:, 15, :].detach(),    # 15 (dip)
        "ring_tip": hand_keypoints[:, 16, :].detach(),       # 16
        "pinky_proximal": hand_keypoints[:, 17, :].detach(), # 17 (mcp)
        "pinky_intermediate": hand_keypoints[:, 18, :].detach(), # 18 (pip)
        "pinky_distal": hand_keypoints[:, 19, :].detach(),   # 19 (dip)
        "pinky_tip": hand_keypoints[:, 20, :].detach()
    }
    

    
    # 손 좌우 반전 처리 (왼손을 오른손 좌표계로 변환)
    if side == "left":
        for name in gigahands_mano_joints:
            gigahands_mano_joints[name] = gigahands_mano_joints[name].clone()
            gigahands_mano_joints[name][:, 1] *= -1  # Y축 반전
        wrist_pos = wrist_pos.clone()
        wrist_pos[:, 1] *= -1
        wrist_rot = wrist_rot.clone()
        wrist_rot[:, 1] *= -1  # Y축 회전도 반전
        cprint("Applied left-to-right hand coordinate transformation", "cyan")
    
    wrist_rot = torch.tensor(wrist_rot, dtype=torch.float32)
    wrist_pos = torch.tensor(wrist_pos, dtype=torch.float32)
    # ManipTrans 형식으로 패킹 (원본 mano2dexhand.py와 동일한 구조)
    packed_data = {
        "wrist_pos": wrist_pos,
        "wrist_rot": wrist_rot,
        "mano_joints": gigahands_mano_joints
    }
    
    cprint(f"Packed {side} hand data: {T} frames, {len(gigahands_mano_joints)} joints", "green")
    cprint(f"Available joints: {list(gigahands_mano_joints.keys())}", "blue")
    
    return packed_data


def get_all_gigahands_sequences(data_dir):
    """GigaHands 데이터셋의 모든 시퀀스 정보를 가져오기"""
    annotation_file = os.path.join(data_dir, "annotations_v2.jsonl")
    
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    sequences = []
    with open(annotation_file, 'r', encoding='utf-8') as file:
        for line in file:
            script_info = json.loads(line)
            scene_name = script_info['scene']
            seq = script_info['sequence']
            
            # 키포인트 파일 경로 확인
            kp_path = os.path.join(data_dir, 'handpose', scene_name, 'keypoints_3d_mano', seq + '.json')
            
            if os.path.exists(kp_path):
                seq_id = f"{scene_name}/{seq}"
                sequences.append({
                    'seq_id': seq_id,
                    'scene': scene_name,
                    'sequence': seq,
                    'kp_path': kp_path,
                    'start_frame': script_info['start_frame_id'],
                    'end_frame': script_info['end_frame_id']
                })
    
    cprint(f"Found {len(sequences)} valid sequences", "green")
    return sequences


def load_gigahands_sequence(data_dir, seq_id, side="right"):
    """GigaHands 시퀀스 데이터 로드"""
    # annotation 파일에서 시퀀스 정보 찾기
    annotation_file = os.path.join(data_dir, "annotations_v2.jsonl")
    
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    # seq_id 파싱 (scene/sequence 또는 scene_sequence 형식)
    if '/' in seq_id:
        target_scene, target_seq = seq_id.split('/', 1)
    else:
        # 다른 형식도 지원 (예: "p001-folder_000")
        parts = seq_id.split('_')
        if len(parts) >= 2:
            target_scene = '_'.join(parts[:-1])
            target_seq = parts[-1]
        else:
            raise ValueError(f"Invalid sequence ID format: {seq_id}. Use format 'scene/sequence' or 'scene_sequence'")
    
    # 시퀀스 정보 찾기
    with open(annotation_file, 'r', encoding='utf-8') as file:
        for line in file:
            script_info = json.loads(line)
            scene_name = script_info['scene']
            seq = script_info['sequence']
            
            # 문자열 매칭
            if scene_name == target_scene and seq == target_seq:
                sf, ef = script_info['start_frame_id'], script_info['end_frame_id']
                
                # 키포인트 파일 경로 (handpose 디렉토리 기준)
                kp_path = os.path.join(data_dir, 'handpose', scene_name, 'keypoints_3d_mano', seq + '.json')
                
                if os.path.exists(kp_path):
                    # 키포인트 데이터 로드
                    with open(kp_path, "r") as f:
                        mano_kp = json.load(f)  # [F, 126]
                    
                    # numpy 배열로 변환 및 reshape
                    motion = np.array(mano_kp)  # [F, 126]
                    motion = motion.reshape(-1, 42, 3)  # [F, 42, 3]
                    
                    # 프레임 범위 적용
                    if ef == -1:
                        motion = motion[sf:]
                    else:
                        motion = motion[sf:ef+1]
                    
                    return torch.tensor(motion, dtype=torch.float32), scene_name, seq
    
    raise ValueError(f"Sequence {seq_id} not found in annotations. Available format: scene/sequence (e.g., 'p001-folder/000')")


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class Mano2DexhandGigaHands:
    def __init__(self, args, dexhand):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.headless = args.headless
        if self.headless:
            self.graphics_device_id = -1

        assert args.physics_engine == gymapi.SIM_PHYSX

        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu

        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"

        self.sim = self.gym.create_sim(
            args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        asset_root = os.path.split(self.dexhand.urdf_path)[0]
        asset_file = os.path.split(self.dexhand.urdf_path)[1]

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        dexhand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.chain = pk.build_chain_from_urdf(open(os.path.join(asset_root, asset_file)).read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        dexhand_dof_stiffness = torch.tensor(
            [10] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [1] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        
        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]

            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self._dexhand_effort_limits = torch.tensor(self._dexhand_effort_limits, device=self.sim_device)
        self._dexhand_dof_speed_limits = torch.tensor(self._dexhand_dof_speed_limits, device=self.sim_device)
        
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        if self.dexhand.name == "inspire":
            default_dof_state["pos"][8] = 0.8
            default_dof_state["pos"][9] = 0.05
        self.dexhand_default_dof_pos = default_dof_state
        
        self.dexhand_default_pose = gymapi.Transform()
        self.dexhand_default_pose.p = gymapi.Vec3(0, 0, 0)
        self.dexhand_default_pose.r = gymapi.Quat(0, 0, 0, 1)

        # 좌표계 변환 (GigaHands -> Isaac Gym)
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.5])  # 높이 조정
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        self.num_envs = args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.hand_idxs = []

        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            dexhand_actor = self.gym.create_actor(
                env,
                dexhand_asset,
                self.dexhand_default_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),
            )

            # Set initial DOF states
            self.gym.set_actor_dof_states(env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL)

            # Set DOF control properties
            self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)

            # MANO 키포인트 시각화용 구체들
            scene_asset_options = gymapi.AssetOptions()
            scene_asset_options.fix_base_link = True
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                joint_name = self.dexhand.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1
                )
                # 손가락별 색상 설정
                if "thumb" in joint_name:
                    c = gymapi.Vec3(1.0, 0.0, 0.0)  # 엄지 - 빨강
                elif "index" in joint_name:
                    c = gymapi.Vec3(1.0, 0.5, 0.0)  # 검지 - 주황
                elif "middle" in joint_name:
                    c = gymapi.Vec3(1.0, 1.0, 0.0)  # 중지 - 노랑
                elif "ring" in joint_name:
                    c = gymapi.Vec3(0.0, 1.0, 0.0)  # 약지 - 초록
                elif "pinky" in joint_name:
                    c = gymapi.Vec3(0.5, 0.0, 1.0)  # 새끼 - 보라
                elif "wrist" in joint_name or joint_name == "palm":
                    c = gymapi.Vec3(0.0, 1.0, 1.0)  # 손목/손바닥 - 하늘색
                else:
                    c = gymapi.Vec3(0.7, 0.7, 0.7)  # 기타 - 회색
                
                # 기존 복잡한 색상 로직 (주석처리)
                # if "index" in joint_name:
                #     inter_c = 70
                # elif "middle" in joint_name:
                #     inter_c = 130
                # elif "ring" in joint_name:
                #     inter_c = 190
                # elif "pinky" in joint_name:
                #     inter_c = 250
                # elif "thumb" in joint_name:
                #     inter_c = 10
                # else:
                #     inter_c = 0
                # if "tip" in joint_name:
                #     c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                # elif "proximal" in joint_name:
                #     c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                # elif "intermediate" in joint_name:
                #     c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                # else:
                #     c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env, a, 0, gymapi.MESH_VISUAL, c)

        env_ptr = self.envs[0]
        dexhand_handle = 0
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        self.dexhand_dof_handles = {
            k: self.gym.find_actor_dof_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.dof_names
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.q = self._dof_state[..., 0]
        self.qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        self.isaac2chain_order = [
            self.gym.get_actor_dof_names(env_ptr, dexhand_handle).index(j)
            for j in self.chain.get_joint_parameter_names()
        ]

        self.mano_joint_points = [
            self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
            for i in range(len(self.dexhand.body_names))
        ]

        if not self.headless:
            cam_pos = gymapi.Vec3(4, 3, 3)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

    def fitting(self, max_iter, target_wrist_pos, target_wrist_rot, target_mano_joints):
        """MANO 키포인트를 로봇 손으로 retargeting"""
        
        assert target_mano_joints.shape[0] == self.num_envs
        
        # 좌표계 변환
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        # STEP 1: MANO joint를 로컬에서 월드 좌표계로 변환
        # print(target_wrist_rot.shape, target_mano_joints.shape)
        # # 배치 행렬 곱셈: [B, 3, 3] @ [B, 3, N] = [B, 3, N]
        # target_mano_joints = torch.bmm(
        #     target_wrist_rot, 
        #     target_mano_joints.transpose(-2, -1)
        # ).transpose(-2, -1) + target_wrist_pos[:, None]

        # # STEP 2: mujoco2gym 좌표계로 변환
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (
            (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T
            + self.mujoco2gym_transf[:3, 3]
        )
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

        # target_mano_joints = target_mano_joints.view(-1, 3)
        # target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        # target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

        # 최적화 변수 초기화
        opt_wrist_pos = torch.tensor(
            target_wrist_pos,
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt_wrist_rot = torch.tensor(
            rotmat_to_rot6d(target_wrist_rot), device=self.sim_device, dtype=torch.float32, requires_grad=True
        )
        opt_dof_pos = torch.tensor(
            self.dexhand_default_dof_pos["pos"][None].repeat(self.num_envs, axis=0),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        
        # Adam optimizer supports parameter groups with different learning rates
        opti = torch.optim.Adam(
            [{"params": [opt_dof_pos], "lr": 0.0005}, {"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.001}]
        )

        # 손가락별 가중치 설정
        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "tip" in k:
                if "index" in k:
                    weight.append(20)
                elif "middle" in k:
                    weight.append(10)
                elif "ring" in k:
                    weight.append(25)
                elif "pinky" in k:
                    weight.append(10)
                elif "thumb" in k:
                    weight.append(25)
                else:
                    weight.append(1)
            elif "proximal" in k:
                weight.append(1)
            elif "intermediate" in k:
                weight.append(1)
            else:
                weight.append(1)
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)
        
        iter = 0
        past_loss = 1e10
        convergence_count = 0  # 연속 수렴 카운터
        
        while (self.headless and iter < max_iter) or (
            not self.headless and not self.gym.query_viewer_has_closed(self.viewer)
        ):
            iter += 1

            # Use updated tensors for simulation
            opt_wrist_quat = rot6d_to_quat(opt_wrist_rot)[:, [1, 2, 3, 0]]
            opt_wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot)
            self._root_state[:, 0, :3] = opt_wrist_pos.detach()
            self._root_state[:, 0, 3:7] = opt_wrist_quat.detach()
            self._root_state[:, 0, 7:] = torch.zeros_like(self._root_state[:, 0, 7:])

            opt_dof_pos_clamped = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)
            
            # 디버깅: clamp 효과 확인
            if iter % 200 == 0:
                clamped_count = ((opt_dof_pos <= self.dexhand_dof_lower_limits + 1e-6) | 
                               (opt_dof_pos >= self.dexhand_dof_upper_limits - 1e-6)).sum().item()
                cprint(f"Iter {iter}: {clamped_count}/{opt_dof_pos.numel()} DOFs at limits", "magenta")

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(opt_dof_pos_clamped))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self.headless:
                self.gym.step_graphics(self.sim)

            # Update tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            
            # Step rendering
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            isaac_joints = torch.stack(
                [self._rigid_body_state[:, self.dexhand_handles[k], :3] for k in self.dexhand.body_names],
                dim=1,
            )
            with torch.no_grad():

                ret = self.chain.forward_kinematics(opt_dof_pos_clamped[:, self.isaac2chain_order])
                pk_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)
                pk_joints = (rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)).transpose(
                    -1, -2
                ) + opt_wrist_pos[:, None]

            target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
            
            # MANO 키포인트 시각화 업데이트
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]
            
            # Adam optimization
            opti.zero_grad()
            
            # Forward kinematics calculation
            opt_dof_pos_clamped_local = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)
            ret = self.chain.forward_kinematics(opt_dof_pos_clamped_local[:, self.isaac2chain_order])
            pk_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)
            pk_joints = (rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)).transpose(
                -1, -2
            ) + opt_wrist_pos[:, None]
            
            # Loss calculation with weighted joint and rotation errors
            joint_errors = torch.norm(pk_joints - target_joints, dim=-1)  # [B, N_joints]
            rotation_error = torch.norm(pk_joints[:, 0] - target_joints[:, 0], dim=-1)  # wrist joint error
            joint_loss = torch.mean(joint_errors * weight[None]) * 10.0
            rotation_loss = torch.mean(rotation_error) * 2.0  # Additional weight for wrist rotation
            loss = joint_loss + rotation_loss
            loss.backward()
            
            # Adam optimization step
            opti.step()
            
            loss_value = loss.item()
            
            # Print gradient info
            if iter % 100 == 0:
                dof_grad_norm = opt_dof_pos.grad.norm().item() if opt_dof_pos.grad is not None else 0.0
                wrist_pos_grad_norm = opt_wrist_pos.grad.norm().item() if opt_wrist_pos.grad is not None else 0.0
                wrist_rot_grad_norm = opt_wrist_rot.grad.norm().item() if opt_wrist_rot.grad is not None else 0.0
                
                cprint(f"{iter} loss:{loss_value:.6f} grad[dof:{dof_grad_norm:.4f}, pos:{wrist_pos_grad_norm:.4f}, rot:{wrist_rot_grad_norm:.4f}]", "green")
                if iter > 500:  # 최소 500번 실행 후 수렴 체크
                    loss_diff = past_loss - loss_value
                    if loss_diff < 1e-4:  # 수렴 조건 완화
                        cprint(f"Converged at iteration {iter} (loss diff: {loss_diff:.2e})", "cyan")
                        break
                past_loss = loss_value
            
            # 매우 작은 loss에서 조기 종료
            if iter % 50 == 0 and loss_value < 1e-3:
                cprint(f"Early stopping at iteration {iter} (loss: {loss_value:.2e})", "cyan")
                break

        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
        }

        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return to_dump


if __name__ == "__main__":
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
                "default": "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/retargeted",
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
            }
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    def run(parser, data_idx):
        # data_idx 파싱 (예: "20aed@0")
        if "@" in data_idx:
            seq_id, frame_offset = data_idx.split("@")
            frame_offset = int(frame_offset)
        else:
            seq_id = data_idx
            frame_offset = 0
        
        cprint(f"Loading GigaHands sequence: {seq_id} with offset {frame_offset}", "blue")
        
        # GigaHands 데이터 로드
        motion_data, scene_name, sequence_name = load_gigahands_sequence(
            parser.data_dir, seq_id, parser.side
        )
        
        # 프레임 오프셋 적용
        if frame_offset > 0 and frame_offset < motion_data.shape[0]:
            motion_data = motion_data[frame_offset:]
        
        cprint(f"Motion data shape: {motion_data.shape}", "green")
        cprint(f"Scene: {scene_name}, Sequence: {sequence_name}", "green")
        
        # DexHand 조인트 이름 확인
        cprint("DexHand body names:", "cyan")
        for j_name in dexhand.body_names:
            hand_joint_name = dexhand.to_hand(j_name)[0]
            cprint(f"  {j_name} -> {hand_joint_name}", "cyan")
        
        # ManipTrans 형식으로 변환 (원본과 동일한 방식)
        demo_data = pack_gigahands_data(motion_data, dexhand, parser.side)
        
        # 텐서들을 GPU로 이동 (device 일치)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        demo_data["wrist_pos"] = demo_data["wrist_pos"].to(device)
        demo_data["wrist_rot"] = demo_data["wrist_rot"].to(device)
        for joint_name in demo_data["mano_joints"]:
            demo_data["mano_joints"][joint_name] = demo_data["mano_joints"][joint_name].to(device)
        
        parser.num_envs = demo_data["mano_joints"]["wrist"].shape[0]
        
        cprint(f"Number of environments: {parser.num_envs}", "blue")

        # MANO to Dexhand 변환
        mano2inspire = Mano2DexhandGigaHands(parser, dexhand)
        
        # MANO 조인트를 하나의 텐서로 결합 (원본 mano2dexhand.py와 동일한 방식)
        mano_joints_tensor = torch.cat([
            demo_data["mano_joints"][dexhand.to_hand(j_name)[0]]
            for j_name in dexhand.body_names
            if dexhand.to_hand(j_name)[0] != "wrist"
        ], dim=-1)

        to_dump = mano2inspire.fitting(
            parser.iter,
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            mano_joints_tensor,
        )

        # 결과 저장 - 원본과 동일한 디렉토리 구조 유지
        # seq_id 파싱 (예: "p001-folder/001")
        parts = seq_id.split("/")
        if len(parts) == 2:
            scene_folder, sequence_id = parts
            # 원본과 동일한 구조로 저장
            dump_dir = os.path.join(
                parser.save_dir, 
                f"mano2{str(dexhand)}", 
                scene_folder, 
                "keypoints_3d_mano"
            )
        else:
            # 예외 케이스: flat 구조로 저장
            dump_dir = f"{parser.save_dir}/mano2{str(dexhand)}"
            
        os.makedirs(dump_dir, exist_ok=True)
        
        # 파일명은 간단하게
        dump_path = os.path.join(dump_dir, f"{sequence_id}_retargeted.pkl")
        
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)
            
        cprint(f"Retargeted data saved to: {dump_path}", "green")

    def run_batch_processing(parser):
        """모든 GigaHands 시퀀스를 배치 처리"""
        cprint("Starting batch processing of all GigaHands sequences", "cyan")
        
        # 모든 시퀀스 정보 가져오기
        sequences = get_all_gigahands_sequences(parser.data_dir)
        
        if parser.max_sequences > 0:
            sequences = sequences[:parser.max_sequences]
            cprint(f"Limited to {len(sequences)} sequences", "yellow")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for i, seq_info in enumerate(sequences):
            seq_id = seq_info['seq_id']
            scene_name = seq_info['scene']
            sequence_name = seq_info['sequence']
            
            cprint(f"\n[{i+1}/{len(sequences)}] Processing: {seq_id}", "blue")
            
            # 출력 파일 경로 확인 - 원본과 동일한 구조 유지
            parts = seq_id.split("/")
            if len(parts) == 2:
                scene_folder, sequence_id = parts
                dump_dir = os.path.join(
                    parser.save_dir, 
                    f"mano2{str(dexhand)}", 
                    scene_folder, 
                    "keypoints_3d_mano"
                )
            else:
                dump_dir = f"{parser.save_dir}/mano2{str(dexhand)}"
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, f"{sequence_id}_retargeted.pkl")
            
            # 이미 처리된 파일 스킵
            if parser.skip_existing and os.path.exists(dump_path):
                cprint(f"  Skipping (already exists): {dump_path}", "yellow")
                skipped_count += 1
                continue
            
            # try:
            # 개별 시퀀스 처리
            run(parser, seq_id)
            processed_count += 1
            cprint(f"  ✓ Successfully processed: {seq_id}", "green")
                
            # except Exception as e:
            #     cprint(f"  ✗ Error processing {seq_id}: {str(e)}", "red")
            #     error_count += 1
            #     continue
        
        # 최종 결과 출력
        cprint(f"\n=== Batch Processing Complete ===", "cyan")
        cprint(f"Total sequences: {len(sequences)}", "white")
        cprint(f"Successfully processed: {processed_count}", "green")
        cprint(f"Skipped (existing): {skipped_count}", "yellow")
        cprint(f"Errors: {error_count}", "red")

    # 실행 로직
    if _parser.data_idx.lower() == "all":
        run_batch_processing(_parser)
    else:
        run(_parser, _parser.data_idx) 