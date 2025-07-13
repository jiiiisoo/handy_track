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


def get_zero_pose_mano_keypoints(use_pca=True, ncomps=6):
    """Zero pose MANO 키포인트들과 메쉬 추출"""
    
    cprint("🤚 Extracting zero pose MANO keypoints and mesh...", "cyan")
    
    # MANO 모델 로드
    mano_layer = ManoLayer(
        mano_root='/workspace/manopth/mano/models', 
        use_pca=use_pca, 
        ncomps=ncomps, 
        flat_hand_mean=True
    )
    cprint("✅ MANO model loaded", "green")
    
    # Zero pose 파라미터 설정
    batch_size = 1
    pose = torch.zeros(batch_size, ncomps + 3)  # 모든 pose 파라미터 0 (zero pose)
    shape = torch.zeros(batch_size, 10)  # 모든 shape 파라미터 0 (average shape)
    
    # Forward pass
    with torch.no_grad():
        hand_verts, hand_joints, transform_abs = mano_layer(pose, shape)
    
    # NumPy로 변환
    vertices = hand_verts[0].numpy()  # [778, 3]
    keypoints = hand_joints[0].numpy()  # [21, 3]  
    
    cprint(f"✅ Zero pose MANO: {vertices.shape[0]} vertices, {keypoints.shape[0]} keypoints", "green")
    cprint(f"Pose parameters shape: {pose.shape}", "blue")
    cprint(f"Shape parameters shape: {shape.shape}", "blue")
    
    return vertices, keypoints, transform_abs, pose, shape, mano_layer


def pack_zero_pose_mano_data(dexhand, side="right"):
    """Zero pose MANO 데이터를 ManipTrans 형식으로 변환"""
    
    cprint("📦 Packing zero pose MANO data for ManipTrans format...", "cyan")
    
    # Zero pose MANO 키포인트 추출
    vertices, hand_keypoints, transform_abs, pose, shape, mano_layer = get_zero_pose_mano_keypoints()
    
    # Tensor로 변환
    hand_keypoints = torch.tensor(hand_keypoints, dtype=torch.float32).unsqueeze(0)  # [1, 21, 3]
    
    # 손목 위치 계산 (hack for wrist position)
    wrist_pos = hand_keypoints[:, 0, :]  # [1, 3]
    middle_pos = hand_keypoints[:, 9, :]  # [1, 3] - middle_mcp
    wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25
    
    # DexHand 설정
    dexhand_obj = DexHandFactory.create_hand(dexhand_type=dexhand, side=side)
    wrist_pos += torch.tensor(dexhand_obj.relative_translation, dtype=torch.float32).unsqueeze(0)
    mano_rot_offset = dexhand_obj.relative_rotation
    
    # 손목 회전 계산
    wrist_rot_matrix = transform_abs[:, 0, :3, :3].detach() @ torch.tensor(mano_rot_offset, dtype=torch.float32).unsqueeze(0)
    wrist_rot = rotmat_to_aa(wrist_rot_matrix).detach()  # [1, 3] angle-axis 형태
    
    # 상대 좌표 계산 및 회전 적용
    wrist_pos_expanded = hand_keypoints[:, 0, :].unsqueeze(1).expand(-1, 21, -1)  # [1, 21, 3]
    joints_rel = hand_keypoints - wrist_pos_expanded  # [1, 21, 3]
    
    # angle-axis를 rotation matrix로 변환
    wrist_rot_matrix = aa_to_rotmat(wrist_rot)  # [1, 3, 3]
    
    # batch matrix multiplication으로 회전 적용
    joints_rel_rotated = torch.bmm(wrist_rot_matrix, joints_rel.transpose(-2, -1))  # [1, 3, 21]
    joints_rel_rotated = joints_rel_rotated.transpose(-2, -1)  # [1, 21, 3]
    
    # 손목 위치 추가
    wrist_pos_tensor = torch.tensor(wrist_pos, dtype=torch.float32)
    joints_in_dexhand = joints_rel_rotated + wrist_pos_tensor.unsqueeze(1)  # [1, 21, 3]
    
    # Zero pose MANO 조인트 매핑 (21개)
    zero_pose_mano_joints = {
        "wrist": joints_in_dexhand[:, 0, :].squeeze(0),           # 0
        "thumb_proximal": joints_in_dexhand[:, 1, :].squeeze(0),  # 1 (mcp)
        "thumb_intermediate": joints_in_dexhand[:, 2, :].squeeze(0), # 2 (pip)
        "thumb_distal": joints_in_dexhand[:, 3, :].squeeze(0),    # 3 (dip)
        "thumb_tip": joints_in_dexhand[:, 4, :].squeeze(0),       # 4
        "index_proximal": joints_in_dexhand[:, 5, :].squeeze(0),  # 5 (mcp)
        "index_intermediate": joints_in_dexhand[:, 6, :].squeeze(0), # 6 (pip)
        "index_distal": joints_in_dexhand[:, 7, :].squeeze(0),    # 7 (dip)
        "index_tip": joints_in_dexhand[:, 8, :].squeeze(0),       # 8
        "middle_proximal": joints_in_dexhand[:, 9, :].squeeze(0), # 9 (mcp)
        "middle_intermediate": joints_in_dexhand[:, 10, :].squeeze(0), # 10 (pip)
        "middle_distal": joints_in_dexhand[:, 11, :].squeeze(0),  # 11 (dip)
        "middle_tip": joints_in_dexhand[:, 12, :].squeeze(0),     # 12
        "ring_proximal": joints_in_dexhand[:, 13, :].squeeze(0),  # 13 (mcp)
        "ring_intermediate": joints_in_dexhand[:, 14, :].squeeze(0), # 14 (pip)
        "ring_distal": joints_in_dexhand[:, 15, :].squeeze(0),    # 15 (dip)
        "ring_tip": joints_in_dexhand[:, 16, :].squeeze(0),       # 16
        "pinky_proximal": joints_in_dexhand[:, 17, :].squeeze(0), # 17 (mcp)
        "pinky_intermediate": joints_in_dexhand[:, 18, :].squeeze(0), # 18 (pip)
        "pinky_distal": joints_in_dexhand[:, 19, :].squeeze(0),   # 19 (dip)
        "pinky_tip": joints_in_dexhand[:, 20, :].squeeze(0)       # 20
    }
    
    # 손 좌우 반전 처리 (왼손을 오른손 좌표계로 변환)
    if side == "left":
        for name in zero_pose_mano_joints:
            zero_pose_mano_joints[name] = zero_pose_mano_joints[name].clone()
            zero_pose_mano_joints[name][1] *= -1  # Y축 반전
        wrist_pos = wrist_pos.clone()
        wrist_pos[:, 1] *= -1
        wrist_rot = wrist_rot.clone()
        wrist_rot[:, 1] *= -1  # Y축 회전도 반전
        cprint("Applied left-to-right hand coordinate transformation", "cyan")
    
    # ManipTrans 형식으로 패킹
    packed_data = {
        "wrist_pos": wrist_pos.squeeze(0),  # [3]
        "wrist_rot": wrist_rot.squeeze(0),  # [3]
        "mano_joints": zero_pose_mano_joints,
        "pose_params": pose,                # MANO pose parameters
        "shape_params": shape,              # MANO shape parameters
        "vertices": torch.tensor(vertices, dtype=torch.float32),  # MANO mesh vertices
        "transform_abs": transform_abs      # Absolute transformations
    }
    
    cprint(f"Packed zero pose {side} hand data: {len(zero_pose_mano_joints)} joints", "green")
    cprint(f"Available joints: {list(zero_pose_mano_joints.keys())}", "blue")
    cprint(f"Pose parameters: {pose.shape}, Shape parameters: {shape.shape}", "blue")
    
    return packed_data


class ZeroPoseManoTracker:
    """Zero pose MANO tracking을 위한 클래스"""
    
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

        # 좌표계 변환 (MANO -> Isaac Gym) - 높이 조정
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, 1.0])  # 높이를 0.5에서 1.0으로 증가
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        self.num_envs = 1  # Zero pose tracking은 단일 환경

        # 환경 생성
        num_per_row = 1
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.hand_idxs = []

        # 단일 환경 생성
        env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
        self.envs.append(env)
        dexhand_actor = self.gym.create_actor(
            env,
            dexhand_asset,
            self.dexhand_default_pose,
            "dexhand",
            0,
            (1 if self.dexhand.self_collision else 0),
        )

        # 초기 DOF 상태 설정
        self.gym.set_actor_dof_states(env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL)
        self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)

        # MANO 키포인트 시각화용 구체들 (크기 증가)
        scene_asset_options = gymapi.AssetOptions()
        scene_asset_options.fix_base_link = True
        for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
            joint_name = self.dexhand.to_hand(joint_name)[0]
            joint_point = self.gym.create_sphere(self.sim, 0.02, scene_asset_options)  # 0.005 -> 0.02로 크기 증가
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
            # 카메라 위치를 손 위치에 맞게 조정
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.5)  # 손 위쪽에서 보기
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)  # 변환된 손목 위치 쪽을 바라보기
            self.gym.viewer_camera_look_at(self.viewer, env_ptr, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

    def track_zero_pose(self, max_iter, target_wrist_pos, target_wrist_rot, target_mano_joints):
        """Zero pose MANO 키포인트를 로봇 손으로 tracking"""
        
        # 좌표계 변환 (디버깅 정보 추가)
        cprint(f"원본 wrist_pos: {target_wrist_pos}", "yellow")
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        cprint(f"변환된 wrist_pos: {target_wrist_pos}", "yellow")
        
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)
        cprint(f"변환된 mano_joints 범위: min={target_mano_joints.min():.3f}, max={target_mano_joints.max():.3f}", "yellow")

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
        
        opti = torch.optim.Adam(
            [{"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.0008}, {"params": [opt_dof_pos], "lr": 0.0004}]
        )

        # 손가락별 가중치 설정 (zero pose에 맞게 조정)
        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "tip" in k:
                weight.append(20)  # tip joints 강조
            elif "proximal" in k:
                weight.append(10)  # proximal joints 중간 가중치
            elif "intermediate" in k:
                weight.append(10)  # intermediate joints 중간 가중치
            else:
                weight.append(5)   # 기타 joints
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)
        
        iter = 0
        past_loss = 1e10
        
        cprint(f"Starting zero pose tracking optimization...", "cyan")
        
        while (self.headless and iter < max_iter) or (
            not self.headless and not self.gym.query_viewer_has_closed(self.viewer)
        ):
            iter += 1

            opt_wrist_quat = rot6d_to_quat(opt_wrist_rot)[:, [1, 2, 3, 0]]
            opt_wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot)
            self._root_state[:, 0, :3] = opt_wrist_pos.detach()
            self._root_state[:, 0, 3:7] = opt_wrist_quat.detach()
            self._root_state[:, 0, 7:] = torch.zeros_like(self._root_state[:, 0, 7:])

            opt_dof_pos_clamped = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)

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

            ret = self.chain.forward_kinematics(opt_dof_pos_clamped[:, self.isaac2chain_order])
            pk_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)
            pk_joints = (rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)).transpose(
                -1, -2
            ) + opt_wrist_pos[:, None]

            target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
            
            # MANO 키포인트 시각화 업데이트
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]
                
            loss = torch.mean(torch.norm(pk_joints - target_joints, dim=-1) * weight[None])
            opti.zero_grad()
            loss.backward()
            opti.step()

            if iter % 100 == 0 or iter == 1:  # 첫 번째 iteration도 출력
                cprint(f"Zero pose tracking iter {iter}: loss {loss.item():.6f}", "green")
                if iter == 1:  # 첫 번째 iteration에서 위치 정보 출력
                    cprint(f"  DexHand 위치: {opt_wrist_pos}", "blue")
                    cprint(f"  타겟 위치: {target_wrist_pos}", "blue")
                    cprint(f"  Isaac joints 위치 범위: min={isaac_joints.min():.3f}, max={isaac_joints.max():.3f}", "blue")
                if iter > 500:  # 최소 500번 실행 후 수렴 체크
                    loss_diff = past_loss - loss.item()
                    if loss_diff < 1e-4:  # 수렴 조건
                        cprint(f"Zero pose tracking converged at iteration {iter} (loss diff: {loss_diff:.2e})", "cyan")
                        break
                past_loss = loss.item()
            
            # 매우 작은 loss에서 조기 종료
            if iter % 50 == 0 and loss.item() < 1e-4:
                cprint(f"Zero pose tracking early stopping at iteration {iter} (loss: {loss.item():.2e})", "cyan")
                break

        tracked_result = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
            "final_loss": loss.item(),
            "total_iterations": iter
        }

        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        
        cprint(f"✅ Zero pose tracking completed! Final loss: {loss.item():.6f}", "green")
        return tracked_result


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="MANO Zero Pose Tracking",
        headless=False,  # GUI 보이도록 변경
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 3000,
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
                "name": "--save_dir",
                "type": str,
                "default": "/workspace/ManipTrans",
                "help": "Directory to save tracked zero pose data"
            }
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    cprint("🚀 Starting MANO Zero Pose Tracking...", "cyan", attrs=['bold'])
    
    # Zero pose MANO 데이터 생성
    zero_pose_data = pack_zero_pose_mano_data(dexhand=_parser.dexhand, side=_parser.side)
    
    cprint(f"📊 Zero pose data extracted:", "blue")
    cprint(f"  - Wrist position: {zero_pose_data['wrist_pos'].shape} = {zero_pose_data['wrist_pos']}", "blue")
    cprint(f"  - Wrist rotation: {zero_pose_data['wrist_rot'].shape} = {zero_pose_data['wrist_rot']}", "blue")
    cprint(f"  - MANO joints: {len(zero_pose_data['mano_joints'])}", "blue")
    cprint(f"  - Pose params: {zero_pose_data['pose_params'].shape}", "blue")
    cprint(f"  - Shape params: {zero_pose_data['shape_params'].shape}", "blue")
    
    # 텐서들을 GPU로 이동
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    zero_pose_data["wrist_pos"] = zero_pose_data["wrist_pos"].to(device).unsqueeze(0)  # [1, 3]
    zero_pose_data["wrist_rot"] = zero_pose_data["wrist_rot"].to(device).unsqueeze(0)  # [1, 3]
    
    for joint_name in zero_pose_data["mano_joints"]:
        zero_pose_data["mano_joints"][joint_name] = zero_pose_data["mano_joints"][joint_name].to(device).unsqueeze(0)  # [1, 3]
    
    _parser.num_envs = 1  # 단일 환경

    cprint(f"Number of environments: {_parser.num_envs}", "blue")

    # Zero pose tracking 실행
    zero_pose_tracker = ZeroPoseManoTracker(_parser, dexhand)
    
    # MANO 조인트를 하나의 텐서로 결합
    mano_joints_tensor = torch.cat([
        zero_pose_data["mano_joints"][dexhand.to_hand(j_name)[0]]
        for j_name in dexhand.body_names
        if dexhand.to_hand(j_name)[0] != "wrist"
    ], dim=-1).view(_parser.num_envs, -1, 3)

    tracked_result = zero_pose_tracker.track_zero_pose(
        _parser.iter,
        zero_pose_data["wrist_pos"],
        zero_pose_data["wrist_rot"],
        mano_joints_tensor,
    )

    # 결과 저장
    save_path = os.path.join(_parser.save_dir, f"zero_pose_mano2{str(dexhand)}_tracked.pkl")
    
    # 원본 MANO 데이터도 함께 저장
    full_result = {
        "tracked_data": tracked_result,
        "original_mano_data": {
            "wrist_pos": zero_pose_data["wrist_pos"].cpu().numpy(),
            "wrist_rot": zero_pose_data["wrist_rot"].cpu().numpy(),
            "mano_joints": {k: v.cpu().numpy() for k, v in zero_pose_data["mano_joints"].items()},
            "pose_params": zero_pose_data["pose_params"].cpu().numpy(),
            "shape_params": zero_pose_data["shape_params"].cpu().numpy(),
            "vertices": zero_pose_data["vertices"].cpu().numpy()
        },
        "dexhand_info": {
            "name": dexhand.name,
            "side": _parser.side,
            "body_names": dexhand.body_names,
            "dof_names": dexhand.dof_names
        }
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(full_result, f)
        
    cprint(f"✅ Zero pose tracking data saved to: {save_path}", "green")
    cprint(f"📈 Final tracking loss: {tracked_result['final_loss']:.6f}", "yellow")
    cprint(f"🔄 Total iterations: {tracked_result['total_iterations']}", "yellow")
