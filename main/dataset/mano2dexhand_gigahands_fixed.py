# CRITICAL: Import isaacgym modules FIRST before any other imports
from isaacgym import gymapi, gymtorch, gymutil

import math
import os
import pickle
import torch
import logging
import numpy as np
import pytorch_kinematics as pk
from termcolor import cprint
import json

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


def pack_gigahands_data_fixed(data_item, dexhand, side="right"):
    """GigaHands 데이터를 ManipTrans 형식으로 변환 (수정된 버전)"""
    
    motion_data = data_item
    T = motion_data.shape[0]
    
    if not isinstance(motion_data, torch.Tensor):
        motion_data = torch.tensor(motion_data, dtype=torch.float32)
    
    # 손 구분에 따른 키포인트 추출
    if side == "right":
        hand_keypoints = motion_data[:, :21, :]  # [T, 21, 3]
    elif side == "left":
        hand_keypoints = motion_data[:, 21:42, :]  # [T, 21, 3]
        # 왼손을 오른손 좌표계로 변환 (Y축 반전)
        hand_keypoints = hand_keypoints.clone()
        hand_keypoints[:, :, 1] *= -1
    else:
        raise ValueError(f"Invalid side: {side}")
    
    # 기본 wrist position과 rotation 추출
    wrist_pos = hand_keypoints[:, 0, :].numpy()  # [T, 3]
    
    # 간단한 wrist rotation 추정 (index finger 방향 기반)
    wrist_to_index = hand_keypoints[:, 8, :] - hand_keypoints[:, 0, :]  # index_tip - wrist
    wrist_to_index = wrist_to_index / (torch.norm(wrist_to_index, dim=-1, keepdim=True) + 1e-8)
    
    # Forward direction [1,0,0]에서 wrist_to_index로의 회전
    forward = torch.tensor([1.0, 0.0, 0.0]).expand_as(wrist_to_index)
    rotation_axis = torch.cross(forward, wrist_to_index, dim=-1)
    rotation_axis = rotation_axis / (torch.norm(rotation_axis, dim=-1, keepdim=True) + 1e-8)
    cos_angle = torch.sum(forward * wrist_to_index, dim=-1, keepdim=True)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    wrist_rot = (rotation_axis * angle).numpy()  # [T, 3] axis-angle
    
    # DEXHAND-SPECIFIC ADJUSTMENT (원본 OakInk 방식과 동일)
    # 1. Wrist position adjustment using middle finger
    middle_pos = hand_keypoints[:, 9, :].numpy()  # middle_proximal
    wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # hack for wrist position
    
    # 2. Dexhand robot specific offset 적용
    wrist_pos += np.array(dexhand.relative_translation)
    
    # 3. Wrist rotation에 dexhand offset 적용
    mano_rot_offset = dexhand.relative_rotation
    wrist_rot_matrix = aa_to_rotmat(torch.tensor(wrist_rot, dtype=torch.float32))
    dexhand_rot_offset = torch.tensor(mano_rot_offset, dtype=torch.float32).expand(T, 3, 3)
    wrist_rot_matrix = wrist_rot_matrix @ dexhand_rot_offset
    wrist_rot = rotmat_to_aa(wrist_rot_matrix).numpy()
    
    # MANO joints mapping (wrist 제외)
    gigahands_mano_joints = {
        "thumb_proximal": hand_keypoints[:, 1, :],
        "thumb_intermediate": hand_keypoints[:, 2, :],
        "thumb_distal": hand_keypoints[:, 3, :],
        "thumb_tip": hand_keypoints[:, 4, :],
        "index_proximal": hand_keypoints[:, 5, :],
        "index_intermediate": hand_keypoints[:, 6, :],
        "index_distal": hand_keypoints[:, 7, :],
        "index_tip": hand_keypoints[:, 8, :],
        "middle_proximal": hand_keypoints[:, 9, :],
        "middle_intermediate": hand_keypoints[:, 10, :],
        "middle_distal": hand_keypoints[:, 11, :],
        "middle_tip": hand_keypoints[:, 12, :],
        "ring_proximal": hand_keypoints[:, 13, :],
        "ring_intermediate": hand_keypoints[:, 14, :],
        "ring_distal": hand_keypoints[:, 15, :],
        "ring_tip": hand_keypoints[:, 16, :],
        "pinky_proximal": hand_keypoints[:, 17, :],
        "pinky_intermediate": hand_keypoints[:, 18, :],
        "pinky_distal": hand_keypoints[:, 19, :],
        "pinky_tip": hand_keypoints[:, 20, :]
    }
    
    packed_data = {
        "wrist_pos": torch.tensor(wrist_pos, dtype=torch.float32),
        "wrist_rot": torch.tensor(wrist_rot, dtype=torch.float32),
        "mano_joints": gigahands_mano_joints
    }
    
    cprint(f"Packed {side} hand data: {T} frames, {len(gigahands_mano_joints)} joints", "green")
    return packed_data


def load_gigahands_sequence(data_dir, seq_id, side="right"):
    """GigaHands 시퀀스 데이터 로드"""
    annotation_file = os.path.join(data_dir, "annotations_v2.jsonl")
    
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    # seq_id 파싱
    if '/' in seq_id:
        target_scene, target_seq = seq_id.split('/', 1)
    else:
        parts = seq_id.split('_')
        if len(parts) >= 2:
            target_scene = '_'.join(parts[:-1])
            target_seq = parts[-1]
        else:
            raise ValueError(f"Invalid sequence ID format: {seq_id}")
    
    # 시퀀스 정보 찾기
    with open(annotation_file, 'r', encoding='utf-8') as file:
        for line in file:
            script_info = json.loads(line)
            scene_name = script_info['scene']
            seq = script_info['sequence']
            
            if scene_name == target_scene and seq == target_seq:
                sf, ef = script_info['start_frame_id'], script_info['end_frame_id']
                
                kp_path = os.path.join(data_dir, 'handpose', scene_name, 'keypoints_3d_mano', seq + '.json')
                
                if os.path.exists(kp_path):
                    with open(kp_path, "r") as f:
                        mano_kp = json.load(f)
                    
                    motion = np.array(mano_kp).reshape(-1, 42, 3)
                    
                    if ef == -1:
                        motion = motion[sf:]
                    else:
                        motion = motion[sf:ef+1]
                    
                    return torch.tensor(motion, dtype=torch.float32), scene_name, seq
    
    raise ValueError(f"Sequence {seq_id} not found")


class Mano2DexhandGigaHandsFixed:
    def __init__(self, args, dexhand):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.headless = args.headless

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

        # Ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Load dexhand asset
        asset_root = os.path.split(self.dexhand.urdf_path)[0]
        asset_file = os.path.split(self.dexhand.urdf_path)[1]

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        dexhand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # PyTorch kinematics chain
        self.chain = pk.build_chain_from_urdf(open(os.path.join(asset_root, asset_file)).read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        # DOF properties
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)
        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        
        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = 10
            dexhand_dof_props["damping"][i] = 1
            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        
        # Default DOF state
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        if self.dexhand.name == "inspire":
            default_dof_state["pos"][8] = 0.8
            default_dof_state["pos"][9] = 0.05
        self.dexhand_default_dof_pos = default_dof_state
        
        # Coordinate transformation (GigaHands -> Isaac Gym)
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.5])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        # Create environments
        self.num_envs = args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            
            dexhand_actor = self.gym.create_actor(
                env, dexhand_asset, gymapi.Transform(), "dexhand", i, 
                (1 if self.dexhand.self_collision else 0)
            )
            
            self.gym.set_actor_dof_states(env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL)
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
                else:
                    c = gymapi.Vec3(0.0, 1.0, 1.0)  # 기타 - 하늘색
                
                self.gym.set_rigid_body_color(env, a, 0, gymapi.MESH_VISUAL, c)

        # Isaac Gym handles and tensors
        env_ptr = self.envs[0]
        dexhand_handle = 0
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) 
            for k in self.dexhand.body_names
        }
        
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

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
        """MANO 키포인트를 로봇 손으로 retargeting (수정된 버전)"""
        
        assert target_mano_joints.shape[0] == self.num_envs
        
        # 좌표계 변환 (이미 dexhand-specific adjustment가 적용된 상태)
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

        # 안전 margin 추가 (object 없어도 테이블과의 충돌 방지)
        safety_offset = torch.tensor([0.0, 0.0, 0.05], device=self.sim_device)  # 5cm 위로
        
        # 최적화 변수 초기화 (TARGET 근처에서 시작)
        opt_wrist_pos = torch.tensor(
            target_wrist_pos + safety_offset,  # Safety margin 추가
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt_wrist_rot = torch.tensor(
            rotmat_to_rot6d(target_wrist_rot), 
            device=self.sim_device, 
            dtype=torch.float32, 
            requires_grad=True
        )
        opt_dof_pos = torch.tensor(
            self.dexhand_default_dof_pos["pos"][None].repeat(self.num_envs, axis=0),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        
        # Optimizer (wrist는 fine-tuning, DOF는 full search)
        opti = torch.optim.Adam(
            [{"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.0008}, 
             {"params": [opt_dof_pos], "lr": 0.0004}]
        )

        # 손가락별 가중치 설정
        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "tip" in k:
                if "thumb" in k:
                    weight.append(25)
                elif "index" in k:
                    weight.append(20)
                else:
                    weight.append(10)
            elif "proximal" in k or "intermediate" in k:
                weight.append(5)
            else:
                weight.append(1)
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)
        
        iter = 0
        past_loss = 1e10
        
        while (self.headless and iter < max_iter) or (
            not self.headless and not self.gym.query_viewer_has_closed(self.viewer)
        ):
            iter += 1

            opt_wrist_quat = rot6d_to_quat(opt_wrist_rot)[:, [1, 2, 3, 0]]
            self._root_state[:, 0, :3] = opt_wrist_pos.detach()
            self._root_state[:, 0, 3:7] = opt_wrist_quat.detach()
            self._root_state[:, 0, 7:] = torch.zeros_like(self._root_state[:, 0, 7:])

            opt_dof_pos_clamped = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(opt_dof_pos_clamped))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

            # Physics simulation
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            if not self.headless:
                self.gym.step_graphics(self.sim)

            # Update tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            
            if not self.headless:
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            # Forward kinematics for loss calculation
            ret = self.chain.forward_kinematics(opt_dof_pos_clamped[:, self.isaac2chain_order])
            pk_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)
            pk_joints = (rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)).transpose(-1, -2) + opt_wrist_pos[:, None]

            # Target joints (wrist 중복 제거)
            target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
            
            # MANO 키포인트 시각화 업데이트
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]
                
            loss = torch.mean(torch.norm(pk_joints - target_joints, dim=-1) * weight[None])
            opti.zero_grad()
            loss.backward()
            opti.step()

            if iter % 100 == 0:
                cprint(f"{iter} {loss.item():.6f}", "green")
                if iter > 500:
                    loss_diff = past_loss - loss.item()
                    if loss_diff < 1e-4:
                        cprint(f"Converged at iteration {iter} (loss diff: {loss_diff:.2e})", "cyan")
                        break
                past_loss = loss.item()
            
            if iter % 50 == 0 and loss.item() < 1e-4:
                cprint(f"Early stopping at iteration {iter} (loss: {loss.item():.2e})", "cyan")
                break

        # Isaac Gym simulation에서 실제 joint positions 획득
        isaac_joints = torch.stack(
            [self._rigid_body_state[:, self.dexhand_handles[k], :3] for k in self.dexhand.body_names],
            dim=1,
        )

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
        description="GigaHands MANO to Dexhand (Fixed Version)",
        headless=True,
        custom_parameters=[
            {"name": "--iter", "type": int, "default": 7000},
            {"name": "--data_idx", "type": str, "default": "20aed@0"},
            {"name": "--dexhand", "type": str, "default": "inspire"},
            {"name": "--side", "type": str, "default": "right"},
            {"name": "--data_dir", "type": str, "default": "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands"},
            {"name": "--save_dir", "type": str, "default": "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/retargeted_fixed"}
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    def run_fixed(parser, data_idx):
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
        
        if frame_offset > 0 and frame_offset < motion_data.shape[0]:
            motion_data = motion_data[frame_offset:]
        
        # 수정된 데이터 처리
        demo_data = pack_gigahands_data_fixed(motion_data, dexhand, parser.side)
        
        # GPU로 이동
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for key in demo_data:
            if isinstance(demo_data[key], torch.Tensor):
                demo_data[key] = demo_data[key].to(device)
            elif isinstance(demo_data[key], dict):
                for joint_name in demo_data[key]:
                    demo_data[key][joint_name] = demo_data[key][joint_name].to(device)
        
        parser.num_envs = demo_data["wrist_pos"].shape[0]
        
        cprint(f"Number of environments: {parser.num_envs}", "blue")

        # MANO to Dexhand 변환
        mano2dexhand = Mano2DexhandGigaHandsFixed(parser, dexhand)
        
        # MANO 조인트를 하나의 텐서로 결합 (wrist 제외)
        mano_joints_tensor = torch.cat([
            demo_data["mano_joints"][dexhand.to_hand(j_name)[0]]
            for j_name in dexhand.body_names
            if dexhand.to_hand(j_name)[0] != "wrist"
        ], dim=-1).view(parser.num_envs, -1, 3)

        to_dump = mano2dexhand.fitting(
            parser.iter,
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            mano_joints_tensor,
        )

        # 결과 저장
        parts = seq_id.split("/")
        if len(parts) == 2:
            scene_folder, sequence_id = parts
            dump_dir = os.path.join(parser.save_dir, f"mano2{str(dexhand)}", scene_folder, "keypoints_3d_mano")
        else:
            dump_dir = f"{parser.save_dir}/mano2{str(dexhand)}"
            
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, f"{sequence_id}_retargeted_fixed.pkl")
        
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)
            
        cprint(f"Fixed retargeted data saved to: {dump_path}", "green")

    run_fixed(_parser, _parser.data_idx) 