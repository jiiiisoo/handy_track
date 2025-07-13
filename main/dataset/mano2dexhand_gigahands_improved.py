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

# 기존 함수들 import
from main.dataset.mano2dexhand_gigahands import (
    pack_gigahands_data, 
    load_gigahands_sequence,
    get_all_gigahands_sequences
)

def get_hand_connections():
    """손의 골격 연결 구조 정의 (시각화용)"""
    connections = [
        # 손목에서 각 손가락 기저부로
        (0, 1),   # wrist -> thumb_proximal
        (0, 5),   # wrist -> index_proximal
        (0, 9),   # wrist -> middle_proximal
        (0, 13),  # wrist -> ring_proximal
        (0, 17),  # wrist -> pinky_proximal
        
        # 엄지 연결
        (1, 2), (2, 3), (3, 4),
        # 검지 연결  
        (5, 6), (6, 7), (7, 8),
        # 중지 연결
        (9, 10), (10, 11), (11, 12),
        # 약지 연결
        (13, 14), (14, 15), (15, 16),
        # 새끼 연결
        (17, 18), (18, 19), (19, 20),
    ]
    return connections

class Mano2DexhandGigaHands_Improved:
    def __init__(self, args, dexhand):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand
        self.args = args

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

        self.headless = args.headless
        if self.headless:
            self.graphics_device_id = -1

        assert args.physics_engine == gymapi.SIM_PHYSX

        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8  # 증가
        self.sim_params.physx.num_velocity_iterations = 4  # 증가
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
        asset_options.disable_gravity = True  # 중력 비활성화로 "누워버림" 방지
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        dexhand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.chain = pk.build_chain_from_urdf(open(os.path.join(asset_root, asset_file)).read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        # 더 강한 제어
        dexhand_dof_stiffness = torch.tensor(
            [50] * self.dexhand.n_dofs,  # 강성 증가
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [5] * self.dexhand.n_dofs,   # 댐핑 증가
            dtype=torch.float,
            device=self.sim_device,
        )

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        
        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self.dexhand_dof_effort_limits = []
        self.dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]

            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self.dexhand_dof_effort_limits.append(dexhand_dof_props["effort"][i])
            self.dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self.dexhand_dof_effort_limits = torch.tensor(self.dexhand_dof_effort_limits, device=self.sim_device)
        self.dexhand_dof_speed_limits = torch.tensor(self.dexhand_dof_speed_limits, device=self.sim_device)
        
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        if self.dexhand.name == "inspire":
            default_dof_state["pos"][8] = 0.8
            default_dof_state["pos"][9] = 0.05
        self.dexhand_default_dof_pos = default_dof_state
        
        # 손목을 더 높은 위치에 배치
        self.dexhand_default_pose = gymapi.Transform()
        self.dexhand_default_pose.p = gymapi.Vec3(0, 0, 0.5)  # Z축으로 0.5m 올림
        self.dexhand_default_pose.r = gymapi.Quat(0, 0, 0, 1)

        # 좌표계 변환 (GigaHands -> Isaac Gym)
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.8])  # 높이를 더 올림
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        self.num_envs = args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 2.0  # 간격 증가
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, 2.0)  # 높이 증가

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

            # MANO 키포인트 시각화용 구체들 (더 크게)
            scene_asset_options = gymapi.AssetOptions()
            scene_asset_options.fix_base_link = True
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                joint_name = self.dexhand.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.01, scene_asset_options)  # 크기 증가
                a = self.gym.create_actor(
                    env, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1
                )
                # 손가락별 색상 설정 (더 밝게)
                if "index" in joint_name:
                    inter_c = 100
                elif "middle" in joint_name:
                    inter_c = 160
                elif "ring" in joint_name:
                    inter_c = 220
                elif "pinky" in joint_name:
                    inter_c = 255
                elif "thumb" in joint_name:
                    inter_c = 50
                else:
                    inter_c = 0
                    
                if "tip" in joint_name:
                    c = gymapi.Vec3(1.0, inter_c / 255, inter_c / 255)  # 빨강 계열
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 1.0, inter_c / 255)  # 초록 계열
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, inter_c / 255, 1.0)  # 파랑 계열
                else:
                    c = gymapi.Vec3(0.5, 0.7, 1.0)  # 기본 색상
                self.gym.set_rigid_body_color(env, a, 0, gymapi.MESH_VISUAL, c)

            # 목표 MANO 키포인트 시각화용 (더 큰 구체들)
            for joint_vis_id in range(21):  # MANO 21개 키포인트
                target_point = self.gym.create_sphere(self.sim, 0.015, scene_asset_options)  # 더 큰 구체
                target_actor = self.gym.create_actor(
                    env, target_point, gymapi.Transform(), f"target_joint_{joint_vis_id}", self.num_envs + 2, 0b1
                )
                # 목표점은 반투명 빨간색
                c = gymapi.Vec3(1.0, 0.2, 0.2)
                self.gym.set_rigid_body_color(env, target_actor, 0, gymapi.MESH_VISUAL, c)

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

        # MANO 키포인트 시각화용
        self.mano_joint_points = [
            self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
            for i in range(len(self.dexhand.body_names))
        ]
        
        # 목표 키포인트 시각화용
        self.target_joint_points = [
            self._root_state[:, self.gym.find_actor_handle(env_ptr, f"target_joint_{i}"), :]
            for i in range(21)
        ]

        if not self.headless:
            cam_pos = gymapi.Vec3(2, 2, 1.5)  # 카메라 위치 조정
            cam_target = gymapi.Vec3(0, 0, 0.5)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

    def fitting(self, max_iter, target_wrist_pos, target_wrist_rot, target_mano_joints):
        """MANO 키포인트를 로봇 손으로 retargeting"""
        
        assert target_mano_joints.shape[0] == self.num_envs
        
        # 좌표계 변환
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

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
        
        # 더 합리적인 초기 관절 각도 (완전히 펴진 상태가 아닌 자연스러운 자세)
        initial_dof = self.dexhand_default_dof_pos["pos"].copy()
        # 손가락들을 살짝 구부린 자세로 시작
        if self.dexhand.name == "inspire":
            initial_dof[2:8] = 0.3   # 검지~새끼 첫 번째 관절
            initial_dof[10:16] = 0.2 # 검지~새끼 두 번째 관절
        
        opt_dof_pos = torch.tensor(
            initial_dof[None].repeat(self.num_envs, axis=0),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        
        # 학습률 조정
        opti = torch.optim.Adam(
            [{"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.001}, 
             {"params": [opt_dof_pos], "lr": 0.002}]  # 관절 각도 학습률 증가
        )

        # 손가락별 가중치 설정 (더 균형있게)
        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "tip" in k:
                if "thumb" in k:
                    weight.append(30)  # 엄지 끝
                elif "index" in k:
                    weight.append(25)  # 검지 끝
                else:
                    weight.append(15)  # 다른 손가락 끝
            elif "distal" in k:
                weight.append(8)
            elif "intermediate" in k:
                weight.append(5)
            elif "proximal" in k:
                weight.append(3)
            else:
                weight.append(2)  # 손목
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)
        
        iter = 0
        past_loss = 1e10
        
        # 목표 키포인트 설정 (시각화용)
        target_joints_full = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)  # [num_envs, 21, 3]
        
        cprint(f"🚀 Starting optimization with {max_iter} iterations", "cyan")
        cprint(f"   Target joints shape: {target_joints_full.shape}", "blue")
        
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

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Update tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            
            # Step rendering
            if not self.headless:
                self.gym.step_graphics(self.sim)

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
            
            # 목표 키포인트 시각화 업데이트
            # print(len(self.target_joint_points))
            # 1/0
            for k in range(21):
                if k < len(self.target_joint_points):
                    self.target_joint_points[k][:, :3] = target_joints_full[:, k]
            
            # MANO 키포인트 시각화 업데이트 (현재 로봇 손 위치)
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]
                
            loss = torch.mean(torch.norm(pk_joints - target_joints, dim=-1) * weight[None])
            opti.zero_grad()
            loss.backward()
            opti.step()

            if not self.headless:
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            if iter % 50 == 0:
                avg_distance = torch.mean(torch.norm(pk_joints - target_joints, dim=-1)).item() * 100
                cprint(f"Iter {iter:4d}: Loss={loss.item():.6f}, Avg Distance={avg_distance:.1f}cm", "green")
                
                if iter > 200:  # 최소 200번 실행 후 수렴 체크
                    loss_diff = past_loss - loss.item()
                    if loss_diff < 1e-4:  # 수렴 조건
                        cprint(f"✅ Converged at iteration {iter} (loss diff: {loss_diff:.2e})", "cyan")
                        break
                past_loss = loss.item()
            
            # 매우 작은 loss에서 조기 종료
            if loss.item() < 5e-4:
                cprint(f"🎯 Excellent fit at iteration {iter} (loss: {loss.item():.2e})", "cyan")
                break

        final_distance = torch.mean(torch.norm(pk_joints - target_joints, dim=-1)).item() * 100
        cprint(f"🏁 Final average distance: {final_distance:.1f}cm", "yellow")

        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
        }

        if not self.headless:
            input("Press Enter to close viewer...")
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return to_dump


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="GigaHands MANO to Dexhand (Improved)",
        custom_parameters=[
            {
                "name": "--headless",
                "type": bool,
                "default": False,
            },
            {
                "name": "--iter",
                "type": int,
                "default": 2000,  # 반복 횟수 증가
            },
            {
                "name": "--data_idx",
                "type": str,
                "default": "p001-folder/000@0",
                "help": "Single sequence ID (e.g., 'p001-folder/000@30')"
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
                "help": "Path to GigaHands dataset directory"
            },
            {
                "name": "--save_dir",
                "type": str,
                "default": "/workspace/ManipTrans/analysis_results",
                "help": "Directory to save retargeted data"
            },
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    def run(parser, data_idx):
        # data_idx 파싱
        if "@" in data_idx:
            seq_id, frame_offset = data_idx.split("@")
            frame_offset = int(frame_offset)
        else:
            seq_id = data_idx
            frame_offset = 0
        
        cprint(f"🎯 Loading GigaHands sequence: {seq_id} with offset {frame_offset}", "blue")
        
        # GigaHands 데이터 로드
        motion_data, scene_name, sequence_name = load_gigahands_sequence(
            parser.data_dir, seq_id, parser.side
        )
        
        # 프레임 오프셋 적용
        if frame_offset > 0 and frame_offset < motion_data.shape[0]:
            motion_data = motion_data[frame_offset:frame_offset+1]  # 단일 프레임만
        else:
            motion_data = motion_data[0:1]  # 첫 번째 프레임
        
        cprint(f"✅ Motion data shape: {motion_data.shape}", "green")
        
        # ManipTrans 형식으로 변환
        demo_data = pack_gigahands_data(motion_data, dexhand, parser.side)
        
        # 텐서들을 GPU로 이동
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        demo_data["wrist_pos"] = demo_data["wrist_pos"].to(device)
        demo_data["wrist_rot"] = demo_data["wrist_rot"].to(device)
        for joint_name in demo_data["mano_joints"]:
            demo_data["mano_joints"][joint_name] = demo_data["mano_joints"][joint_name].to(device)
        
        parser.num_envs = 1  # 단일 환경으로 설정
        
        cprint(f"🤖 Creating improved MANO to DexHand optimizer", "blue")

        # 개선된 변환기 생성
        mano2inspire = Mano2DexhandGigaHands_Improved(parser, dexhand)
        
        # MANO 조인트를 하나의 텐서로 결합
        mano_joints_tensor = torch.cat([
            demo_data["mano_joints"][dexhand.to_hand(j_name)[0]]
            for j_name in dexhand.body_names
            if dexhand.to_hand(j_name)[0] != "wrist"
        ], dim=-1).view(parser.num_envs, -1, 3)

        cprint(f"🎯 Starting retargeting optimization...", "cyan")
        to_dump = mano2inspire.fitting(
            parser.iter,
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            mano_joints_tensor,
        )

        # 결과 저장
        os.makedirs(parser.save_dir, exist_ok=True)
        dump_path = os.path.join(parser.save_dir, f"improved_{seq_id.replace('/', '_')}_{frame_offset}_retargeted.pkl")
        
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)
            
        cprint(f"💾 Retargeted data saved to: {dump_path}", "green")

    # 실행
    run(_parser, _parser.data_idx) 