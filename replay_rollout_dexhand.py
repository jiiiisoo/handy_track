#!/usr/bin/env python3
"""
DexHandImitator 환경을 기반으로 한 정확한 롤아웃 replay 스크립트
원본 훈련 환경과 동일한 설정을 사용하여 최대한 정확한 재현
"""

import h5py
import numpy as np
import torch
import os
import argparse
from time import time, sleep
from pathlib import Path

# Isaac Gym imports
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul

# ManipTrans imports
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.transform import aa_to_rotmat, rotmat_to_aa, aa_to_quat
from maniptrans_envs.lib.envs.core.sim_config import sim_config


class DexHandReplayEnv:
    """DexHandImitator와 동일한 설정을 사용하는 replay 환경"""
    
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # DexHand 설정
        self.dexhand = DexHandFactory.create_hand(args.dexhand, args.side)
        
        # 시뮬레이션 설정 (DexHandImitator와 동일)
        self.cfg = {
            "physics_engine": gymapi.SIM_PHYSX,
            "num_threads": 0,
            "solver_type": 1,
            "num_subscenes": 0,
            "use_gpu": True,
            "use_gpu_pipeline": True,
        }
        
        # Isaac Gym 초기화
        self.gym = gymapi.acquire_gym()
        self._create_sim()
        self._create_ground_plane()
        self._create_envs()
        
        # 뷰어 생성
        if not args.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self._setup_camera()
        
        # 시뮬레이션 준비
        self.gym.prepare_sim(self.sim)
        self._init_tensors()
        
    def _create_sim(self):
        """DexHandImitator와 동일한 시뮬레이션 설정"""
        # 시뮬레이션 파라미터
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim_params.dt = 1.0/60.0
        
        # PhysX 설정
        self.sim_params.substeps = 2
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = 0
        self.sim_params.physx.use_gpu = True
        
        self.sim_params.use_gpu_pipeline = True
        
        # 시뮬레이션 생성
        device_id = 0
        graphics_device_id = 0 if not self.args.headless else -1
        
        self.sim = self.gym.create_sim(
            device_id, graphics_device_id,
            gymapi.SIM_PHYSX, self.sim_params
        )
        
    def _create_ground_plane(self):
        """바닥면 생성"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_envs(self):
        """DexHandImitator와 동일한 환경 생성"""
        self.num_envs = self.args.num_envs
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # 테이블 에셋 (DexHandImitator와 동일)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)
        
        # 테이블 위치 설정
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        table_surface_z = table_pos.z + 0.015  # table_half_height
        table_half_width = 0.4
        
        # DexHand 포즈 설정 (DexHandImitator와 동일)
        self.dexhand_pose = gymapi.Transform()
        ROBOT_HEIGHT = 0.09  # config에서 가져온 값
        self.dexhand_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)
        
        # DexHand 에셋 로드
        dexhand_asset_file = self.dexhand.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        
        dexhand_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_asset_file), asset_options)
        
        # DOF 설정 (DexHandImitator와 동일)
        dexhand_dof_stiffness = torch.tensor([500] * self.dexhand.n_dofs, dtype=torch.float, device=self.device)
        dexhand_dof_damping = torch.tensor([30] * self.dexhand.n_dofs, dtype=torch.float, device=self.device)
        
        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.num_dof = self.gym.get_asset_dof_count(dexhand_asset)
        
        for i in range(self.num_dof):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]
        
        # Friction 설정
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_props_asset)
        
        # 환경들 생성
        self.envs = []
        self.dexhands = []
        
        num_per_row = int(np.sqrt(self.num_envs))
        
        for i in range(self.num_envs):
            # 환경 생성
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)
            
            # DexHand 생성 (반드시 첫 번째로!)
            dexhand_actor = self.gym.create_actor(
                env_ptr, dexhand_asset, self.dexhand_pose, "dexhand", i,
                (1 if self.dexhand.self_collision else 0)
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_actor, dexhand_dof_props)
            self.dexhands.append(dexhand_actor)
            
            # 테이블 생성
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i + self.num_envs, 0b11)
            
            # 테이블 색상 설정
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))
        
        # 핸들 저장
        env_ptr = self.envs[0]
        dexhand_handle = self.gym.find_actor_handle(env_ptr, "dexhand")
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) 
            for k in self.dexhand.body_names
        }
        
    def _setup_camera(self):
        """카메라 설정"""
        if not self.args.headless:
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            middle_env = self.envs[self.num_envs // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
            
    def _init_tensors(self):
        """텐서 초기화"""
        # Actor root state
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        
        # DOF state
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        
        # Rigid body state
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        
        # Base state (dexhand root)
        self._base_state = self._root_state[:, 0, :]  # dexhand는 첫 번째 actor
        
        # Position control
        self._pos_control = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        
        # Global indices
        self._global_dexhand_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        
    def load_rollout_data(self, hdf5_path, rollout_name="rollout_0"):
        """HDF5에서 rollout 데이터 로드"""
        print(f"🔍 Loading rollout: {rollout_name} from {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            if 'rollouts/successful' not in f:
                raise ValueError("No successful rollouts found!")
            
            if rollout_name not in f['rollouts/successful']:
                available = list(f['rollouts/successful'].keys())
                print(f"Available rollouts: {available}")
                rollout_name = available[0]
                print(f"Using: {rollout_name}")
            
            rollout = f[f'rollouts/successful/{rollout_name}']
            
            # 데이터 로드
            self.rollout_data = {
                'q': torch.tensor(rollout['q'][:], dtype=torch.float32, device=self.device),
                'dq': torch.tensor(rollout['dq'][:], dtype=torch.float32, device=self.device),
                'base_state': torch.tensor(rollout['base_state'][:], dtype=torch.float32, device=self.device),
            }
            
            # 추가 데이터 로드
            for key in ['actions', 'rewards', 'dones']:
                if key in rollout:
                    self.rollout_data[key] = torch.tensor(rollout[key][:], dtype=torch.float32, device=self.device)
            
            print(f"📊 Loaded data:")
            print(f"  - Timesteps: {len(self.rollout_data['q'])}")
            print(f"  - Joint positions: {self.rollout_data['q'].shape}")
            print(f"  - Joint velocities: {self.rollout_data['dq'].shape}")
            print(f"  - Base state: {self.rollout_data['base_state'].shape}")
            
            return rollout_name
            
    def replay_rollout(self, rollout_name, playback_speed=1.0):
        """롤아웃 replay 실행"""
        print(f"🎬 Starting rollout replay: {rollout_name}")
        print(f"⏩ Playback speed: {playback_speed}x")
        
        timesteps = len(self.rollout_data['q'])
        target_dt = self.sim_params.dt / playback_speed
        
        for step in range(timesteps):
            step_start_time = time()
            
            # 현재 스텝 데이터
            current_q = self.rollout_data['q'][step]
            current_dq = self.rollout_data['dq'][step]
            current_base = self.rollout_data['base_state'][step]
            
            # 모든 환경에 상태 적용
            self._q[:] = current_q[None].repeat(self.num_envs, 1)
            self._qd[:] = current_dq[None].repeat(self.num_envs, 1)
            self._base_state[:] = current_base[None].repeat(self.num_envs, 1)
            self._pos_control[:] = current_q[None].repeat(self.num_envs, 1)
            
            # 상태 업데이트
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(self._global_dexhand_indices.flatten()),
                len(self._global_dexhand_indices.flatten())
            )
            
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(self._global_dexhand_indices.flatten()),
                len(self._global_dexhand_indices.flatten())
            )
            
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._pos_control),
                gymtorch.unwrap_tensor(self._global_dexhand_indices.flatten()),
                len(self._global_dexhand_indices.flatten())
            )
            
            # 시뮬레이션 스텝
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 뷰어 업데이트
            if not self.args.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
            
            # 프레임레이트 제어
            step_time = time() - step_start_time
            sleep_time = max(0, target_dt - step_time)
            if sleep_time > 0:
                sleep(sleep_time)
            
            # 진행상황 출력
            if step % 60 == 0 or step == timesteps - 1:
                progress = (step + 1) / timesteps * 100
                print(f"⏱️  Step {step+1}/{timesteps} ({progress:.1f}%)")
                
                # 리워드 정보 출력 (있는 경우)
                if 'rewards' in self.rollout_data:
                    reward = self.rollout_data['rewards'][step].item()
                    print(f"    Reward: {reward:.3f}")
        
        print(f"✅ Replay completed!")
        
    def cleanup(self):
        """정리"""
        if not self.args.headless and hasattr(self, 'viewer'):
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def main():
    parser = argparse.ArgumentParser(description='Replay rollout in DexHandImitator-like environment')
    
    # 필수 인자
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='Path to HDF5 rollout file')
    
    # DexHand 설정
    parser.add_argument('--dexhand', type=str, default='inspire',
                       help='DexHand type')
    parser.add_argument('--side', type=str, default='right',
                       help='Hand side')
    parser.add_argument('--num_envs', type=int, default=1,
                       help='Number of environments')
    parser.add_argument('--headless', action='store_true',
                       help='Run without viewer')
    
    # Replay 설정
    parser.add_argument('--rollout_name', type=str, default=None,
                       help='Specific rollout to replay')
    parser.add_argument('--playback_speed', type=float, default=1.0,
                       help='Playback speed multiplier')
    parser.add_argument('--loop', action='store_true',
                       help='Loop continuously')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_path):
        print(f"❌ HDF5 file not found: {args.hdf5_path}")
        return
    
    print(f"🚀 DexHand Rollout Replay")
    print(f"📁 File: {args.hdf5_path}")
    print(f"🤖 DexHand: {args.dexhand} ({args.side})")
    
    try:
        # 환경 초기화
        env = DexHandReplayEnv(args)
        
        # 데이터 로드
        rollout_name = env.load_rollout_data(args.hdf5_path, args.rollout_name)
        
        # Replay 실행
        if args.loop:
            print(f"🔄 Looping enabled. Press Ctrl+C or close viewer to stop.")
            try:
                while True:
                    env.replay_rollout(rollout_name, args.playback_speed)
                    if not args.headless:
                        if env.gym.query_viewer_has_closed(env.viewer):
                            break
            except KeyboardInterrupt:
                print(f"\n🛑 Stopped by user")
        else:
            env.replay_rollout(rollout_name, args.playback_speed)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'env' in locals():
            env.cleanup()
        print(f"🏁 Cleanup completed")


if __name__ == "__main__":
    main() 