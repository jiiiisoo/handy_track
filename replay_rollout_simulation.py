#!/usr/bin/env python3
"""
HDF5 rollout 데이터를 Isaac Gym 시뮬레이션에서 replay하는 스크립트
저장된 q, base_state, dq 등을 사용해서 실제 시뮬레이션 환경에서 동작을 재현
"""

import h5py
import isaacgym
import numpy as np
import torch
import os
import argparse
import cv2
from time import time, sleep
from pathlib import Path

# Isaac Gym imports
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_conjugate, quat_mul

# ManipTrans imports
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.transform import aa_to_rotmat, rotmat_to_aa, aa_to_quat


class RolloutReplaySimulation:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Isaac Gym 초기화
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self._setup_sim_params()
        
        # DexHand 설정
        self.dexhand = DexHandFactory.create_hand(args.dexhand, args.side)
        
        # 시뮬레이션 생성
        self.sim = self._create_sim()
        self._create_ground_plane()
        self._create_envs()
        
        # 뷰어 생성 (headless가 아닌 경우)
        if not args.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self._setup_camera()
        
        # 시뮬레이션 준비
        self.gym.prepare_sim(self.sim)
        
        # 텐서 초기화
        self._init_tensors()
        
    def _setup_sim_params(self):
        """시뮬레이션 파라미터 설정"""
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim_params.dt = 1.0 / 60.0  # 60 FPS
        
        # PhysX 설정
        self.sim_params.substeps = 2
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.use_gpu = True
        
        self.sim_params.use_gpu_pipeline = True
        
    def _create_sim(self):
        """시뮬레이션 생성"""
        compute_device_id = 0
        graphics_device_id = 0 if not self.args.headless else -1
        
        return self.gym.create_sim(
            compute_device_id, graphics_device_id, 
            gymapi.SIM_PHYSX, self.sim_params
        )
    
    def _create_ground_plane(self):
        """바닥면 생성"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self):
        """환경 생성"""
        self.num_envs = self.args.num_envs
        
        # 환경 설정
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # 테이블 생성
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 1.0, 1.6, 0.03, table_asset_options)
        
        # DexHand 에셋 로드
        dexhand_asset_file = self.dexhand.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        
        dexhand_asset = self.gym.load_asset(
            self.sim, *os.path.split(dexhand_asset_file), asset_options
        )
        
        # DOF 속성 설정
        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.num_dof = self.gym.get_asset_dof_count(dexhand_asset)
        
        # 스티프니스와 댐핑 설정
        for i in range(self.num_dof):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = 1000.0
            dexhand_dof_props["damping"][i] = 100.0
        
        # 환경들 생성
        self.envs = []
        self.dexhand_actors = []
        
        num_per_row = int(np.sqrt(self.num_envs))
        
        for i in range(self.num_envs):
            # 환경 생성
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            
            # 테이블 추가
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, 0.4)
            table_actor = self.gym.create_actor(env, table_asset, table_pose, "table", i, 0)
            
            # DexHand 추가
            dexhand_pose = gymapi.Transform()
            dexhand_pose.p = gymapi.Vec3(0.0, 0.0, 0.6)  # 테이블 위
            dexhand_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi/2, 0)
            
            dexhand_actor = self.gym.create_actor(
                env, dexhand_asset, dexhand_pose, "dexhand", i, 0
            )
            self.dexhand_actors.append(dexhand_actor)
            
            # DOF 속성 적용
            self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)
        
        # DOF 핸들 저장
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(self.envs[0], 0, k) 
            for k in self.dexhand.body_names
        }
    
    def _setup_camera(self):
        """카메라 설정"""
        if not self.args.headless:
            # 사용자 정의 카메라 거리와 각도 사용
            distance = self.args.camera_distance
            angle_rad = np.radians(self.args.camera_angle)
            
            # 카메라 위치 계산 (손 앞쪽에서 더 가까이)
            cam_pos = gymapi.Vec3(
                distance * np.cos(angle_rad),  # x: 거리 * cos(각도)
                distance * np.sin(angle_rad) * 0.5,  # y: 약간 옆에서
                0.8 + distance * 0.3  # z: 약간 위에서
            )
            cam_target = gymapi.Vec3(0.0, 0.0, 0.6)  # 손목 높이로 타겟 조정
            middle_env = self.envs[self.num_envs // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        
        # 영상 저장용 카메라 설정
        if self.args.save_video:
            self._setup_recording_camera()
    
    def _setup_recording_camera(self):
        """영상 저장용 카메라 설정"""
        # 카메라 속성 설정
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1920
        camera_props.height = 1080
        camera_props.horizontal_fov = 75
        camera_props.enable_tensors = True
        
        # 각 환경에 카메라 생성
        self.cameras = []
        for i, env in enumerate(self.envs):
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            
            # 카메라 위치 설정 (손을 잘 보이도록)
            cam_pos = gymapi.Vec3(1.0, 0.5, 0.9)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.6)
            self.gym.set_camera_location(camera_handle, env, cam_pos, cam_target)
            
            self.cameras.append(camera_handle)
        
        # 영상 저장용 설정
        import cv2
        self.video_writer = None
        self.frame_count = 0
    
    def _init_tensors(self):
        """텐서 초기화"""
        # Actor root state
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        
        # DOF state  
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        
        # Rigid body state
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        
        # Position control
        self._pos_control = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        
        # Global indices for updating
        self._global_indices = torch.tensor(
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
            
            # 추가 데이터가 있다면 로드
            if 'actions' in rollout:
                self.rollout_data['actions'] = torch.tensor(rollout['actions'][:], dtype=torch.float32, device=self.device)
            
            print(f"📊 Loaded data:")
            print(f"  - Timesteps: {len(self.rollout_data['q'])}")
            print(f"  - Joint positions: {self.rollout_data['q'].shape}")
            print(f"  - Joint velocities: {self.rollout_data['dq'].shape}")
            print(f"  - Base state: {self.rollout_data['base_state'].shape}")
            
            return rollout_name
    
    def replay_rollout(self, rollout_name, playback_speed=1.0):
        """롤아웃 데이터를 시뮬레이션에서 replay"""
        print(f"🎬 Starting rollout replay: {rollout_name}")
        print(f"⏩ Playback speed: {playback_speed}x")
        
        timesteps = len(self.rollout_data['q'])
        target_dt = self.sim_params.dt / playback_speed
        
        # 영상 저장 초기화
        if self.args.save_video and hasattr(self, 'cameras'):
            output_path = f"rollout_{rollout_name}_{int(time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(1.0 / target_dt)
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
            print(f"📹 Recording video to: {output_path}")
        
        for step in range(timesteps):
            step_start_time = time()
            
            # 현재 스텝의 데이터 가져오기
            current_q = self.rollout_data['q'][step]       # [num_dof]
            current_dq = self.rollout_data['dq'][step]     # [num_dof]  
            current_base = self.rollout_data['base_state'][step]  # [13]
            
            # 모든 환경에 동일한 상태 적용 (멀티 환경 지원)
            for env_idx in range(self.num_envs):
                # DOF 상태 설정
                self._dof_state[env_idx, :, 0] = current_q  # 위치
                self._dof_state[env_idx, :, 1] = current_dq  # 속도
                
                # Base state 설정 (dexhand actor는 인덱스 1, 테이블이 0)
                self._root_state[env_idx, 1, :] = current_base
                
                # Position control target 설정
                self._pos_control[env_idx, :] = current_q
            
            # 상태를 시뮬레이션에 적용
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
            
            # 시뮬레이션 스텝
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 뷰어 업데이트
            if not self.args.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                
                # 종료 체크
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
            
            # 영상 저장
            if self.args.save_video and hasattr(self, 'cameras') and self.video_writer is not None:
                self._capture_frame()
            
            # 프레임레이트 제어
            step_time = time() - step_start_time
            sleep_time = max(0, target_dt - step_time)
            if sleep_time > 0:
                sleep(sleep_time)
            
            # 진행상황 출력
            if step % 30 == 0 or step == timesteps - 1:
                print(f"⏱️  Step {step+1}/{timesteps} ({(step+1)/timesteps*100:.1f}%)")
        
        # 영상 저장 완료
        if self.args.save_video and self.video_writer is not None:
            self.video_writer.release()
            print(f"📹 Video saved successfully!")
        
        print(f"✅ Replay completed!")
    
    def _capture_frame(self):
        """현재 프레임 캡처하여 영상에 저장"""
        try:
            # 첫 번째 환경의 카메라에서 이미지 캡처
            self.gym.render_all_camera_sensors(self.sim)
            camera_image = self.gym.get_camera_image(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
            
            # 이미지 형태 변환 (RGBA -> BGR)
            camera_image = camera_image.reshape(1080, 1920, 4)
            camera_image_bgr = cv2.cvtColor(camera_image, cv2.COLOR_RGBA2BGR)
            
            # 영상에 프레임 추가
            self.video_writer.write(camera_image_bgr)
            self.frame_count += 1
            
        except Exception as e:
            print(f"⚠️  Frame capture error: {e}")
    
    def cleanup(self):
        """정리"""
        # 영상 저장 정리
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            print(f"📹 Video writer released (captured {self.frame_count} frames)")
        
        # 뷰어와 시뮬레이션 정리
        if not self.args.headless and hasattr(self, 'viewer'):
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def main():
    parser = argparse.ArgumentParser(description='Replay rollout data in Isaac Gym simulation')
    
    # 필수 인자
    parser.add_argument('--hdf5_path', type=str, required=True,
                       help='Path to HDF5 rollout file')
    
    # 시뮬레이션 설정
    parser.add_argument('--dexhand', type=str, default='inspire',
                       help='DexHand type (inspire, etc.)')
    parser.add_argument('--side', type=str, default='right',
                       help='Hand side (right/left)')
    parser.add_argument('--num_envs', type=int, default=1,
                       help='Number of environments')
    parser.add_argument('--headless', action='store_true',
                       help='Run without viewer')
    
    # 롤아웃 설정
    parser.add_argument('--rollout_name', type=str, default=None,
                       help='Specific rollout to replay')
    parser.add_argument('--playback_speed', type=float, default=1.0,
                       help='Playback speed multiplier')
    parser.add_argument('--loop', action='store_true',
                       help='Loop the replay continuously')
    parser.add_argument('--save_video', action='store_true',
                       help='Save replay as video file')
    parser.add_argument('--camera_distance', type=float, default=1.2,
                       help='Distance of camera from hand')
    parser.add_argument('--camera_angle', type=float, default=45,
                       help='Camera viewing angle in degrees')
    
    args = parser.parse_args()
    
    # 파일 확인
    if not os.path.exists(args.hdf5_path):
        print(f"❌ HDF5 file not found: {args.hdf5_path}")
        return
    
    print(f"🚀 Isaac Gym Rollout Replay")
    print(f"📁 File: {args.hdf5_path}")
    print(f"🤖 DexHand: {args.dexhand} ({args.side})")
    print(f"🌍 Environments: {args.num_envs}")
    print(f"📷 Camera: distance={args.camera_distance:.1f}m, angle={args.camera_angle}°")
    if args.save_video:
        print(f"📹 Video recording: ENABLED (1920x1080)")
    if args.loop:
        print(f"🔄 Loop mode: ENABLED")
    
    try:
        # 시뮬레이션 초기화
        sim = RolloutReplaySimulation(args)
        
        # 롤아웃 데이터 로드
        rollout_name = sim.load_rollout_data(args.hdf5_path, args.rollout_name)
        
        # Replay 실행
        if args.loop:
            print(f"🔄 Looping enabled. Press Ctrl+C to stop.")
            try:
                while True:
                    sim.replay_rollout(rollout_name, args.playback_speed)
                    if not args.headless:
                        if sim.gym.query_viewer_has_closed(sim.viewer):
                            break
            except KeyboardInterrupt:
                print(f"\n🛑 Stopped by user")
        else:
            sim.replay_rollout(rollout_name, args.playback_speed)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'sim' in locals():
            sim.cleanup()
        print(f"🏁 Cleanup completed")


if __name__ == "__main__":
    main() 