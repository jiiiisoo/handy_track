# Import isaacgym first to avoid PyTorch import order issues
import sys
sys.path.append('/workspace/ManipTrans')

# Import isaacgym modules first
from isaacgym import gymapi, gymtorch, gymutil

# Now safe to import other modules
import numpy as np
import torch
from termcolor import cprint
import os

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def create_isaacgym_inspire_hand():
    """IsaacGym에서 실제 Inspire Hand 환경 생성"""
    
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
    
    # Asset 옵션 설정
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    # asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params = gymapi.VhacdParams()
    asset_options.vhacd_params.resolution = 300000
    
    # URDF 로드
    cprint(f"Loading URDF: {dexhand.urdf_path}", "blue")
    
    if not os.path.exists(dexhand.urdf_path):
        raise FileNotFoundError(f"URDF file not found: {dexhand.urdf_path}")
    
    asset = gym.load_asset(sim, os.path.dirname(dexhand.urdf_path), 
                          os.path.basename(dexhand.urdf_path), asset_options)
    
    if asset is None:
        raise Exception("Failed to load hand asset")
    
    return asset


def get_real_inspire_hand_keypoints(pose="open"):
    """실제 Inspire Hand에서 키포인트들 추출"""
    
    cprint("🚀 Creating real Inspire Hand in IsaacGym...", "cyan")
    
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
        # 완전히 펼친 포즈
        joint_angles = torch.zeros(dof_props.shape[0])
    elif pose == "fist":
        # 주먹 쥔 포즈
        joint_angles = torch.ones(dof_props.shape[0]) * 0.8
    else:  # default
        # 기본 포즈 (약간 구부린 상태)
        joint_angles = torch.zeros(dof_props.shape[0])
        if len(joint_angles) > 10:
            joint_angles[2:8] = 0.3  # 손가락들 약간 구부림
    
    # 7. 관절 각도 적용
    gym.set_actor_dof_position_targets(env, actor_handle, joint_angles.numpy())
    
    # 8. 시뮬레이션 단계 실행 (포즈 안정화)
    for _ in range(100):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    
    # 9. Asset에서 body names 가져오기
    num_bodies = gym.get_asset_rigid_body_count(asset)
    body_names = []
    for i in range(num_bodies):
        body_name = gym.get_asset_rigid_body_name(asset, i)
        body_names.append(body_name)
    
    cprint(f"✅ Got {num_bodies} bodies: {body_names[:5]}{'...' if len(body_names) > 5 else ''}", "green")
    
    # 10. Body states 가져오기
    body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
    
    # Body states 구조 디버깅
    cprint(f"📊 Body states type: {type(body_states)}", "cyan")
    if hasattr(body_states, 'shape'):
        cprint(f"📊 Body states shape: {body_states.shape}", "cyan")
    if hasattr(body_states, 'dtype'):
        cprint(f"📊 Body states dtype: {body_states.dtype}", "cyan")
    
    # 11. 키포인트 추출
    keypoints = {}
    
    for i in range(min(num_bodies, len(body_states))):
        body_name = body_names[i]
        
        # Body states 구조 확인 및 위치 추출
        if hasattr(body_states, 'shape') and len(body_states.shape) > 0:
            # Structured array인 경우
            pos_data = body_states[i] if i < len(body_states) else body_states[0]
            
            # 다양한 접근 방법 시도
            try:
                if hasattr(pos_data, 'pose'):
                    pos = pos_data.pose.p
                elif 'pose' in pos_data.dtype.names:
                    pos = pos_data['pose']['p']
                else:
                    # 직접 위치 데이터 접근
                    pos = pos_data[:3] if len(pos_data) >= 3 else [0, 0, 0]
                
                # Position 데이터 변환
                if hasattr(pos, 'x'):
                    position = np.array([pos.x, pos.y, pos.z])
                elif isinstance(pos, (list, tuple, np.ndarray)):
                    position = np.array(pos[:3])
                else:
                    # 기본값 사용
                    position = np.array([0.0, 0.0, 0.0])
                    
            except Exception as e:
                cprint(f"Warning: Failed to extract position for {body_name}: {e}", "yellow")
                position = np.array([0.0, 0.0, 0.0])
        else:
            # 다른 구조인 경우 기본값 사용
            position = np.array([0.0, 0.0, 0.0])
        
        # Hand joint name 변환
        try:
            hand_joint_name = dexhand.to_hand(body_name)[0]
        except:
            hand_joint_name = body_name
        
        keypoints[hand_joint_name] = position
        
        cprint(f"  {i:2d}. {body_name:20s} -> {hand_joint_name:20s}: [{position[0]:7.3f}, {position[1]:7.3f}, {position[2]:7.3f}]", "white")
    
    # 11. 정리
    gym.destroy_sim(sim)
    
    return keypoints, dexhand


def print_inspire_hand_keypoints(keypoints, dexhand, pose="open"):
    """Inspire Hand 키포인트들을 보기 좋게 출력"""
    
    print(f"\n🤖 INSPIRE HAND - Real {pose.upper()} Pose Keypoints")
    print("="*70)
    
    # 손가락별로 분류
    finger_keypoints = {
        "wrist": [],
        "thumb": [],
        "index": [], 
        "middle": [],
        "ring": [],
        "pinky": []
    }
    
    for joint_name, position in keypoints.items():
        joint_lower = joint_name.lower()
        if "wrist" in joint_lower or "palm" in joint_lower:
            finger_keypoints["wrist"].append((joint_name, position))
        elif "thumb" in joint_lower:
            finger_keypoints["thumb"].append((joint_name, position))
        elif "index" in joint_lower or "ff" in joint_lower:
            finger_keypoints["index"].append((joint_name, position))
        elif "middle" in joint_lower or "mf" in joint_lower:
            finger_keypoints["middle"].append((joint_name, position))
        elif "ring" in joint_lower or "rf" in joint_lower:
            finger_keypoints["ring"].append((joint_name, position))
        elif "pinky" in joint_lower or "lf" in joint_lower:
            finger_keypoints["pinky"].append((joint_name, position))
        else:
            finger_keypoints["wrist"].append((joint_name, position))
    
    colors = {
        "wrist": "white",
        "thumb": "red",
        "index": "green", 
        "middle": "blue",
        "ring": "magenta",
        "pinky": "yellow"
    }
    
    for finger_name, joints in finger_keypoints.items():
        if joints:  # 빈 리스트가 아닌 경우만
            color = colors[finger_name]
            cprint(f"\n📍 {finger_name.upper()}:", color, attrs=['bold'])
            print("-" * 50)
            
            for i, (joint_name, position) in enumerate(joints):
                # 미터를 센티미터로 변환
                pos_cm = position * 100
                print(f"  {i:2d}. {joint_name:25s}: [{position[0]:7.3f}, {position[1]:7.3f}, {position[2]:7.3f}]m = [{pos_cm[0]:6.1f}, {pos_cm[1]:6.1f}, {pos_cm[2]:6.1f}]cm")
    
    # 전체 손 크기 정보
    all_positions = np.array(list(keypoints.values()))
    x_span = np.ptp(all_positions[:, 0])
    y_span = np.ptp(all_positions[:, 1])
    z_span = np.ptp(all_positions[:, 2])
    center = np.mean(all_positions, axis=0)
    
    print(f"\n📏 HAND DIMENSIONS:")
    print("-" * 30)
    print(f"   Length (X): {x_span:.3f}m ({x_span*100:.1f}cm)")
    print(f"   Width  (Y): {y_span:.3f}m ({y_span*100:.1f}cm)")
    print(f"   Height (Z): {z_span:.3f}m ({z_span*100:.1f}cm)")
    print(f"   Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]m")
    
    print(f"\n📊 TOTAL KEYPOINTS: {len(keypoints)}")
    print(f"📊 TOTAL DOFs: {dexhand.n_dofs}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real Inspire Hand Keypoints Extractor")
    parser.add_argument("--pose", type=str, default="open", choices=["open", "fist", "default"], 
                       help="Hand pose")
    
    args = parser.parse_args()
    
    print("🚀 Real Inspire Hand Keypoints Extractor")
    print("="*60)
    
    try:
        # 실제 Inspire Hand 키포인트 추출
        keypoints, dexhand = get_real_inspire_hand_keypoints(args.pose)
        
        # 키포인트 출력
        print_inspire_hand_keypoints(keypoints, dexhand, args.pose)
        
        print("\n🎉 Real keypoints extraction complete!")
        
    except Exception as e:
        cprint(f"❌ Error: {e}", "red")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 