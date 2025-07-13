# CRITICAL: Import isaacgym modules FIRST
from isaacgym import gymapi, gymtorch, gymutil

import torch
import numpy as np
import pickle
import sys
import os
from termcolor import cprint

sys.path.append('/workspace/ManipTrans')

from main.dataset.mano2dexhand_gigahands import load_gigahands_sequence, pack_gigahands_data
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.transform import aa_to_rotmat
from analyze_retargeting_quality import calculate_retargeting_metrics, create_retargeting_report, visualize_distance_heatmap


def analyze_real_retargeting_result(pickle_path, seq_id="p001-folder/000", frame_idx=0):
    """실제 retargeting 결과 분석"""
    
    cprint(f"🔍 Analyzing Real Retargeting Result", "cyan")
    cprint(f"📂 File: {pickle_path}", "blue")
    cprint(f"🎯 Sequence: {seq_id}, Frame: {frame_idx}", "blue")
    
    # 1. Pickle 파일 로드
    try:
        with open(pickle_path, 'rb') as f:
            result = pickle.load(f)
        
        cprint("✅ Pickle file loaded successfully", "green")
        print("📋 Available keys:", list(result.keys()))
        
        # 결과 구조 확인
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
        
    except Exception as e:
        cprint(f"❌ Error loading pickle: {e}", "red")
        return
    
    # 2. 원본 GigaHands 데이터 로드 (target keypoints 계산용)
    try:
        data_dir = "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands"
        motion_data, scene_name, sequence_name = load_gigahands_sequence(data_dir, seq_id, "right")
        
        # 시퀀스 파일명에서 frame_idx 추출 시도
        if seq_id.endswith('/000'):
            actual_frame_idx = 0
        elif seq_id.endswith('/001'):
            actual_frame_idx = 1
        else:
            actual_frame_idx = frame_idx
            
        if actual_frame_idx >= motion_data.shape[0]:
            actual_frame_idx = motion_data.shape[0] - 1
            
        cprint(f"📊 Using frame index: {actual_frame_idx}", "yellow")
        
        # 단일 프레임 추출
        frame_data = motion_data[actual_frame_idx:actual_frame_idx+1]
        
        # DexHand 모델 생성
        dexhand = DexHandFactory.create_hand("inspire", "right")
        
        # 데이터 변환
        demo_data = pack_gigahands_data(frame_data, dexhand, "right")
        
        # 목표 키포인트 준비
        target_mano_joints = torch.cat([
            demo_data["mano_joints"][dexhand.to_hand(j_name)[0]]
            for j_name in dexhand.body_names
            if dexhand.to_hand(j_name)[0] != "wrist"
        ], dim=-1).view(1, -1, 3)
        
        target_wrist_pos = demo_data["wrist_pos"]
        target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
        
        cprint(f"✅ Target keypoints loaded: {target_joints.shape}", "green")
        
    except Exception as e:
        cprint(f"❌ Error loading GigaHands data: {e}", "red")
        return
    
    # 3. 좌표계 변환 적용 (retargeting에서 사용된 것과 동일)
    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.5])
    mujoco2gym_transf = torch.tensor(mujoco2gym_transf, dtype=torch.float32)
    
    # 목표 좌표 변환
    target_wrist_pos = demo_data["wrist_pos"]
    target_wrist_rot = demo_data["wrist_rot"]
    target_mano_joints = target_mano_joints
    
    # Transform coordinates
    target_wrist_pos = (mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + mujoco2gym_transf[:3, 3]
    target_wrist_rot = mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
    target_mano_joints = target_mano_joints.view(-1, 3)
    target_mano_joints = (mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + mujoco2gym_transf[:3, 3]
    target_mano_joints = target_mano_joints.view(1, -1, 3)
    
    # 최종 target joints
    transformed_target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
    
    # 4. 최적화된 관절 위치 추출
    try:
        # 결과에서 최적화된 관절 위치 가져오기
        if 'opt_joints_pos' in result:
            optimized_joints = torch.tensor(result['opt_joints_pos'][0], dtype=torch.float32)  # 첫 번째 환경
            cprint(f"✅ Optimized joints loaded: {optimized_joints.shape}", "green")
        else:
            cprint("❌ opt_joints_pos not found in result", "red")
            return
            
    except Exception as e:
        cprint(f"❌ Error extracting optimized joints: {e}", "red")
        return
    
    # 5. 메트릭 계산
    joint_names = [dexhand.to_hand(j_name)[0] for j_name in dexhand.body_names]
    
    metrics = calculate_retargeting_metrics(
        target_keypoints=transformed_target_joints[0],
        optimized_joints=optimized_joints,
        finger_info=joint_names
    )
    
    # 6. 리포트 생성
    sequence_info = {
        'seq_id': seq_id,
        'frame_idx': actual_frame_idx,
        'side': 'right',
        'dexhand': 'inspire'
    }
    
    create_retargeting_report(metrics, sequence_info)
    
    # 7. 히트맵 시각화
    output_dir = "/workspace/ManipTrans/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    file_basename = os.path.basename(pickle_path).replace('.pkl', '')
    heatmap_path = os.path.join(output_dir, f"real_retargeting_heatmap_{file_basename}.png")
    
    cprint(f"\n🎨 Generating distance heatmap...", "cyan")
    visualize_distance_heatmap(metrics, heatmap_path)
    
    # 8. 추가 정보 출력
    print(f"\n📈 DETAILED ANALYSIS:")
    print(f"   Original loss achieved: {result.get('final_loss', 'N/A')}")
    print(f"   Optimization iterations: {result.get('iterations', 'N/A')}")
    
    # DOF 정보 출력
    if 'opt_dof_pos' in result:
        dof_pos = result['opt_dof_pos'][0]
        print(f"   DOF range: [{np.min(dof_pos):.3f}, {np.max(dof_pos):.3f}]")
        print(f"   DOF std: {np.std(dof_pos):.3f}")
    
    return metrics, result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real Retargeting Result Analyzer")
    parser.add_argument("--pickle_path", type=str, 
                       default="/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/retargeted/mano2inspire_rh/p001-folder/keypoints_3d_mano/000_retargeted.pkl",
                       help="Path to retargeted pickle file")
    parser.add_argument("--seq_id", type=str, default="p001-folder/000",
                       help="Sequence ID")
    parser.add_argument("--frame_idx", type=int, default=0,
                       help="Frame index")
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.pickle_path):
        cprint(f"❌ File not found: {args.pickle_path}", "red")
        cprint("📂 Available files:", "yellow")
        
        base_dir = "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/retargeted/mano2inspire_rh/p001-folder/keypoints_3d_mano/"
        if os.path.exists(base_dir):
            files = [f for f in os.listdir(base_dir) if f.endswith('.pkl')][:5]
            for f in files:
                print(f"   {os.path.join(base_dir, f)}")
        return
    
    # 분석 실행
    print("🚀 Real Retargeting Analysis")
    analyze_real_retargeting_result(args.pickle_path, args.seq_id, args.frame_idx)
    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main() 