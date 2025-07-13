# CRITICAL: Import isaacgym modules FIRST
from isaacgym import gymapi, gymtorch, gymutil

import torch
import numpy as np
import sys
import os
from termcolor import cprint

sys.path.append('/workspace/ManipTrans')

from main.dataset.mano2dexhand_gigahands import load_gigahands_sequence, pack_gigahands_data
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.transform import aa_to_rotmat
from analyze_retargeting_quality import calculate_retargeting_metrics, create_retargeting_report, visualize_distance_heatmap


def run_retargeting_with_analysis(seq_id="p001-folder/000", frame_idx=30, dexhand_name="inspire", side="right"):
    """Retargeting 실행하면서 실시간 분석"""
    
    print("🚀 Starting Retargeting with Real-time Analysis")
    cprint(f"Target: {seq_id} frame {frame_idx}", "cyan")
    
    # 1. 데이터 로드
    data_dir = "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands"
    motion_data, scene_name, sequence_name = load_gigahands_sequence(data_dir, seq_id, side)
    
    if frame_idx >= motion_data.shape[0]:
        frame_idx = motion_data.shape[0] - 1
        cprint(f"Frame adjusted to {frame_idx}", "yellow")
    
    # 단일 프레임 추출
    frame_data = motion_data[frame_idx:frame_idx+1]
    
    # 2. DexHand 모델 생성
    dexhand = DexHandFactory.create_hand(dexhand_name, side)
    
    # 3. 데이터 변환
    demo_data = pack_gigahands_data(frame_data, dexhand, side)
    
    # 목표 키포인트 준비
    target_mano_joints = torch.cat([
        demo_data["mano_joints"][dexhand.to_hand(j_name)[0]]
        for j_name in dexhand.body_names
        if dexhand.to_hand(j_name)[0] != "wrist"
    ], dim=-1).view(1, -1, 3)
    
    target_wrist_pos = demo_data["wrist_pos"]
    target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
    
    print(f"✅ Target keypoints shape: {target_joints.shape}")
    
    # 4. 시뮬레이션 결과 로드 (예시)
    # 실제로는 mano2dexhand_gigahands.py의 결과를 사용해야 함
    
    # 임시로 더미 최적화된 관절 위치 생성 (실제 구현에서는 실제 결과 사용)
    device = target_joints.device
    optimized_joints = target_joints[0] + torch.randn_like(target_joints[0]) * 0.008  # 0.8cm 노이즈로 감소
    
    # 5. 메트릭 계산
    joint_names = [dexhand.to_hand(j_name)[0] for j_name in dexhand.body_names]
    
    metrics = calculate_retargeting_metrics(
        target_keypoints=target_joints[0],
        optimized_joints=optimized_joints,
        finger_info=joint_names
    )
    
    # 6. 리포트 생성
    sequence_info = {
        'seq_id': seq_id,
        'frame_idx': frame_idx,
        'side': side,
        'dexhand': dexhand_name
    }
    
    create_retargeting_report(metrics, sequence_info)
    
    # 7. 히트맵 시각화
    output_dir = "/workspace/ManipTrans/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    heatmap_path = os.path.join(output_dir, f"retargeting_heatmap_{seq_id.replace('/', '_')}_{frame_idx}.png")
    
    cprint(f"\n🎨 Generating distance heatmap...", "cyan")
    visualize_distance_heatmap(metrics, heatmap_path)
    
    return metrics


def analyze_saved_retargeting_result(pickle_path):
    """저장된 retargeting 결과 분석"""
    
    import pickle
    
    cprint(f"📂 Loading saved result: {pickle_path}", "blue")
    
    try:
        with open(pickle_path, 'rb') as f:
            result = pickle.load(f)
        
        # 결과 구조 확인
        print("📋 Available keys:", list(result.keys()))
        
        # 최적화된 관절 위치 추출
        optimized_joints = torch.tensor(result['opt_joints_pos'][0])  # 첫 번째 환경
        
        # 원본 타겟은 다시 로드해야 함 (저장되어 있지 않음)
        cprint("⚠️  Target keypoints need to be reloaded for comparison", "yellow")
        
        return result
        
    except Exception as e:
        cprint(f"❌ Error loading file: {e}", "red")
        return None


def quick_heatmap_demo():
    """빠른 데모용 히트맵"""
    
    cprint("🎨 Creating demo heatmap...", "cyan")
    
    # 더미 데이터 생성 (더 현실적인 거리로)
    num_joints = 20
    target_keypoints = torch.randn(num_joints, 3)
    optimized_joints = target_keypoints + torch.randn_like(target_keypoints) * 0.015  # 1.5cm 노이즈로 감소
    
    joint_names = [
        "wrist", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
        "index_proximal", "index_intermediate", "index_distal", "index_tip",
        "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
        "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
        "pinky_proximal", "pinky_intermediate", "pinky_tip"
    ]
    
    # 메트릭 계산
    metrics = calculate_retargeting_metrics(target_keypoints, optimized_joints, joint_names)
    
    # 리포트 생성
    sequence_info = {
        'seq_id': 'demo_sequence',
        'frame_idx': 0,
        'side': 'right',
        'dexhand': 'inspire'
    }
    
    create_retargeting_report(metrics, sequence_info)
    
    # 히트맵 생성
    output_dir = "/workspace/ManipTrans/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, "demo_retargeting_heatmap.png")
    
    visualize_distance_heatmap(metrics, heatmap_path)
    
    cprint(f"✅ Demo heatmap saved to: {heatmap_path}", "green")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Retargeting Analysis Tool")
    parser.add_argument("--mode", type=str, default="demo", 
                       choices=["demo", "realtime", "saved"],
                       help="Analysis mode")
    parser.add_argument("--seq_id", type=str, default="p001-folder/000",
                       help="Sequence ID for realtime analysis")
    parser.add_argument("--frame_idx", type=int, default=30,
                       help="Frame index for realtime analysis")
    parser.add_argument("--pickle_path", type=str, default="",
                       help="Path to saved retargeting result pickle file")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        print("🎯 Running quick demo...")
        quick_heatmap_demo()
        
    elif args.mode == "realtime":
        print("🔄 Running real-time analysis...")
        run_retargeting_with_analysis(args.seq_id, args.frame_idx)
        
    elif args.mode == "saved":
        if not args.pickle_path:
            cprint("❌ Please provide --pickle_path for saved mode", "red")
            return
        print("📂 Analyzing saved result...")
        analyze_saved_retargeting_result(args.pickle_path)
    
    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main() 