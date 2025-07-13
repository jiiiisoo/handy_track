# CRITICAL: Import isaacgym modules FIRST
from isaacgym import gymapi, gymtorch, gymutil

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from termcolor import cprint

sys.path.append('/workspace/ManipTrans')

from main.dataset.mano2dexhand_gigahands import load_gigahands_sequence, pack_gigahands_data
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.transform import aa_to_rotmat


def analyze_retargeting_quality():
    """리타겟팅 품질 분석"""
    
    print("\n" + "="*60)
    print("🔍 RETARGETING QUALITY ANALYSIS")
    print("="*60)
    
    # 1. 거리 기반 성공도 측정
    print("\n📏 DISTANCE-BASED SUCCESS METRICS:")
    print("   ✅ EXCELLENT: < 2cm average distance")
    print("   🟡 GOOD:      < 5cm average distance") 
    print("   🟠 FAIR:      < 10cm average distance")
    print("   ❌ POOR:      > 10cm average distance")
    
    # 2. 손가락별 세부 분석
    print("\n👆 FINGER-WISE ANALYSIS:")
    finger_weights = {
        "thumb": 1.5,    # 엄지가 가장 중요
        "index": 1.3,    # 검지 중요
        "middle": 1.0,   # 중지 보통
        "ring": 0.8,     # 약지 덜 중요
        "pinky": 0.6     # 새끼 가장 덜 중요
    }
    
    for finger, weight in finger_weights.items():
        print(f"   {finger:6}: weight {weight:.1f}")
    
    # 3. 물리적 제약 체크
    print("\n⚙️  PHYSICAL CONSTRAINT CHECKS:")
    print("   • Joint angle limits respected")
    print("   • No self-collision")
    print("   • Natural hand pose maintained")
    
    # 4. 시각적 품질 지표
    print("\n👁️  VISUAL QUALITY INDICATORS:")
    print("   • Keypoint-joint alignment")
    print("   • Smooth finger transitions") 
    print("   • Realistic hand configuration")
    
    return True


def calculate_retargeting_metrics(target_keypoints, optimized_joints, finger_info):
    """리타겟팅 메트릭 계산"""
    
    # 거리 계산
    distances = torch.norm(target_keypoints - optimized_joints, dim=-1)
    
    # 전체 평균 거리
    avg_distance = torch.mean(distances).item()
    
    # 손가락별 거리
    finger_distances = {}
    finger_weights = {"thumb": 1.5, "index": 1.3, "middle": 1.0, "ring": 0.8, "pinky": 0.6}
    
    weighted_sum = 0
    weight_sum = 0
    
    for i, joint_name in enumerate(finger_info):
        distance = distances[i].item()
        
        # 손가락 종류 판별
        finger_type = None
        for finger in finger_weights.keys():
            if finger in joint_name:
                finger_type = finger
                break
        
        if finger_type:
            if finger_type not in finger_distances:
                finger_distances[finger_type] = []
            finger_distances[finger_type].append(distance)
            
            # 가중 평균 계산
            weight = finger_weights[finger_type]
            weighted_sum += distance * weight
            weight_sum += weight
    
    # 가중 평균 거리
    weighted_avg_distance = weighted_sum / weight_sum if weight_sum > 0 else avg_distance
    
    # 성공도 등급 판정
    if weighted_avg_distance < 0.02:  # 2cm
        grade = "EXCELLENT ✅"
        color = "green"
    elif weighted_avg_distance < 0.05:  # 5cm
        grade = "GOOD 🟡"
        color = "yellow"
    elif weighted_avg_distance < 0.10:  # 10cm
        grade = "FAIR 🟠"
        color = "blue"
    else:
        grade = "POOR ❌"
        color = "red"
    
    return {
        "avg_distance": avg_distance,
        "weighted_avg_distance": weighted_avg_distance,
        "finger_distances": finger_distances,
        "grade": grade,
        "color": color,
        "individual_distances": distances
    }


def create_retargeting_report(metrics, sequence_info):
    """리타겟팅 결과 리포트 생성"""
    
    print(f"\n" + "="*60)
    print(f"📊 RETARGETING REPORT")
    print(f"="*60)
    
    print(f"🎯 Sequence: {sequence_info['seq_id']}")
    print(f"🎬 Frame: {sequence_info['frame_idx']}")
    print(f"✋ Hand: {sequence_info['side']} {sequence_info['dexhand']}")
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    cprint(f"   Grade: {metrics['grade']}", metrics['color'])
    print(f"   Average Distance: {metrics['avg_distance']*100:.2f} cm")
    print(f"   Weighted Distance: {metrics['weighted_avg_distance']*100:.2f} cm")
    
    print(f"\n👆 FINGER PERFORMANCE:")
    for finger, distances in metrics['finger_distances'].items():
        avg_dist = np.mean(distances)
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        
        if avg_dist < 0.02:
            status = "✅"
        elif avg_dist < 0.05:
            status = "🟡"
        elif avg_dist < 0.10:
            status = "🟠"
        else:
            status = "❌"
            
        print(f"   {finger:6}: {avg_dist*100:5.2f}cm avg, {max_dist*100:5.2f}cm max {status}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if metrics['weighted_avg_distance'] > 0.10:
        print("   • Increase optimization iterations")
        print("   • Check joint angle limits")
        print("   • Verify coordinate transformations")
    elif metrics['weighted_avg_distance'] > 0.05:
        print("   • Fine-tune learning rates")
        print("   • Add finger-specific weights")
    else:
        print("   • Retargeting quality is excellent!")
        print("   • Ready for downstream applications")
    
    return True


def visualize_distance_heatmap(metrics, save_path=None):
    """거리 히트맵 시각화"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 개별 관절 거리
    distances = metrics['individual_distances'].cpu().numpy()
    joint_names = [f"Joint_{i}" for i in range(len(distances))]
    
    colors = ['green' if d < 0.02 else 'yellow' if d < 0.05 else 'orange' if d < 0.10 else 'red' 
              for d in distances]
    
    bars1 = ax1.bar(range(len(distances)), distances * 100, color=colors, alpha=0.7)
    ax1.set_xlabel('Joint Index')
    ax1.set_ylabel('Distance (cm)')
    ax1.set_title('Individual Joint Distances')
    ax1.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Excellent (2cm)')
    ax1.axhline(y=5, color='yellow', linestyle='--', alpha=0.5, label='Good (5cm)')
    ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Fair (10cm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 손가락별 평균 거리
    finger_names = list(metrics['finger_distances'].keys())
    finger_avgs = [np.mean(metrics['finger_distances'][finger]) * 100 for finger in finger_names]
    
    finger_colors = ['green' if avg < 2 else 'yellow' if avg < 5 else 'orange' if avg < 10 else 'red' 
                     for avg in finger_avgs]
    
    bars2 = ax2.bar(finger_names, finger_avgs, color=finger_colors, alpha=0.7)
    ax2.set_xlabel('Finger')
    ax2.set_ylabel('Average Distance (cm)')
    ax2.set_title('Finger-wise Performance')
    ax2.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Excellent')
    ax2.axhline(y=5, color='yellow', linestyle='--', alpha=0.5, label='Good')
    ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Fair')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 값 표시
    for bar, value in zip(bars2, finger_avgs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}cm', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Distance heatmap saved to: {save_path}")
    
    plt.show()
    return fig


def main():
    print("🚀 Retargeting Quality Analyzer")
    
    # 분석 개요 출력
    analyze_retargeting_quality()
    
    # 실제 사용 예시
    print(f"\n" + "="*60)
    print("📝 USAGE EXAMPLE:")
    print("="*60)
    
    print("""
# 사용 방법:
from analyze_retargeting_quality import calculate_retargeting_metrics, create_retargeting_report

# 메트릭 계산
metrics = calculate_retargeting_metrics(
    target_keypoints=target_joints,    # MANO 키포인트 (떠있는 점들)
    optimized_joints=hand_joints,      # 최적화된 손 관절 위치
    finger_info=joint_names           # 관절 이름 리스트
)

# 리포트 생성
create_retargeting_report(metrics, {
    'seq_id': 'p001-folder/000',
    'frame_idx': 30,
    'side': 'right',
    'dexhand': 'inspire'
})

# 시각화
visualize_distance_heatmap(metrics, 'retargeting_quality.png')
""")
    
    print(f"\n💡 TIP: 현재 IsaacGym 창에서:")
    print("   • 색깔 구체 = MANO 키포인트 (목표)")
    print("   • 흰색 손 = Inspire Hand (현재 위치)")
    print("   • 가까울수록 = 리타겟팅 성공!")


if __name__ == "__main__":
    main() 