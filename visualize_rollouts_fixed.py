#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def explore_hdf5_structure(filepath):
    """HDF5 파일 구조 탐색"""
    print(f"🔍 Exploring: {filepath}")
    print("="*50)
    
    with h5py.File(filepath, 'r') as f:
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 {name}: {obj.shape} {obj.dtype}")
            else:
                print(f"{indent}📁 {name}/")
        
        f.visititems(print_structure)
    print()

def get_rollout_summary(filepath):
    """롤아웃 요약 정보"""
    with h5py.File(filepath, 'r') as f:
        successful_rollouts = list(f['rollouts/successful'].keys()) if 'rollouts/successful' in f else []
        failed_rollouts = list(f['rollouts/failed'].keys()) if 'rollouts/failed' in f else []
        
        print(f"📈 Successful rollouts: {len(successful_rollouts)}")
        print(f"📉 Failed rollouts: {len(failed_rollouts)}")
        
        if successful_rollouts:
            # 첫 번째 성공 롤아웃 정보
            first_rollout = f[f'rollouts/successful/{successful_rollouts[0]}']
            available_keys = list(first_rollout.keys())
            print(f"🔑 Available data keys: {available_keys}")
            print(f"📏 Episode length: {first_rollout['q'].shape[0]} steps")
            print(f"🎯 Action dimension: {first_rollout['actions'].shape[1]}")
            print(f"🤖 Joint dimension: {first_rollout['q'].shape[1]}")
            
            # 에피소드 길이 분포
            episode_lengths = []
            for rollout_name in successful_rollouts:
                rollout = f[f'rollouts/successful/{rollout_name}']
                episode_lengths.append(rollout['q'].shape[0])
            
            print(f"📊 Episode length stats: mean={np.mean(episode_lengths):.1f}, std={np.std(episode_lengths):.1f}")
            print(f"    min={np.min(episode_lengths)}, max={np.max(episode_lengths)}")
        
        return successful_rollouts, failed_rollouts

def visualize_rollouts(filepath, num_rollouts=3):
    """롤아웃 시각화 (reward 없이)"""
    with h5py.File(filepath, 'r') as f:
        successful_rollouts, _ = get_rollout_summary(filepath)
        
        if not successful_rollouts:
            print("❌ No successful rollouts found!")
            return
            
        # 처음 몇 개 롤아웃 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Rollout Analysis: {Path(filepath).name}', fontsize=16)
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(num_rollouts, len(successful_rollouts))))
        
        all_episode_lengths = []
        
        for i, rollout_name in enumerate(successful_rollouts[:num_rollouts]):
            rollout = f[f'rollouts/successful/{rollout_name}']
            
            # 데이터 로드
            q = rollout['q'][:]
            dq = rollout['dq'][:]
            actions = rollout['actions'][:]
            base_state = rollout['base_state'][:]
            
            episode_length = len(q)
            all_episode_lengths.append(episode_length)
            
            # 1. 관절 각도 변화 (처음 6개 관절)
            time_steps = np.arange(len(q))
            for j in range(min(6, q.shape[1])):
                axes[0,0].plot(time_steps, q[:, j], alpha=0.7, label=f'Joint {j+1}' if i == 0 else "")
            
            # 2. 관절 속도 RMS
            dq_rms = np.sqrt(np.mean(dq**2, axis=1))
            axes[0,1].plot(time_steps, dq_rms, color=colors[i], label=f'{rollout_name} (len: {episode_length})')
            
            # 3. 손목 위치 궤적 (XY)
            axes[0,2].plot(base_state[:, 0], base_state[:, 1], color=colors[i], 
                          label=f'{rollout_name}', linewidth=2)
            axes[0,2].scatter(base_state[0, 0], base_state[0, 1], color=colors[i], 
                             marker='o', s=50, zorder=5)  # 시작점
            axes[0,2].scatter(base_state[-1, 0], base_state[-1, 1], color=colors[i], 
                             marker='*', s=100, zorder=5)  # 끝점
            
            # 4. 행동 히트맵 (첫 번째 롤아웃만)
            if i == 0:
                im = axes[1,0].imshow(actions.T, aspect='auto', cmap='coolwarm')
                plt.colorbar(im, ax=axes[1,0], label='Action Value')
            
            # 5. 손목 높이 변화
            axes[1,1].plot(time_steps, base_state[:, 2], color=colors[i], 
                          label=f'{rollout_name}', linewidth=2)
            
            # 6. 관절 속도 분포 (첫 번째 롤아웃만)
            if i == 0:
                axes[1,2].hist(dq.flatten(), bins=50, alpha=0.7, density=True, color=colors[i])
        
        # 그래프 설정
        axes[0,0].set_title('Joint Angles (first 6 joints)')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Angle (rad)')
        if num_rollouts == 1:  # 하나만 표시할 때만 legend
            axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True)
        
        axes[0,1].set_title('Joint Velocity RMS')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('RMS Velocity (rad/s)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        axes[0,2].set_title('Hand Base Trajectory (XY)')
        axes[0,2].set_xlabel('X Position (m)')
        axes[0,2].set_ylabel('Y Position (m)')
        axes[0,2].legend()
        axes[0,2].grid(True)
        axes[0,2].axis('equal')
        
        axes[1,0].set_title(f'Actions Heatmap ({successful_rollouts[0]})')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('Action Index')
        
        axes[1,1].set_title('Hand Height')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('Z Position (m)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        axes[1,2].set_title(f'Joint Velocity Distribution ({successful_rollouts[0]})')
        axes[1,2].set_xlabel('Velocity (rad/s)')
        axes[1,2].set_ylabel('Density')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        
        # 저장
        output_path = Path(filepath).parent / f"{Path(filepath).stem}_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization: {output_path}")
        
        plt.show()
        
        # 통계 출력
        print(f"\n📊 Summary Statistics:")
        print(f"   Average episode length: {np.mean(all_episode_lengths):.1f} ± {np.std(all_episode_lengths):.1f}")
        print(f"   Number of successful rollouts: {len(successful_rollouts)}")

def analyze_motion_patterns(filepath):
    """모션 패턴 상세 분석"""
    with h5py.File(filepath, 'r') as f:
        successful_rollouts, _ = get_rollout_summary(filepath)
        
        if not successful_rollouts:
            print("❌ No successful rollouts found!")
            return
            
        print(f"\n🔍 Analyzing motion patterns from {len(successful_rollouts)} rollouts...")
        
        # 모든 롤아웃의 통계 수집
        all_joint_ranges = []
        all_speeds = []
        all_trajectories = []
        
        for rollout_name in successful_rollouts:
            rollout = f[f'rollouts/successful/{rollout_name}']
            
            q = rollout['q'][:]
            dq = rollout['dq'][:]
            base_state = rollout['base_state'][:]
            
            # 관절 가동 범위
            joint_ranges = np.max(q, axis=0) - np.min(q, axis=0)
            all_joint_ranges.append(joint_ranges)
            
            # 평균 속도
            avg_speed = np.mean(np.abs(dq), axis=0)
            all_speeds.append(avg_speed)
            
            # 궤적 길이
            trajectory_length = np.sum(np.sqrt(np.sum(np.diff(base_state[:, :3], axis=0)**2, axis=1)))
            all_trajectories.append(trajectory_length)
        
        # 통계 분석
        all_joint_ranges = np.array(all_joint_ranges)
        all_speeds = np.array(all_speeds)
        
        print(f"📈 Motion Statistics:")
        print(f"   Average trajectory length: {np.mean(all_trajectories):.3f} ± {np.std(all_trajectories):.3f} m")
        print(f"   Most active joint: Joint {np.argmax(np.mean(all_joint_ranges, axis=0)) + 1} (range: {np.max(np.mean(all_joint_ranges, axis=0)):.3f} rad)")
        print(f"   Fastest joint: Joint {np.argmax(np.mean(all_speeds, axis=0)) + 1} (speed: {np.max(np.mean(all_speeds, axis=0)):.3f} rad/s)")
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 관절별 가동 범위
        joint_indices = np.arange(all_joint_ranges.shape[1])
        axes[0].boxplot([all_joint_ranges[:, i] for i in joint_indices])
        axes[0].set_title('Joint Range of Motion')
        axes[0].set_xlabel('Joint Index')
        axes[0].set_ylabel('Range (rad)')
        axes[0].grid(True)
        
        # 관절별 평균 속도
        axes[1].boxplot([all_speeds[:, i] for i in joint_indices])
        axes[1].set_title('Average Joint Speeds')
        axes[1].set_xlabel('Joint Index')
        axes[1].set_ylabel('Speed (rad/s)')
        axes[1].grid(True)
        
        # 궤적 길이 분포
        axes[2].hist(all_trajectories, bins=10, alpha=0.7, edgecolor='black')
        axes[2].set_title('Trajectory Length Distribution')
        axes[2].set_xlabel('Length (m)')
        axes[2].set_ylabel('Count')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # 저장
        output_path = Path(filepath).parent / f"{Path(filepath).stem}_motion_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved motion analysis: {output_path}")
        
        plt.show()

if __name__ == "__main__":
    # 파일 경로
    hdf5_path = "/scratch2/jisoo6687/handy_track/dumps/dump_DexHandImitator__07-17-14-56-30/rollouts.hdf5"
    
    # 실행
    print("🚀 HDF5 Rollout Analysis (Fixed Version)")
    print("="*50)
    
    # 1. 구조 탐색
    explore_hdf5_structure(hdf5_path)
    
    # 2. 기본 시각화
    visualize_rollouts(hdf5_path, num_rollouts=3)
    
    # 3. 모션 패턴 분석
    analyze_motion_patterns(hdf5_path) 