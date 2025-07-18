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
            print(f"📏 Episode length: {first_rollout['reward'].shape[0]} steps")
            print(f"🎯 Action dimension: {first_rollout['actions'].shape[1]}")
            
            # 보상 통계
            all_rewards = []
            for rollout_name in successful_rollouts[:10]:  # 처음 10개만
                rollout = f[f'rollouts/successful/{rollout_name}']
                total_reward = rollout['reward'][:].sum()
                all_rewards.append(total_reward)
            
            if all_rewards:
                print(f"💰 Reward stats: mean={np.mean(all_rewards):.2f}, std={np.std(all_rewards):.2f}")
                print(f"    min={np.min(all_rewards):.2f}, max={np.max(all_rewards):.2f}")
        
        return successful_rollouts, failed_rollouts

def visualize_rollouts(filepath, num_rollouts=3):
    """롤아웃 시각화"""
    with h5py.File(filepath, 'r') as f:
        successful_rollouts, _ = get_rollout_summary(filepath)
        
        if not successful_rollouts:
            print("❌ No successful rollouts found!")
            return
            
        # 처음 몇 개 롤아웃 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Rollout Analysis: {Path(filepath).name}', fontsize=16)
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(num_rollouts, len(successful_rollouts))))
        
        all_rewards = []
        all_episode_lengths = []
        
        for i, rollout_name in enumerate(successful_rollouts[:num_rollouts]):
            rollout = f[f'rollouts/successful/{rollout_name}']
            
            # 데이터 로드
            rewards = rollout['reward'][:]
            q = rollout['q'][:]
            dq = rollout['dq'][:]
            actions = rollout['actions'][:]
            
            total_reward = rewards.sum()
            all_rewards.append(total_reward)
            all_episode_lengths.append(len(rewards))
            
            # 1. 보상 변화
            axes[0,0].plot(rewards, color=colors[i], label=f'{rollout_name} (total: {total_reward:.1f})')
            
            # 2. 관절 각도 변화 (처음 5개 관절만)
            axes[0,1].plot(q[:, :5], alpha=0.7)
            
            # 3. 관절 속도 (RMS)
            dq_rms = np.sqrt(np.mean(dq**2, axis=1))
            axes[1,0].plot(dq_rms, color=colors[i], label=f'{rollout_name}')
            
            # 4. 행동 분포 (첫 번째 롤아웃만)
            if i == 0:
                axes[1,1].hist(actions.flatten(), bins=50, alpha=0.7, density=True)
        
        # 그래프 설정
        axes[0,0].set_title('Rewards over Time')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        axes[0,1].set_title('Joint Angles (first 5 joints)')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Angle (rad)')
        axes[0,1].grid(True)
        
        axes[1,0].set_title('Joint Velocity RMS')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('RMS Velocity (rad/s)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        axes[1,1].set_title('Action Distribution (first rollout)')
        axes[1,1].set_xlabel('Action Value')
        axes[1,1].set_ylabel('Density')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        
        # 저장
        output_path = Path(filepath).parent / f"{Path(filepath).stem}_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization: {output_path}")
        
        plt.show()
        
        # 통계 출력
        print(f"\n📊 Summary Statistics:")
        print(f"   Average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        print(f"   Average episode length: {np.mean(all_episode_lengths):.1f} ± {np.std(all_episode_lengths):.1f}")

def visualize_best_rollout(filepath):
    """최고 성능 롤아웃 상세 분석"""
    with h5py.File(filepath, 'r') as f:
        successful_rollouts, _ = get_rollout_summary(filepath)
        
        if not successful_rollouts:
            print("❌ No successful rollouts found!")
            return
            
        # 최고 보상 롤아웃 찾기
        best_rollout_name = None
        best_reward = -float('inf')
        
        for rollout_name in successful_rollouts:
            rollout = f[f'rollouts/successful/{rollout_name}']
            total_reward = rollout['reward'][:].sum()
            if total_reward > best_reward:
                best_reward = total_reward
                best_rollout_name = rollout_name
        
        print(f"🏆 Best rollout: {best_rollout_name} (reward: {best_reward:.2f})")
        
        # 최고 롤아웃 상세 분석
        best_rollout = f[f'rollouts/successful/{best_rollout_name}']
        
        rewards = best_rollout['reward'][:]
        q = best_rollout['q'][:]
        dq = best_rollout['dq'][:]
        base_state = best_rollout['base_state'][:]
        actions = best_rollout['actions'][:]
        
        # 상세 시각화
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Best Rollout Analysis: {best_rollout_name}', fontsize=16)
        
        # 1. 보상 변화
        axes[0,0].plot(rewards, 'b-', linewidth=2)
        axes[0,0].set_title(f'Reward (Total: {best_reward:.2f})')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].grid(True)
        
        # 2. 손목 위치 궤적
        axes[0,1].plot(base_state[:, 0], base_state[:, 1], 'r-', linewidth=2)
        axes[0,1].set_title('Hand Base Trajectory (XY)')
        axes[0,1].set_xlabel('X Position (m)')
        axes[0,1].set_ylabel('Y Position (m)')
        axes[0,1].grid(True)
        axes[0,1].axis('equal')
        
        # 3. 모든 관절 각도
        im1 = axes[1,0].imshow(q.T, aspect='auto', cmap='viridis')
        axes[1,0].set_title('Joint Angles Heatmap')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('Joint Index')
        plt.colorbar(im1, ax=axes[1,0], label='Angle (rad)')
        
        # 4. 관절 속도
        im2 = axes[1,1].imshow(dq.T, aspect='auto', cmap='plasma')
        axes[1,1].set_title('Joint Velocities Heatmap')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('Joint Index')
        plt.colorbar(im2, ax=axes[1,1], label='Velocity (rad/s)')
        
        # 5. 행동 히트맵
        im3 = axes[2,0].imshow(actions.T, aspect='auto', cmap='coolwarm')
        axes[2,0].set_title('Actions Heatmap')
        axes[2,0].set_xlabel('Time Step')
        axes[2,0].set_ylabel('Action Index')
        plt.colorbar(im3, ax=axes[2,0], label='Action Value')
        
        # 6. 손목 높이 변화
        axes[2,1].plot(base_state[:, 2], 'g-', linewidth=2)
        axes[2,1].set_title('Hand Height')
        axes[2,1].set_xlabel('Time Step')
        axes[2,1].set_ylabel('Z Position (m)')
        axes[2,1].grid(True)
        
        plt.tight_layout()
        
        # 저장
        output_path = Path(filepath).parent / f"{Path(filepath).stem}_best_rollout.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved best rollout analysis: {output_path}")
        
        plt.show()

if __name__ == "__main__":
    # 파일 경로
    hdf5_path = "/scratch2/jisoo6687/handy_track/dumps/dump_DexHandImitator__07-17-14-56-30/rollouts.hdf5"
    
    # 실행
    print("🚀 HDF5 Rollout Analysis")
    print("="*50)
    
    # 1. 구조 탐색
    explore_hdf5_structure(hdf5_path)
    
    # 2. 기본 시각화
    visualize_rollouts(hdf5_path, num_rollouts=3)
    
    # 3. 최고 성능 롤아웃 분석
    visualize_best_rollout(hdf5_path) 