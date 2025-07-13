import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import cv2
from typing import Dict, List, Union, Optional, Tuple

try:
    import manopth
    from manopth.manolayer import ManoLayer
    MANO_AVAILABLE = True
except ImportError:
    print("Warning: manopth not available. Using mock implementation.")
    MANO_AVAILABLE = False

class MockManoLayer:
    """Mock MANO layer when manopth is not available"""
    def __init__(self, *args, **kwargs):
        pass
        
    def __call__(self, pose, beta):
        batch_size = pose.shape[0]
        # Mock joints: 21 joints for MANO
        joints = torch.randn(batch_size, 21, 3) * 0.1
        vertices = torch.randn(batch_size, 778, 3) * 0.1
        return vertices, joints

class MANOKeypointRenderer:
    """MANO 키포인트 렌더링 클래스"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        MANO 키포인트 렌더러 초기화
        
        Args:
            model_path: MANO 모델 파일 경로 (옵션)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if MANO_AVAILABLE and model_path and os.path.exists(model_path):
            # Real MANO layer
            self.mano_layer = {
                'left': ManoLayer(
                    mano_root=model_path, 
                    use_pca=False, 
                    ncomps=45,
                    side='left'
                ).to(self.device),
                'right': ManoLayer(
                    mano_root=model_path, 
                    use_pca=False, 
                    ncomps=45,
                    side='right'
                ).to(self.device)
            }
            print("Using real MANO model for keypoints")
        else:
            # Mock MANO layer
            self.mano_layer = {
                'left': MockManoLayer(),
                'right': MockManoLayer()
            }
            print("Using mock MANO model for keypoints")
        
        # MANO 관절 정보
        self.joint_names = [
            'wrist',           # 0
            'thumb_mcp',       # 1
            'thumb_pip',       # 2
            'thumb_dip',       # 3
            'thumb_tip',       # 4
            'index_mcp',       # 5
            'index_pip',       # 6
            'index_dip',       # 7
            'index_tip',       # 8
            'middle_mcp',      # 9
            'middle_pip',      # 10
            'middle_dip',      # 11
            'middle_tip',      # 12
            'ring_mcp',        # 13
            'ring_pip',        # 14
            'ring_dip',        # 15
            'ring_tip',        # 16
            'pinky_mcp',       # 17
            'pinky_pip',       # 18
            'pinky_dip',       # 19
            'pinky_tip'        # 20
        ]
        
        # 손가락별 색상
        self.finger_colors = {
            'thumb': 'red',
            'index': 'blue',
            'middle': 'green',
            'ring': 'orange',
            'pinky': 'purple',
            'wrist': 'black'
        }
    
    def load_gigahands_data(self, json_path: str) -> Dict:
        """
        GigaHands JSON 파일로부터 데이터 로드
        
        Args:
            json_path: JSON 파일 경로
            
        Returns:
            로드된 데이터 딕셔너리
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def poses_to_tensor(self, poses: List[List[float]]) -> torch.Tensor:
        """
        pose 리스트를 torch tensor로 변환
        
        Args:
            poses: pose 파라미터 리스트 (48D per pose)
            
        Returns:
            torch tensor [N, 48]
        """
        poses_array = np.array(poses)
        return torch.from_numpy(poses_array).float().to(self.device)
    
    def generate_keypoints(self, pose: torch.Tensor, beta: torch.Tensor, hand_type: str = 'right') -> torch.Tensor:
        """
        MANO 파라미터로부터 키포인트 생성
        
        Args:
            pose: pose 파라미터 [batch_size, 48]
            beta: shape 파라미터 [batch_size, 10]  
            hand_type: 'left' 또는 'right'
            
        Returns:
            joints [batch_size, 21, 3]
        """
        mano = self.mano_layer[hand_type]
        vertices, joints = mano(pose, beta)
        return joints
    
    def render_keypoint_sequence(
        self, 
        json_path: str, 
        hand_type: str = 'left',
        beta: Optional[torch.Tensor] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        output_dir: str = "mano_keypoint_outputs"
    ):
        """
        GigaHands 시퀀스를 키포인트로 렌더링
        
        Args:
            json_path: JSON 파일 경로
            hand_type: 'left' 또는 'right'
            beta: shape 파라미터 (None이면 기본값 사용)
            frame_range: 렌더링할 프레임 범위 (start, end)
            output_dir: 출력 디렉토리
        """
        # 데이터 로드
        data = self.load_gigahands_data(json_path)
        poses = data[hand_type]['poses']
        
        # 프레임 범위 설정
        if frame_range is None:
            start_frame, end_frame = 0, len(poses)
        else:
            start_frame, end_frame = frame_range
            end_frame = min(end_frame, len(poses))
        
        # Shape 파라미터 설정
        if beta is None:
            beta = torch.zeros(1, 10).to(self.device)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Rendering {end_frame - start_frame} keypoint frames for {hand_type} hand...")
        
        for frame_idx in range(start_frame, end_frame):
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}/{end_frame}")
            
            # 현재 프레임 pose
            pose = self.poses_to_tensor([poses[frame_idx]])
            
            # 키포인트 생성
            joints = self.generate_keypoints(pose, beta, hand_type)
            
            # numpy로 변환
            joints_np = joints[0].cpu().numpy()
            
            # 키포인트 시각화
            self._visualize_keypoints(
                joints_np, 
                title=f"{hand_type.capitalize()} Hand Keypoints - Frame {frame_idx}",
                save_path=os.path.join(output_dir, f"{hand_type}_keypoints_frame_{frame_idx:04d}.png")
            )
    
    def _visualize_keypoints(
        self, 
        joints: np.ndarray, 
        title: str = "MANO Keypoints",
        save_path: Optional[str] = None,
        show_labels: bool = True
    ):
        """
        키포인트 시각화
        
        Args:
            joints: 관절 좌표 [21, 3]
            title: 그래프 제목
            save_path: 저장 경로 (None이면 표시만)
            show_labels: 관절 이름 표시 여부
        """
        fig = plt.figure(figsize=(16, 6))
        
        # 3D 키포인트 플롯
        ax1 = fig.add_subplot(131, projection='3d')
        self._plot_3d_keypoints(ax1, joints, title + " - 3D", show_labels)
        
        # XY 평면 투영
        ax2 = fig.add_subplot(132)
        self._plot_2d_keypoints(ax2, joints[:, :2], title + " - XY View", show_labels)
        
        # XZ 평면 투영
        ax3 = fig.add_subplot(133)
        self._plot_2d_keypoints(ax3, joints[:, [0, 2]], title + " - XZ View", show_labels, 
                               xlabel='X', ylabel='Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_3d_keypoints(self, ax, joints: np.ndarray, title: str, show_labels: bool = True):
        """3D 키포인트 플롯"""
        # 관절별 색상과 크기
        colors = []
        sizes = []
        
        for i, joint_name in enumerate(self.joint_names):
            if 'thumb' in joint_name:
                colors.append(self.finger_colors['thumb'])
            elif 'index' in joint_name:
                colors.append(self.finger_colors['index'])
            elif 'middle' in joint_name:
                colors.append(self.finger_colors['middle'])
            elif 'ring' in joint_name:
                colors.append(self.finger_colors['ring'])
            elif 'pinky' in joint_name:
                colors.append(self.finger_colors['pinky'])
            else:  # wrist
                colors.append(self.finger_colors['wrist'])
            
            # 손목은 크게, 다른 관절은 작게
            sizes.append(100 if 'wrist' in joint_name else 50)
        
        # 관절 점들 그리기
        for i, (joint, color, size) in enumerate(zip(joints, colors, sizes)):
            ax.scatter(joint[0], joint[1], joint[2], c=color, s=size, alpha=0.8)
            
            if show_labels:
                ax.text(joint[0], joint[1], joint[2], f'{i}', fontsize=8)
        
        # 스켈레톤 연결선 그리기
        self._draw_hand_skeleton_3d(ax, joints)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 범례 추가
        for finger, color in self.finger_colors.items():
            ax.scatter([], [], [], c=color, s=50, label=finger)
        ax.legend()
    
    def _plot_2d_keypoints(self, ax, joints_2d: np.ndarray, title: str, show_labels: bool = True,
                          xlabel: str = 'X', ylabel: str = 'Y'):
        """2D 키포인트 플롯"""
        colors = []
        sizes = []
        
        for i, joint_name in enumerate(self.joint_names):
            if 'thumb' in joint_name:
                colors.append(self.finger_colors['thumb'])
            elif 'index' in joint_name:
                colors.append(self.finger_colors['index'])
            elif 'middle' in joint_name:
                colors.append(self.finger_colors['middle'])
            elif 'ring' in joint_name:
                colors.append(self.finger_colors['ring'])
            elif 'pinky' in joint_name:
                colors.append(self.finger_colors['pinky'])
            else:  # wrist
                colors.append(self.finger_colors['wrist'])
            
            sizes.append(100 if 'wrist' in joint_name else 50)
        
        # 관절 점들 그리기
        for i, (joint, color, size) in enumerate(zip(joints_2d, colors, sizes)):
            ax.scatter(joint[0], joint[1], c=color, s=size, alpha=0.8)
            
            if show_labels:
                ax.text(joint[0], joint[1], f'{i}', fontsize=8)
        
        # 2D 스켈레톤 연결선 그리기
        self._draw_hand_skeleton_2d(ax, joints_2d)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    def _draw_hand_skeleton_3d(self, ax, joints: np.ndarray):
        """3D 손 스켈레톤 그리기"""
        connections = [
            # 엄지
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 검지
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 중지
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 약지
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 소지
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for start, end in connections:
            ax.plot3D(
                [joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]],
                'gray', linewidth=2, alpha=0.6
            )
    
    def _draw_hand_skeleton_2d(self, ax, joints_2d: np.ndarray):
        """2D 손 스켈레톤 그리기"""
        connections = [
            # 엄지
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 검지
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 중지
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 약지
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 소지
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for start, end in connections:
            ax.plot(
                [joints_2d[start, 0], joints_2d[end, 0]],
                [joints_2d[start, 1], joints_2d[end, 1]],
                'gray', linewidth=2, alpha=0.6
            )
    
    def create_keypoint_animation(
        self,
        json_path: str,
        hand_type: str = 'left',
        beta: Optional[torch.Tensor] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        output_path: str = "mano_keypoint_animation.gif",
        fps: int = 10
    ):
        """
        키포인트 애니메이션 생성
        
        Args:
            json_path: JSON 파일 경로
            hand_type: 'left' 또는 'right'
            beta: shape 파라미터
            frame_range: 프레임 범위
            output_path: 출력 GIF 경로
            fps: 프레임 레이트
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("Animation requires pillow: pip install pillow")
            return
        
        # 데이터 로드
        data = self.load_gigahands_data(json_path)
        poses = data[hand_type]['poses']
        
        if frame_range is None:
            start_frame, end_frame = 0, min(len(poses), 50)  # 최대 50프레임
        else:
            start_frame, end_frame = frame_range
            end_frame = min(end_frame, len(poses))
        
        if beta is None:
            beta = torch.zeros(1, 10).to(self.device)
        
        # 모든 프레임의 키포인트 미리 계산
        all_joints = []
        
        print(f"Pre-computing {end_frame - start_frame} keypoint frames...")
        for frame_idx in range(start_frame, end_frame):
            pose = self.poses_to_tensor([poses[frame_idx]])
            joints = self.generate_keypoints(pose, beta, hand_type)
            all_joints.append(joints[0].cpu().numpy())
        
        # 애니메이션 생성
        fig = plt.figure(figsize=(15, 5))
        
        def animate(frame):
            fig.clear()
            joints = all_joints[frame]
            
            # 3D 뷰
            ax1 = fig.add_subplot(131, projection='3d')
            self._plot_3d_keypoints(ax1, joints, f'3D - Frame {frame + start_frame}', show_labels=False)
            
            # XY 뷰
            ax2 = fig.add_subplot(132)
            self._plot_2d_keypoints(ax2, joints[:, :2], f'XY - Frame {frame + start_frame}', show_labels=False)
            
            # XZ 뷰
            ax3 = fig.add_subplot(133)
            self._plot_2d_keypoints(ax3, joints[:, [0, 2]], f'XZ - Frame {frame + start_frame}', 
                                   show_labels=False, xlabel='X', ylabel='Z')
            
            plt.tight_layout()
        
        print("Creating keypoint animation...")
        anim = FuncAnimation(fig, animate, frames=len(all_joints), interval=1000//fps)
        
        print(f"Saving animation to {output_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()
        print(f"Animation saved to {output_path}")
    
    def analyze_keypoint_trajectory(
        self,
        json_path: str,
        hand_type: str = 'left',
        joint_idx: int = 8,  # index fingertip
        beta: Optional[torch.Tensor] = None,
        output_path: str = "keypoint_trajectory.png"
    ):
        """
        특정 키포인트의 궤적 분석
        
        Args:
            json_path: JSON 파일 경로
            hand_type: 'left' 또는 'right'
            joint_idx: 분석할 관절 인덱스
            beta: shape 파라미터
            output_path: 출력 경로
        """
        # 데이터 로드
        data = self.load_gigahands_data(json_path)
        poses = data[hand_type]['poses']
        
        if beta is None:
            beta = torch.zeros(1, 10).to(self.device)
        
        # 모든 프레임의 키포인트 계산
        trajectory = []
        
        print(f"Computing trajectory for joint {joint_idx} ({self.joint_names[joint_idx]})...")
        for frame_idx, pose_data in enumerate(poses):
            pose = self.poses_to_tensor([pose_data])
            joints = self.generate_keypoints(pose, beta, hand_type)
            trajectory.append(joints[0, joint_idx].cpu().numpy())
        
        trajectory = np.array(trajectory)  # [N, 3]
        
        # 궤적 시각화
        fig = plt.figure(figsize=(15, 10))
        
        # 3D 궤적
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.7)
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                   c='green', s=100, label='Start')
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                   c='red', s=100, label='End')
        ax1.set_title(f'3D Trajectory - {self.joint_names[joint_idx]}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # XY 투영
        ax2 = fig.add_subplot(222)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.7)
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, label='Start')
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, label='End')
        ax2.set_title('XY Projection')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        
        # 시간에 따른 변화
        ax3 = fig.add_subplot(223)
        ax3.plot(trajectory[:, 0], label='X', alpha=0.7)
        ax3.plot(trajectory[:, 1], label='Y', alpha=0.7)
        ax3.plot(trajectory[:, 2], label='Z', alpha=0.7)
        ax3.set_title('Position over Time')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Position')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 속도 분석
        ax4 = fig.add_subplot(224)
        velocity = np.diff(trajectory, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        ax4.plot(speed, 'r-', alpha=0.7)
        ax4.set_title('Speed over Time')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Speed')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory analysis saved to {output_path}")
        print(f"Total distance: {np.sum(speed):.4f}")
        print(f"Max speed: {np.max(speed):.4f}")
        print(f"Average speed: {np.mean(speed):.4f}")

def main():
    """메인 실행 함수"""
    # 렌더러 초기화
    renderer = MANOKeypointRenderer()
    
    # GigaHands 데이터 경로
    json_path = "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose/p001-folder/params/000.json"
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return
    
    print("=== MANO Keypoint Rendering ===")
    
    # 1. 정적 이미지 렌더링 (처음 10프레임)
    print("\n1. Rendering static keypoint images...")
    renderer.render_keypoint_sequence(
        json_path=json_path,
        hand_type='left',
        frame_range=(0, 10),
        output_dir="mano_keypoint_outputs"
    )
    
    # 2. 키포인트 애니메이션 생성 (처음 30프레임)
    print("\n2. Creating keypoint animation...")
    renderer.create_keypoint_animation(
        json_path=json_path,
        hand_type='left',
        frame_range=(0, 30),
        output_path="mano_left_keypoint_animation.gif",
        fps=10
    )
    
    # 3. 궤적 분석 (검지 끝)
    print("\n3. Analyzing keypoint trajectory...")
    renderer.analyze_keypoint_trajectory(
        json_path=json_path,
        hand_type='left',
        joint_idx=8,  # index fingertip
        output_path="index_fingertip_trajectory.png"
    )
    
    print("\nKeypoint rendering completed!")
    print("Check the following outputs:")
    print("- Static images: mano_keypoint_outputs/")
    print("- Animation: mano_left_keypoint_animation.gif")
    print("- Trajectory analysis: index_fingertip_trajectory.png")

if __name__ == "__main__":
    main() 