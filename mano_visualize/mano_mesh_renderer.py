import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import trimesh
import os
from typing import Dict, List, Union, Optional, Tuple

import manopth
from manopth.manolayer import ManoLayer

class MANOMeshRenderer:
    """MANO 메쉬 렌더링 클래스"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        MANO 렌더러 초기화
        
        Args:
            model_path: MANO 모델 파일 경로 (옵션)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Real MANO layer
        self.mano_layer = {
            'left': ManoLayer(
                mano_root=model_path, 
                use_pca=True,
                side='left',
                flat_hand_mean=False
            ).to(self.device),
            'right': ManoLayer(
                mano_root=model_path, 
                use_pca=True,
                side='right',
                flat_hand_mean=False
            ).to(self.device)
        }
        print("Using real MANO model")
    
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
    
    def generate_mesh(self, pose: torch.Tensor, beta: torch.Tensor, hand_type: str = 'right') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MANO 파라미터로부터 메쉬 생성
        
        Args:
            pose: pose 파라미터 [batch_size, 48]
            beta: shape 파라미터 [batch_size, 10]  
            hand_type: 'left' 또는 'right'
            
        Returns:
            vertices, joints
        """
        mano = self.mano_layer[hand_type]
        vertices, joints = mano(pose, beta)
        return vertices, joints
    
    def _get_mano_faces(self):
        """
        MANO mesh faces 가져오기
        
        Returns:
            faces: numpy array [1538, 3]
        """
        # 아무 hand type에서나 faces를 가져옴 (left/right 동일)
        mano = list(self.mano_layer.values())[0]
        if hasattr(mano, 'th_faces'):
            return mano.th_faces.detach().cpu().numpy()
        elif hasattr(mano, 'faces'):
            return mano.faces.detach().cpu().numpy()
        else:
            # Fallback: basic triangle faces (덜 정확하지만 작동함)
            print("Warning: MANO faces not found, using fallback triangulation")
            return self._create_fallback_faces()
    
    def _create_fallback_faces(self):
        """
        MANO faces를 찾을 수 없을 때 사용하는 fallback triangulation
        
        Returns:
            faces: numpy array [N, 3] - 기본적인 삼각형 면들
        """
        # 간단한 Delaunay triangulation 기반 faces 생성
        # 실제로는 MANO topology와 다르지만 시각화는 가능
        from scipy.spatial import Delaunay
        
        # 손 중심 영역의 점들로 기본 triangulation 생성
        # 이는 완전히 정확하지 않지만 기본적인 mesh 구조를 제공
        n_vertices = 778  # MANO 기본 vertex 수
        
        # 간단한 face 패턴 생성 (실제로는 복잡하지만 기본 시각화용)
        faces = []
        step = 3
        for i in range(0, n_vertices - step, step):
            if i + 2 < n_vertices:
                faces.append([i, i+1, i+2])
        
        return np.array(faces)
    
    def render_mesh_sequence(
        self, 
        json_path: str, 
        hand_type: str = 'left',
        beta: Optional[torch.Tensor] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        output_dir: str = "mano_mesh_outputs"
    ):
        """
        GigaHands 시퀀스를 메쉬로 렌더링
        
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
        
        print(f"Rendering {end_frame - start_frame} frames for {hand_type} hand...")
        
        for frame_idx in range(start_frame, end_frame):
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}/{end_frame}")
            
            # 현재 프레임 pose
            pose = self.poses_to_tensor([poses[frame_idx]])
            
            # 메쉬 생성
            vertices, joints = self.generate_mesh(pose, beta, hand_type)
            
            # numpy로 변환
            vertices_np = vertices[0].cpu().numpy()
            joints_np = joints[0].cpu().numpy()
            
            # 메쉬 시각화
            self._visualize_mesh(
                vertices_np, 
                joints_np, 
                title=f"{hand_type.capitalize()} Hand - Frame {frame_idx}",
                save_path=os.path.join(output_dir, f"{hand_type}_mesh_frame_{frame_idx:04d}.png")
            )
    
    def _visualize_mesh(
        self, 
        vertices: np.ndarray, 
        joints: np.ndarray, 
        title: str = "MANO Mesh",
        save_path: Optional[str] = None,
        show_joints: bool = True
    ):
        """
        메쉬 시각화 (개선된 버전 - 실제 mesh faces 사용)
        
        Args:
            vertices: 정점 좌표 [778, 3]
            joints: 관절 좌표 [21, 3]
            title: 그래프 제목
            save_path: 저장 경로 (None이면 표시만)
            show_joints: 관절 표시 여부
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # MANO 손 mesh faces 가져오기 (기본 MANO topology)
        faces = self._get_mano_faces()
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 실제 mesh 렌더링 (faces 포함)
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Mesh collection 생성
        mesh_collection = []
        for face in faces:
            if all(idx < len(vertices) for idx in face):  # 유효한 face인지 확인
                triangle = vertices[face]
                mesh_collection.append(triangle)
        
        # Mesh 추가
        if mesh_collection:
            mesh = Poly3DCollection(mesh_collection, alpha=0.7, facecolor='lightblue', 
                                  edgecolor='darkblue', linewidth=0.1)
            ax1.add_collection3d(mesh)
        
        if show_joints:
            # 관절 표시
            ax1.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                       c='red', s=50, alpha=0.9, label='Joints')
            # 스켈레톤 그리기
            self._draw_hand_skeleton(ax1, joints)
        
        # 동일한 aspect ratio 설정
        max_range = np.array([vertices[:,0].max()-vertices[:,0].min(),
                             vertices[:,1].max()-vertices[:,1].min(),
                             vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
        mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
        mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
        mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
        
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'{title} - 3D Mesh', fontsize=14, fontweight='bold')
        if show_joints:
            ax1.legend()
        
        # 2. Wireframe 뷰
        ax2 = fig.add_subplot(222, projection='3d')
        
        # Vertices as points
        ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   c='lightcoral', s=1, alpha=0.3, label='Vertices')
        
        # Wireframe edges (sample for performance)
        for face in faces[::10]:  # 모든 10번째 face만 그리기
            if all(idx < len(vertices) for idx in face):
                triangle = vertices[face]
                for i in range(3):
                    start = triangle[i]
                    end = triangle[(i+1) % 3]
                    ax2.plot3D([start[0], end[0]], [start[1], end[1]], 
                              [start[2], end[2]], 'b-', alpha=0.2, linewidth=0.5)
        
        if show_joints:
            ax2.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                       c='red', s=50, alpha=0.9, label='Joints')
            self._draw_hand_skeleton(ax2, joints)
        
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Wireframe View', fontsize=14, fontweight='bold')
        if show_joints:
            ax2.legend()
        
        # 3. Top view (XY 투영)
        ax3 = fig.add_subplot(223)
        ax3.scatter(vertices[:, 0], vertices[:, 1], c='lightblue', s=1, alpha=0.5, label='Vertices')
        
        if show_joints:
            ax3.scatter(joints[:, 0], joints[:, 1], c='red', s=50, alpha=0.9, label='Joints')
            self._draw_hand_skeleton_2d(ax3, joints)
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Top View (XY plane)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # 4. Side view (XZ 투영)
        ax4 = fig.add_subplot(224)
        ax4.scatter(vertices[:, 0], vertices[:, 2], c='lightblue', s=1, alpha=0.5, label='Vertices')
        
        if show_joints:
            ax4.scatter(joints[:, 0], joints[:, 2], c='red', s=50, alpha=0.9, label='Joints')
            # 2D 스켈레톤 (XZ plane용)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),    # 엄지
                (0, 5), (5, 6), (6, 7), (7, 8),    # 검지
                (0, 9), (9, 10), (10, 11), (11, 12), # 중지
                (0, 13), (13, 14), (14, 15), (15, 16), # 약지
                (0, 17), (17, 18), (18, 19), (19, 20)  # 소지
            ]
            for start_idx, end_idx in connections:
                if start_idx < len(joints) and end_idx < len(joints):
                    start_pos = joints[start_idx]
                    end_pos = joints[end_idx]
                    ax4.plot([start_pos[0], end_pos[0]], [start_pos[2], end_pos[2]], 
                            'r-', linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        ax4.set_title('Side View (XZ plane)', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def _draw_hand_skeleton(self, ax, joints: np.ndarray):
        """3D 손 스켈레톤 그리기"""
        # MANO 관절 연결 구조
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
                'r-', linewidth=2, alpha=0.7
            )
    
    def _draw_hand_skeleton_2d(self, ax, joints: np.ndarray):
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
                [joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                'r-', linewidth=2, alpha=0.7
            )
    
    def create_mesh_animation(
        self,
        json_path: str,
        hand_type: str = 'left',
        beta: Optional[torch.Tensor] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        output_path: str = "mano_animation.gif",
        fps: int = 10
    ):
        """
        메쉬 애니메이션 생성
        
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
        
        # 모든 프레임의 메쉬 미리 계산
        all_vertices = []
        all_joints = []
        
        print(f"Pre-computing {end_frame - start_frame} frames...")
        for frame_idx in range(start_frame, end_frame):
            pose = self.poses_to_tensor([poses[frame_idx]])
            vertices, joints = self.generate_mesh(pose, beta, hand_type)
            all_vertices.append(vertices[0].cpu().numpy())
            all_joints.append(joints[0].cpu().numpy())
        
        # 애니메이션 생성
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            ax.clear()
            vertices = all_vertices[frame]
            joints = all_joints[frame]
            
            # 실제 mesh faces로 렌더링
            faces = self._get_mano_faces()
            
            # Mesh collection 생성
            mesh_collection = []
            for face in faces[::5]:  # 성능을 위해 일부만 사용
                if all(idx < len(vertices) for idx in face):
                    triangle = vertices[face]
                    mesh_collection.append(triangle)
            
            # Mesh 추가
            if mesh_collection:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                mesh = Poly3DCollection(mesh_collection, alpha=0.6, facecolor='lightblue', 
                                      edgecolor='none')
                ax.add_collection3d(mesh)
            
            # 관절과 스켈레톤
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                      c='red', s=50, alpha=0.8)
            self._draw_hand_skeleton(ax, joints)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{hand_type.capitalize()} Hand Animation - Frame {frame + start_frame}')
            
            # 고정된 뷰 범위
            ax.set_xlim([-0.2, 0.2])
            ax.set_ylim([-0.2, 0.2])
            ax.set_zlim([-0.2, 0.2])
        
        print("Creating animation...")
        anim = FuncAnimation(fig, animate, frames=len(all_vertices), interval=1000//fps)
        
        print(f"Saving animation to {output_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()
        print(f"Animation saved to {output_path}")

def main():
    """메인 실행 함수"""
    # 렌더러 초기화
    renderer = MANOMeshRenderer()
    
    # GigaHands 데이터 경로
    json_path = "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose/p001-folder/params/000.json"
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return
    
    print("=== MANO Mesh Rendering ===")
    
    # 1. 정적 이미지 렌더링 (처음 10프레임)
    print("\n1. Rendering static images...")
    renderer.render_mesh_sequence(
        json_path=json_path,
        hand_type='left',
        frame_range=(0, 10),
        output_dir="mano_mesh_outputs"
    )
    
    # 2. 애니메이션 생성 (처음 30프레임)
    print("\n2. Creating animation...")
    renderer.create_mesh_animation(
        json_path=json_path,
        hand_type='left',
        frame_range=(0, 30),
        output_path="mano_left_hand_animation.gif",
        fps=10
    )
    
    print("\nMesh rendering completed!")
    print("Check the following outputs:")
    print("- Static images: mano_mesh_outputs/")
    print("- Animation: mano_left_hand_animation.gif")

if __name__ == "__main__":
    main() 