from isaacgym import gymapi, gymtorch, gymutil
from main.dataset.mano2dexhand_gigahands import load_gigahands_sequence, pack_gigahands_data
from main.dataset.transform import rot6d_to_rotmat, aa_to_rotmat, rotmat_to_aa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch
import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import subprocess
import ffmpeg
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from termcolor import cprint


def load_original_mano_sequence(data_dir, seq_id, side="right", max_frames=None):
    """GigaHands 원본 MANO 키포인트 전체 시퀀스 로드 및 좌표계 변환"""
    
    # 원본 시퀀스 데이터 로드
    if "@" in seq_id:
        seq_id, frame_offset = seq_id.split("@")
        frame_offset = int(frame_offset)
    else:
        frame_offset = 0
        
    motion_data, scene_name, sequence_name = load_gigahands_sequence(data_dir, seq_id, side)
    cprint(f"Loaded sequence {sequence_name} with {motion_data.shape[0]} frames", "green")
    
    # 프레임 수 제한
    if max_frames is not None and motion_data.shape[0] > max_frames:
        motion_data = motion_data[:max_frames]
        cprint(f"Limited to {max_frames} frames", "yellow")
    
    # dexhand 생성 (좌표계 변환용)
    dexhand = DexHandFactory.create_hand("inspire", side)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # MuJoCo to Isaac Gym 변환 매트릭스
    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
        np.array([np.pi / 2, 0, 0])
    )
    mujoco2gym_transf[:3, 3] = np.array([0, 0, 0.5])
    mujoco2gym_transf = torch.tensor(mujoco2gym_transf, dtype=torch.float32, device=device)
    
    all_joints_sequence = []
    
    cprint("Converting coordinates for all frames...", "cyan")
    for frame_idx in tqdm(range(motion_data.shape[0]), desc="Processing frames"):
        frame_data = motion_data[frame_idx:frame_idx+1]  # [1, 42, 3]
        
        # pack_gigahands_data와 동일한 변환 적용
        demo_data = pack_gigahands_data(frame_data, dexhand, side)
        demo_data["wrist_pos"] = demo_data["wrist_pos"].to(device)
        demo_data["wrist_rot"] = demo_data["wrist_rot"].to(device)
        for joint_name in demo_data["mano_joints"]:
            demo_data["mano_joints"][joint_name] = demo_data["mano_joints"][joint_name].to(device)
        
        # 손목 변환
        wrist_pos = demo_data["wrist_pos"][0]  # [3]
        wrist_rot = demo_data["wrist_rot"][0]  # [3] axis-angle
        wrist_rot_matrix = aa_to_rotmat(wrist_rot.unsqueeze(0))[0]  # [3, 3]
        
        # 좌표계 변환 적용
        wrist_pos_transformed = (mujoco2gym_transf[:3, :3] @ wrist_pos.T).T + mujoco2gym_transf[:3, 3]
        wrist_rot_transformed = mujoco2gym_transf[:3, :3] @ wrist_rot_matrix
        
        # MANO 조인트들을 하나의 텐서로 결합
        mano_joints = torch.cat([
            demo_data["mano_joints"][dexhand.to_hand(j_name)[0]]
            for j_name in dexhand.body_names
            if dexhand.to_hand(j_name)[0] != "wrist"
        ], dim=-1).view(1, -1, 3)
        
        # MANO 조인트들도 좌표계 변환
        mano_joints = mano_joints.view(-1, 3)
        mano_joints_world = mano_joints.view(-1, 3)
        mano_joints_transformed = (mujoco2gym_transf[:3, :3] @ mano_joints_world.T).T + mujoco2gym_transf[:3, 3]
        
        # 손목 포함한 전체 조인트
        all_joints = torch.cat([wrist_pos_transformed.unsqueeze(0), mano_joints_transformed], dim=0).cpu()
        all_joints_sequence.append(all_joints.numpy())
    
    all_joints_sequence = np.stack(all_joints_sequence)  # [num_frames, num_joints, 3]
    
    return all_joints_sequence, dexhand.body_names, sequence_name


def create_video_from_images(image_dir, output_path, fps=30):
    """
    이미지 디렉토리에서 비디오 생성 (해상도 짝수 처리 포함)
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return False
    
    print(f"Creating video: {output_path}")
    print(f"Found {len(image_files)} images")
    
    # 연속된 프레임인지 확인하고 필요시 rename
    expected_pattern = True
    for i, filename in enumerate(image_files):
        expected_name = f"frame_{i:04d}.png"
        if filename != expected_name:
            expected_pattern = False
    
    if not expected_pattern:
        print("Warning: Frames are not in sequential order. Renaming...")
        for i, old_filename in enumerate(image_files):
            old_path = os.path.join(image_dir, old_filename)
            new_filename = f"frame_{i:04d}.png"
            new_path = os.path.join(image_dir, new_filename)
            if old_path != new_path:
                os.rename(old_path, new_path)
        print("Files renamed to sequential order")
    
    input_pattern = os.path.join(image_dir, 'frame_%04d.png')
    
    try:
        print("Creating video with ffmpeg-python...")
        
        # ffmpeg-python 스트림 생성
        input_stream = ffmpeg.input(input_pattern, framerate=fps)
        
        # 해상도를 짝수로 맞추기 위해 scale 필터 적용 (libx264는 짝수 해상도 필요)
        scaled_stream = ffmpeg.filter(input_stream, 'scale', 'trunc(iw/2)*2', 'trunc(ih/2)*2')
        
        output_stream = ffmpeg.output(
            scaled_stream, 
            output_path,
            vcodec='libx264',
            pix_fmt='yuv420p',
            crf=18,
            loglevel='info'
        )
        
        # 기존 파일 덮어쓰기 허용하고 실행
        ffmpeg.run(output_stream, overwrite_output=True, quiet=False)
        
        print(f"Video created successfully: {output_path}")
        
        # 생성된 비디오 파일 확인
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Video file size: {file_size / (1024*1024):.2f} MB")
            return True
        
    except Exception as e:
        print(f"Video creation failed: {e}")
        return False
    
    return False


def render_original_sequence_multiview(joints_sequence, joint_names, sequence_name, 
                                     save_dir="renders_original", fps=30):
    """
    원본 MANO 키포인트 시퀀스를 6개 뷰에서 렌더링하여 비디오 생성
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_frames = len(joints_sequence)
    cprint(f"Rendering {num_frames} frames for sequence: {sequence_name}", "blue")
    
    # 다양한 시점 설정: (elev, azim)
    views = [
        (30, 45),
        (30, -45),
        (0, 90),
        (90, 0),
        (45, 180),
        (20, 270)
    ]
    
    # 손가락별 연결 정의
    finger_connections = {
        'thumb': [],
        'index': [],
        'middle': [],
        'ring': [],
        'pinky': [],
        'wrist': []
    }
    
    # joint_names를 손가락별로 분류
    for i, name in enumerate(joint_names):
        if 'thumb' in name:
            finger_connections['thumb'].append(i)
        elif 'index' in name:
            finger_connections['index'].append(i)
        elif 'middle' in name:
            finger_connections['middle'].append(i)
        elif 'ring' in name:
            finger_connections['ring'].append(i)
        elif 'pinky' in name:
            finger_connections['pinky'].append(i)
        else:
            finger_connections['wrist'].append(i)
    
    # 손가락별 색상 정의
    finger_colors = {
        'thumb': 'red',
        'index': 'blue', 
        'middle': 'green',
        'ring': 'orange',
        'pinky': 'purple',
        'wrist': 'black'
    }

    # 전체 시퀀스의 범위 계산 (일관된 축 범위를 위해)
    all_points = joints_sequence.reshape(-1, 3)
    max_range = np.ptp(all_points, axis=0).max() * 0.6
    center = all_points.mean(axis=0)
    
    # 각 view별로 임시 디렉토리 생성
    temp_dirs = {}
    
    print(f"Processing {len(views)} views...")
    
    # 각 view별로 프레임 렌더링
    for view_idx, (elev, azim) in enumerate(views):
        temp_dir = os.path.join(save_dir, f"temp_view{view_idx+1}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_dirs[view_idx] = temp_dir
        
        cprint(f"Rendering view {view_idx+1}/{len(views)} (elev={elev}, azim={azim})", "cyan")
        
        for frame_idx in tqdm(range(num_frames), desc=f"View {view_idx+1}"):
            joint_positions_np = joints_sequence[frame_idx]  # [num_joints, 3]
            
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")

            # 손가락별로 관절 연결
            for finger, indices in finger_connections.items():
                if len(indices) > 0:
                    color = finger_colors[finger]
                    
                    # 관절 점들 그리기
                    finger_points = joint_positions_np[indices]
                    ax.scatter(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2], 
                              c=color, s=60, label=finger, alpha=0.8)
                    
                    # 손가락 관절들을 순서대로 연결 (wrist 제외)
                    if finger != 'wrist' and len(indices) > 1:
                        # 관절명에 따라 순서 정렬
                        sorted_indices = []
                        for joint_type in ['proximal', 'intermediate', 'tip']:
                            for idx in indices:
                                if joint_type in joint_names[idx]:
                                    sorted_indices.append(idx)
                        
                        # 정렬된 순서대로 연결
                        if len(sorted_indices) > 1:
                            sorted_points = joint_positions_np[sorted_indices]
                            ax.plot(sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2], 
                                   c=color, linewidth=2, alpha=0.7)
                    
                    # 손목에서 각 손가락 첫 관절로 연결
                    if finger != 'wrist' and len(finger_connections['wrist']) > 0:
                        wrist_idx = finger_connections['wrist'][0]
                        if len(indices) > 0:
                            first_joint_idx = indices[0]
                            for idx in indices:
                                if 'proximal' in joint_names[idx]:
                                    first_joint_idx = idx
                                    break
                            
                            wrist_pos = joint_positions_np[wrist_idx]
                            first_joint_pos = joint_positions_np[first_joint_idx]
                            ax.plot([wrist_pos[0], first_joint_pos[0]], 
                                   [wrist_pos[1], first_joint_pos[1]], 
                                   [wrist_pos[2], first_joint_pos[2]], 
                                   c=color, linewidth=1.5, alpha=0.5, linestyle='--')
            
            # 범례는 첫 번째 프레임에만 추가
            if frame_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Original MANO - Frame {frame_idx+1}/{num_frames}\nView {view_idx+1} (elev={elev}, azim={azim})")
            ax.view_init(elev=elev, azim=azim)

            # 일관된 축 범위 설정
            for axis, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
                axis([c - max_range, c + max_range])

            # 임시 이미지 저장
            temp_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            plt.tight_layout()
            plt.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"Generated {num_frames} frames for view {view_idx+1}")
    
    # 각 view별로 비디오 생성
    cprint("Creating videos...", "blue")
    successful_videos = []
    
    for view_idx in range(len(views)):
        elev, azim = views[view_idx]
        video_name = f"{sequence_name}_original_view{view_idx+1}_elev{elev}_azim{azim}.mp4"
        video_path = os.path.join(save_dir, video_name)
        
        print(f"Creating video for view {view_idx+1}: {video_name}")
        success = create_video_from_images(temp_dirs[view_idx], video_path, fps=fps)
        
        if success:
            successful_videos.append(video_name)
            cprint(f"✓ Successfully created {video_name}", "green")
        else:
            cprint(f"✗ Failed to create {video_name}", "red")
        
        # 임시 파일들 정리 (선택적)
        # import shutil
        # shutil.rmtree(temp_dirs[view_idx])
        
    cprint(f"Video generation completed. Successfully created {len(successful_videos)}/{len(views)} videos.", "blue")
    cprint(f"Results saved in: {save_dir}", "green")
    
    return successful_videos


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="Render Original MANO Keypoints Sequence to Multi-view Videos",
        headless=True,
        custom_parameters=[
            {
                "name": "--data_dir",
                "type": str,
                "default": "/scratch2/jisoo6687/gigahands",
                "help": "Path to GigaHands dataset directory"
            },
            {
                "name": "--seq_id",
                "type": str,
                "default": "p001-folder_001",
                "help": "Sequence ID (e.g., '20aed@0')"
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
                "help": "Hand side (right/left)"
            },
            {
                "name": "--save_dir",
                "type": str,
                "default": "renders_original",
                "help": "Directory to save original sequence videos"
            },
            {
                "name": "--max_frames",
                "type": int,
                "default": None,
                "help": "Maximum number of frames to process (None for all)"
            },
            {
                "name": "--fps",
                "type": int,
                "default": 30,
                "help": "FPS for output videos"
            }
        ],
    )

    # 원본 MANO 키포인트 시퀀스 로드
    cprint(f"Loading original MANO sequence: {_parser.seq_id}", "blue")
    cprint(f"Data directory: {_parser.data_dir}", "white")
    
    joints_sequence, joint_names, sequence_name = load_original_mano_sequence(
        data_dir=_parser.data_dir,
        seq_id=_parser.seq_id,
        side=_parser.side,
        max_frames=_parser.max_frames
    )
    
    cprint(f"Loaded sequence '{sequence_name}' with shape: {joints_sequence.shape}", "green")
    cprint(f"Joint names ({len(joint_names)}): {joint_names}", "white")

    # 다중 뷰 비디오 렌더링
    successful_videos = render_original_sequence_multiview(
        joints_sequence=joints_sequence,
        joint_names=joint_names,
        sequence_name=sequence_name,
        save_dir=_parser.save_dir,
        fps=_parser.fps
    )
    
    if successful_videos:
        cprint(f"\n🎉 Successfully created {len(successful_videos)} videos:", "green")
        for video in successful_videos:
            cprint(f"  - {video}", "white")
    else:
        cprint("❌ No videos were created successfully", "red") 