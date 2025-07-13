from isaacgym import gymapi, gymtorch, gymutil
from main.dataset.mano2dexhand import Mano2Dexhand
from main.dataset.mano2dexhand_gigahands import Mano2DexhandGigaHands
from main.dataset.transform import rot6d_to_rotmat, aa_to_rotmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch
import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import subprocess

# ffmpeg-python 올바른 import
# try:
#     # 잘못된 ffmpeg 패키지가 설치된 경우를 대비
#     import sys
#     if 'ffmpeg' in sys.modules:
#         del sys.modules['ffmpeg']
    
#     # 올바른 방식으로 import
#     import ffmpeg
    
#     # 필요한 속성들이 있는지 확인
#     required_attrs = ['input', 'output', 'run', 'probe']
#     missing_attrs = [attr for attr in required_attrs if not hasattr(ffmpeg, attr)]
    
#     if missing_attrs:
#         print(f"ffmpeg module is missing required attributes: {missing_attrs}")
#         print("This suggests wrong package is installed. Please run:")
#         print("pip uninstall ffmpeg ffmpeg-python")
#         print("pip install ffmpeg-python")
#         ffmpeg = None
#     else:
#         print("ffmpeg-python successfully imported with all required attributes")
        
# except ImportError as e:
#     print(f"Failed to import ffmpeg-python: {e}")
#     print("Please install with: pip install ffmpeg-python")
#     ffmpeg = None
# except Exception as e:
#     print(f"Unexpected error importing ffmpeg: {e}")
#     ffmpeg = None
import ffmpeg

# Try to import scipy - install if needed
try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: scipy not available - {e}")
    print("Install with: pip install scipy")
    SCIPY_AVAILABLE = False
    gaussian_filter1d = None
    interp1d = None

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def smooth_temporal_data(opt_wrist_pos_seq, opt_wrist_rot_seq, opt_dof_pos_seq, 
                        sigma=1.0, method='gaussian'):
    """
    Temporal smoothing for hand motion data
    
    Args:
        opt_wrist_pos_seq: List/array of wrist positions [T, 3]
        opt_wrist_rot_seq: List/array of wrist rotations [T, 3] (axis-angle)
        opt_dof_pos_seq: List/array of DOF positions [T, N_dofs]
        sigma: Smoothing parameter (higher = more smoothing)
        method: 'gaussian', 'moving_average', or 'savgol'
    
    Returns:
        Smoothed sequences
    """
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available, falling back to simple moving average")
        method = 'moving_average'
    
    print(f"Applying temporal smoothing with method='{method}', sigma={sigma}")
    
    # Convert to numpy arrays if needed
    if isinstance(opt_wrist_pos_seq, list):
        wrist_pos_array = np.array(opt_wrist_pos_seq)
        wrist_rot_array = np.array(opt_wrist_rot_seq)
        dof_pos_array = np.array(opt_dof_pos_seq)
    else:
        wrist_pos_array = opt_wrist_pos_seq.copy()
        wrist_rot_array = opt_wrist_rot_seq.copy()
        dof_pos_array = opt_dof_pos_seq.copy()
    
    print(f"Input shapes - pos: {wrist_pos_array.shape}, rot: {wrist_rot_array.shape}, dof: {dof_pos_array.shape}")
    
    if len(wrist_pos_array) < 3:
        print("Warning: Sequence too short for meaningful smoothing")
        return opt_wrist_pos_seq, opt_wrist_rot_seq, opt_dof_pos_seq
    
    # Smooth position data
    if method == 'gaussian':
        smooth_wrist_pos = np.zeros_like(wrist_pos_array)
        for i in range(wrist_pos_array.shape[1]):
            smooth_wrist_pos[:, i] = gaussian_filter1d(wrist_pos_array[:, i], sigma=sigma)
        
        smooth_dof_pos = np.zeros_like(dof_pos_array)
        for i in range(dof_pos_array.shape[1]):
            smooth_dof_pos[:, i] = gaussian_filter1d(dof_pos_array[:, i], sigma=sigma)
            
    elif method == 'moving_average':
        window_size = max(3, int(sigma * 3))  # Convert sigma to window size
        if window_size % 2 == 0:
            window_size += 1  # Make odd
        
        def moving_average_1d(data, window):
            return np.convolve(data, np.ones(window)/window, mode='same')
        
        smooth_wrist_pos = np.zeros_like(wrist_pos_array)
        for i in range(wrist_pos_array.shape[1]):
            smooth_wrist_pos[:, i] = moving_average_1d(wrist_pos_array[:, i], window_size)
            
        smooth_dof_pos = np.zeros_like(dof_pos_array)
        for i in range(dof_pos_array.shape[1]):
            smooth_dof_pos[:, i] = moving_average_1d(dof_pos_array[:, i], window_size)
    
    elif method == 'savgol':
        try:
            from scipy.signal import savgol_filter
            window_length = max(5, int(sigma * 3))
            if window_length >= len(wrist_pos_array):
                window_length = len(wrist_pos_array) - 1
            if window_length % 2 == 0:
                window_length += 1
            polyorder = min(3, window_length - 1)
            
            smooth_wrist_pos = np.zeros_like(wrist_pos_array)
            for i in range(wrist_pos_array.shape[1]):
                smooth_wrist_pos[:, i] = savgol_filter(wrist_pos_array[:, i], window_length, polyorder)
                
            smooth_dof_pos = np.zeros_like(dof_pos_array)
            for i in range(dof_pos_array.shape[1]):
                smooth_dof_pos[:, i] = savgol_filter(dof_pos_array[:, i], window_length, polyorder)
        except ImportError:
            print("scipy.signal not available, falling back to moving average")
            window_size = max(3, int(sigma * 3))
            if window_size % 2 == 0:
                window_size += 1
            
            def moving_average_1d(data, window):
                return np.convolve(data, np.ones(window)/window, mode='same')
            
            smooth_wrist_pos = np.zeros_like(wrist_pos_array)
            for i in range(wrist_pos_array.shape[1]):
                smooth_wrist_pos[:, i] = moving_average_1d(wrist_pos_array[:, i], window_size)
                
            smooth_dof_pos = np.zeros_like(dof_pos_array)
            for i in range(dof_pos_array.shape[1]):
                smooth_dof_pos[:, i] = moving_average_1d(dof_pos_array[:, i], window_size)
    
    # Smooth rotation data (axis-angle) - special handling needed
    smooth_wrist_rot = smooth_rotations_axis_angle(wrist_rot_array, sigma=sigma, method=method)
    
    # Convert back to original format
    if isinstance(opt_wrist_pos_seq, list):
        smooth_wrist_pos_seq = [smooth_wrist_pos[i] for i in range(len(smooth_wrist_pos))]
        smooth_wrist_rot_seq = [smooth_wrist_rot[i] for i in range(len(smooth_wrist_rot))]
        smooth_dof_pos_seq = [smooth_dof_pos[i] for i in range(len(smooth_dof_pos))]
    else:
        smooth_wrist_pos_seq = smooth_wrist_pos
        smooth_wrist_rot_seq = smooth_wrist_rot
        smooth_dof_pos_seq = smooth_dof_pos
    
    # Calculate smoothing effect
    pos_diff = np.mean(np.linalg.norm(wrist_pos_array - smooth_wrist_pos, axis=1))
    rot_diff = np.mean(np.linalg.norm(wrist_rot_array - smooth_wrist_rot, axis=1))
    dof_diff = np.mean(np.linalg.norm(dof_pos_array - smooth_dof_pos, axis=1))
    
    print(f"Smoothing effect - pos: {pos_diff:.6f}, rot: {rot_diff:.6f}, dof: {dof_diff:.6f}")
    
    return smooth_wrist_pos_seq, smooth_wrist_rot_seq, smooth_dof_pos_seq


def smooth_rotations_axis_angle(rot_array, sigma=1.0, method='gaussian'):
    """
    Smooth rotation data in axis-angle representation
    """
    if len(rot_array) < 3:
        return rot_array.copy()
    
    # Convert axis-angle to rotation matrices
    from main.dataset.transform import aa_to_rotmat, rotmat_to_aa
    
    rot_matrices = []
    for i in range(len(rot_array)):
        rot_mat = aa_to_rotmat(torch.tensor(rot_array[i]).float())
        rot_matrices.append(rot_mat.numpy())
    rot_matrices = np.array(rot_matrices)
    
    # Smooth each element of rotation matrix
    smooth_rot_matrices = np.zeros_like(rot_matrices)
    
    if method == 'gaussian':
        for i in range(3):
            for j in range(3):
                smooth_rot_matrices[:, i, j] = gaussian_filter1d(rot_matrices[:, i, j], sigma=sigma)
    elif method == 'moving_average':
        window_size = max(3, int(sigma * 3))
        if window_size % 2 == 0:
            window_size += 1
        
        def moving_average_1d(data, window):
            return np.convolve(data, np.ones(window)/window, mode='same')
        
        for i in range(3):
            for j in range(3):
                smooth_rot_matrices[:, i, j] = moving_average_1d(rot_matrices[:, i, j], window_size)
    
    elif method == 'savgol':
        try:
            from scipy.signal import savgol_filter
            window_length = max(5, int(sigma * 3))
            if window_length >= len(rot_array):
                window_length = len(rot_array) - 1
            if window_length % 2 == 0:
                window_length += 1
            polyorder = min(3, window_length - 1)
            
            for i in range(3):
                for j in range(3):
                    smooth_rot_matrices[:, i, j] = savgol_filter(rot_matrices[:, i, j], window_length, polyorder)
        except ImportError:
            print("scipy.signal not available for rotation smoothing, using moving average")
            window_size = max(3, int(sigma * 3))
            if window_size % 2 == 0:
                window_size += 1
            
            def moving_average_1d(data, window):
                return np.convolve(data, np.ones(window)/window, mode='same')
            
            for i in range(3):
                for j in range(3):
                    smooth_rot_matrices[:, i, j] = moving_average_1d(rot_matrices[:, i, j], window_size)
    
    # Re-orthogonalize rotation matrices using SVD
    smooth_rot_matrices_ortho = np.zeros_like(smooth_rot_matrices)
    for i in range(len(smooth_rot_matrices)):
        U, _, Vt = np.linalg.svd(smooth_rot_matrices[i])
        # Ensure proper rotation matrix (det = +1)
        d = np.linalg.det(U @ Vt)
        if d < 0:
            U[:, -1] *= -1
        smooth_rot_matrices_ortho[i] = U @ Vt
    
    # Convert back to axis-angle
    smooth_rot_aa = []
    for i in range(len(smooth_rot_matrices_ortho)):
        rot_aa = rotmat_to_aa(torch.tensor(smooth_rot_matrices_ortho[i]).float())
        smooth_rot_aa.append(rot_aa.numpy())
    
    return np.array(smooth_rot_aa)


def render_multiview_keypoints_sequence(opt_wrist_pos_seq, opt_wrist_rot_seq, opt_dof_pos_seq, 
                                       chain, joint_names, save_dir="renders", prefix="frame", 
                                       fps=30):
    """
    전체 시퀀스에 대해 여러 뷰에서 로봇 손 keypoint를 렌더링하여 비디오 생성
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda:0"
    num_frames = len(opt_wrist_pos_seq)
    
    # 다양한 시점 설정: (elev, azim)
    # views = [
    #     (30, 45),
    #     (30, -45),
    #     (0, 90),
    #     (90, 0),
    #     (45, 180),
    #     (20, 270)
    # ]
    views = [
        (30, 45),
        # (30, -45)
        # (0, 90),
        (90, 0)
        # (45, 180),
        # (20, 270)
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

    # 각 view별로 비디오 writer 생성
    video_writers = {}
    temp_dirs = {}
    
    print(f"Processing {num_frames} frames for {len(views)} views...")
    
    # 전체 시퀀스의 범위 계산 (일관된 축 범위를 위해)
    all_positions = []
    if os.path.exists(os.path.join(save_dir, f"temp_view1")):
        for frame_idx in tqdm(range(num_frames), desc="Calculating joint positions"):
            opt_wrist_pos = torch.tensor(opt_wrist_pos_seq[frame_idx]).to(device)
            opt_wrist_rot = torch.tensor(opt_wrist_rot_seq[frame_idx]).to(device)
            opt_dof_pos = torch.tensor(opt_dof_pos_seq[frame_idx]).to(device)
            
            # 디버깅: 첫 번째와 마지막 프레임의 DOF 값 출력
            if frame_idx == 0 or frame_idx == num_frames - 1:
                print(f"Frame {frame_idx} DOF values (first 10): {opt_dof_pos[:10].cpu().numpy()}")
            
            # rotation 처리
            if opt_wrist_rot.shape[-1] == 6:
                wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot.unsqueeze(0))[0]
            else:
                wrist_rotmat = aa_to_rotmat(opt_wrist_rot.unsqueeze(0))[0]

            # FK 계산
            with torch.no_grad():
                fk_result = chain.forward_kinematics(opt_dof_pos.unsqueeze(0))
                joint_positions_local = torch.stack(
                    [fk_result[k].get_matrix()[0, :3, 3] for k in joint_names], dim=0
                )
                joint_positions_local = joint_positions_local.to(device)
                joint_positions = (wrist_rotmat @ joint_positions_local.T).T + opt_wrist_pos
                
                # 디버깅: 첫 번째 프레임의 경우 local과 world joint positions 비교
                if frame_idx == 0:
                    print(f"Local joint positions (first 5): {joint_positions_local[:5].cpu().numpy()}")
                    print(f"World joint positions (first 5): {joint_positions[:5].cpu().numpy()}")
                    print(f"Wrist position: {opt_wrist_pos.cpu().numpy()}")
                
            all_positions.append(joint_positions.cpu().numpy())
        
        all_positions = np.stack(all_positions)
        
        # 전체 범위 계산
        max_range = np.ptp(all_positions.reshape(-1, 3), axis=0).max() * 0.6
        center = all_positions.reshape(-1, 3).mean(axis=0)

    # 각 view별로 프레임 렌더링
    for view_idx, (elev, azim) in enumerate(views):
        temp_dir = os.path.join(save_dir, f"temp_view{view_idx+1}")
        # 기존 임시 디렉토리 정리
        if os.path.exists(temp_dir):
            temp_dirs[view_idx] = temp_dir
            continue
        os.makedirs(temp_dir, exist_ok=True)
        temp_dirs[view_idx] = temp_dir
        
        print(f"Rendering view {view_idx+1}/{len(views)} (elev={elev}, azim={azim})")
        
        frame_count = 0
        print(f"Rendering {num_frames} frames for view {view_idx+1}")
        for frame_idx in tqdm(range(num_frames), desc=f"View {view_idx+1}"):
            joint_positions_np = all_positions[frame_idx]
            
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
            ax.set_title(f"Frame {frame_idx+1}/{num_frames} - View {view_idx+1}")
            ax.view_init(elev=elev, azim=azim)

            # 일관된 축 범위 설정
            for axis, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
                axis([c - max_range, c + max_range])

            # 임시 이미지 저장
            temp_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            plt.tight_layout()
            plt.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close()
            frame_count += 1
        
        print(f"Generated {frame_count} frames for view {view_idx+1}")
        
        # 실제 생성된 파일 수 확인
        generated_files = [f for f in os.listdir(temp_dir) if f.endswith('.png')]
        print(f"Files in temp_dir: {len(generated_files)}")
        if len(generated_files) != num_frames:
            print(f"Warning: Expected {num_frames} files, but found {len(generated_files)}")
            print(f"Generated files: {sorted(generated_files)[:5]}...")  # 처음 5개만 표시
    
    # 각 view별로 비디오 생성
    print("Creating videos...")
    for view_idx in range(len(views)):
        print(f"Creating video for view {view_idx+1}...")
        success = create_video_from_images(temp_dirs[view_idx], 
                        os.path.join(save_dir, f"view{view_idx+1}_animation.mp4"), 
                        fps=fps)
        
        if success:
            print(f"✓ Successfully created view{view_idx+1}_animation.mp4")
        else:
            print(f"✗ Failed to create view{view_idx+1}_animation.mp4")
        
        # # 임시 파일들 정리
        # import shutil
        # shutil.rmtree(temp_dirs[view_idx])
        
    print(f"Video generation completed. Check {save_dir} for results.")


def create_video_from_images(image_dir, output_path, fps=30):
    """
    이미지 디렉토리에서 비디오 생성 (ffmpeg-python 전용)
    """
    # if ffmpeg is None:
    #     print("ERROR: ffmpeg-python is not properly installed!")
    #     print("Please run the following commands:")
    #     print("pip uninstall ffmpeg ffmpeg-python")
    #     print("pip install ffmpeg-python")
    #     return False
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return False
    
    print(f"Creating video: {output_path}")
    print(f"Found {len(image_files)} images")
    print(f"First few files: {image_files[:5]}")
    print(f"Last few files: {image_files[-5:]}")
    
    # 연속된 프레임인지 확인하고 필요시 rename
    expected_pattern = True
    for i, filename in enumerate(image_files):
        expected_name = f"frame_{i:04d}.png"
        if filename != expected_name:
            print(f"Warning: Non-sequential frame found: {filename}, expected: {expected_name}")
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
    print(f"Input pattern: {input_pattern}")
    
    # 첫 번째와 마지막 프레임 확인
    first_frame = os.path.join(image_dir, 'frame_0000.png')
    last_frame = os.path.join(image_dir, f'frame_{len(image_files)-1:04d}.png')
    print(f"First frame exists: {os.path.exists(first_frame)}")
    print(f"Last frame exists: {os.path.exists(last_frame)}")
    
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
        loglevel='verbose'  # 더 자세한 로그
    )
    
    # 기존 파일 덮어쓰기 허용하고 실행
    ffmpeg.run(output_stream, overwrite_output=True, quiet=False, capture_stdout=False, capture_stderr=False)
    
    print(f"Video created successfully: {output_path}")
    
    # 생성된 비디오 파일 확인
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Video file size: {file_size / (1024*1024):.2f} MB")
        
        # ffprobe로 비디오 정보 확인
        try:
            probe_info = ffmpeg.probe(output_path)
            video_stream = next((stream for stream in probe_info['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                duration = float(video_stream.get('duration', 0))
                frame_count = int(video_stream.get('nb_frames', 0))
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                print(f"Video info - Duration: {duration:.2f}s, Frames: {frame_count}, Resolution: {width}x{height}")
                
                if frame_count == 1:
                    print("WARNING: Video only has 1 frame! Check input images.")
                elif frame_count != len(image_files):
                    print(f"WARNING: Expected {len(image_files)} frames, but video has {frame_count} frames")
                else:
                    print("✓ Video frame count matches input images")
                    
        except Exception as probe_e:
            print(f"Could not probe video (but video was created): {probe_e}")
        
    return True


def render_multiview_keypoints(opt_wrist_pos, opt_wrist_rot, opt_dof_pos, chain, joint_names, save_dir="renders", prefix="frame0"):
    """
    단일 프레임 렌더링 (기존 함수 유지)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda:0"
    opt_wrist_pos = opt_wrist_pos.to(device)
    opt_wrist_rot = opt_wrist_rot.to(device)
    opt_dof_pos = opt_dof_pos.to(device)

    # rotation 처리
    if opt_wrist_rot.shape[-1] == 6:
        wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot.unsqueeze(0))[0]
    else:
        wrist_rotmat = aa_to_rotmat(opt_wrist_rot.unsqueeze(0))[0]

    # FK 계산
    with torch.no_grad():
        fk_result = chain.forward_kinematics(opt_dof_pos.unsqueeze(0))
        joint_positions = torch.stack(
            [fk_result[k].get_matrix()[0, :3, 3] for k in joint_names], dim=0
        )
        joint_positions = joint_positions.to(device)
        joint_positions = (wrist_rotmat @ joint_positions.T).T + opt_wrist_pos

    joint_positions_np = joint_positions.cpu().numpy()

    # 다양한 시점 설정
    # views = [
    #     (30, 45),
    #     (30, -45),
    #     (0, 90),
    #     (90, 0),
    #     (45, 180),
    #     (20, 270)
    # ]
    views = [
        (30, 45),
        (30, -45)]
        # (0, 90),
        # (90, 0),
        # (45, 180),
        # (20, 270)

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

    for i, (elev, azim) in enumerate(views):
        if os.path.exists(os.path.join(save_dir, f"{prefix}_view{i+1}.png")):
            continue
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
        
        # 범례 추가
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"View {i+1} (elev={elev}, azim={azim})")
        ax.view_init(elev=elev, azim=azim)

        # 렌더링 범위 자동 설정
        max_range = np.ptp(joint_positions_np, axis=0).max() * 0.6
        center = joint_positions_np.mean(axis=0)
        for axis, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            axis([c - max_range, c + max_range])

        save_path = os.path.join(save_dir, f"{prefix}_view{i+1}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved view {i+1} to {save_path}")


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="GigaHands MANO to Dexhand",
        headless=True,
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 7000,
            },
            {
                "name": "--data_idx",
                "type": str,
                "default": "20aed@0",
                "help": "Single sequence ID (e.g., '20aed@0') or 'all' for batch processing"
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "inspire",
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
            },
            {
                "name": "--data_dir",
                "type": str,
                "default": "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands",
                "help": "Path to GigaHands dataset directory containing annotations_v2.jsonl and handpose/"
            },
            {
                "name": "--save_dir",
                "type": str,
                "default": "/workspace/ManipTrans/visualize_results",
                "help": "Directory to save retargeted data"
            },
            {
                "name": "--skip_existing",
                "action": "store_true",
                "help": "Skip sequences that already have retargeted files"
            },
            {
                "name": "--max_sequences",
                "type": int,
                "default": -1,
                "help": "Maximum number of sequences to process (-1 for all)"
            },
            {"name": "--num_envs",
                "type": int,
                "default": 1,
                "help": "Number of environments to render"
            },
            {"name": "--fps",
                "type": int,
                "default": 30,
                "help": "FPS for output videos"
            },
            {"name": "--retargeted_data_path",
                "type": str,
                "default": "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/retargeted/mano2inspire_rh/p001-folder/keypoints_3d_mano/000_retargeted.pkl",
                "help": "Path to retargeted data pickle file"
            },
            {"name": "--smooth",
                "action": "store_true",
                "help": "Apply temporal smoothing to the motion data"
            },
            {"name": "--smooth_sigma",
                "type": float,
                "default": 1.5,
                "help": "Smoothing strength (higher = more smoothing)"
            },
            {"name": "--smooth_method",
                "type": str,
                "default": "gaussian",
                "choices": ["gaussian", "moving_average", "savgol"],
                "help": "Smoothing method to use"
            }
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)
    mano2inspire = Mano2DexhandGigaHands(_parser, dexhand)
    
    # 전체 시퀀스 데이터 로드
    import datetime
    file_stat = os.stat(_parser.retargeted_data_path)
    file_time = datetime.datetime.fromtimestamp(file_stat.st_mtime)
    print(f"Loading data from: {_parser.retargeted_data_path}")
    print(f"File last modified: {file_time}")
    to_dump = pickle.load(open(_parser.retargeted_data_path, "rb"))
    
    # 데이터 구조 확인
    print("Data keys:", list(to_dump.keys()))
    for key in ["opt_wrist_pos", "opt_wrist_rot", "opt_dof_pos"]:
        if key in to_dump:
            data = to_dump[key]
            if isinstance(data, (list, tuple)):
                print(f"{key}: type={type(data)}, length={len(data)}")
                if len(data) > 0:
                    print(f"  First element shape: {np.array(data[0]).shape if hasattr(data[0], 'shape') or isinstance(data[0], (list, tuple, np.ndarray)) else type(data[0])}")
            elif isinstance(data, np.ndarray):
                print(f"{key}: type={type(data)}, shape={data.shape}")
            elif hasattr(data, 'shape'):
                print(f"{key}: type={type(data)}, shape={data.shape}")
            else:
                print(f"{key}: type={type(data)}")
    
    # 데이터 형태에 따라 처리
    opt_wrist_pos_seq = to_dump["opt_wrist_pos"]
    opt_wrist_rot_seq = to_dump["opt_wrist_rot"]
    opt_dof_pos_seq = to_dump["opt_dof_pos"]
    
    # 디버깅: DOF 값들의 변화량 확인
    if isinstance(opt_dof_pos_seq, np.ndarray):
        print(f"DOF data shape: {opt_dof_pos_seq.shape}")
        if len(opt_dof_pos_seq.shape) > 1:
            dof_min = np.min(opt_dof_pos_seq, axis=0)
            dof_max = np.max(opt_dof_pos_seq, axis=0)
            dof_range = dof_max - dof_min
            print(f"DOF ranges (first 10): {dof_range[:10]}")
            print(f"DOF min values (first 10): {dof_min[:10]}")
            print(f"DOF max values (first 10): {dof_max[:10]}")
            print(f"Non-zero DOF ranges: {np.sum(dof_range > 1e-6)}/{len(dof_range)}")
        else:
            print(f"Single DOF vector: {opt_dof_pos_seq}")
    else:
        print(f"DOF data type: {type(opt_dof_pos_seq)}")
        if len(opt_dof_pos_seq) > 1:
            dof_array = np.array(opt_dof_pos_seq)
            dof_min = np.min(dof_array, axis=0)
            dof_max = np.max(dof_array, axis=0)
            dof_range = dof_max - dof_min
            print(f"DOF ranges (first 10): {dof_range[:10]}")
            print(f"Non-zero DOF ranges: {np.sum(dof_range > 1e-6)}/{len(dof_range)}")
    
    # numpy array인 경우 리스트로 변환
    if isinstance(opt_wrist_pos_seq, np.ndarray):
        if len(opt_wrist_pos_seq.shape) == 1:
            # 단일 프레임인 경우
            print("Warning: Only single frame detected!")
            opt_wrist_pos_seq = [opt_wrist_pos_seq]
            opt_wrist_rot_seq = [opt_wrist_rot_seq]
            opt_dof_pos_seq = [opt_dof_pos_seq]
        else:
            # 다중 프레임인 경우 각 프레임으로 분할
            opt_wrist_pos_seq = [opt_wrist_pos_seq[i] for i in range(opt_wrist_pos_seq.shape[0])]
            opt_wrist_rot_seq = [opt_wrist_rot_seq[i] for i in range(opt_wrist_rot_seq.shape[0])]
            opt_dof_pos_seq = [opt_dof_pos_seq[i] for i in range(opt_dof_pos_seq.shape[0])]
    
    print(f"Final sequence lengths:")
    print(f"  opt_wrist_pos_seq: {len(opt_wrist_pos_seq)}")
    print(f"  opt_wrist_rot_seq: {len(opt_wrist_rot_seq)}")
    print(f"  opt_dof_pos_seq: {len(opt_dof_pos_seq)}")
    
    if len(opt_wrist_pos_seq) == 1:
        print("Warning: Only 1 frame available. Creating a short video with repeated frames.")
        # 단일 프레임을 여러 번 복사하여 짧은 비디오 생성
        repeat_count = 30  # 1초 분량
        opt_wrist_pos_seq = opt_wrist_pos_seq * repeat_count
        opt_wrist_rot_seq = opt_wrist_rot_seq * repeat_count
        opt_dof_pos_seq = opt_dof_pos_seq * repeat_count
        print(f"Extended to {len(opt_wrist_pos_seq)} frames")
    
    # Apply temporal smoothing if requested
    if _parser.smooth and len(opt_wrist_pos_seq) > 3:
        print(f"\nApplying temporal smoothing...")
        print(f"Method: {_parser.smooth_method}, Sigma: {_parser.smooth_sigma}")
        opt_wrist_pos_seq, opt_wrist_rot_seq, opt_dof_pos_seq = smooth_temporal_data(
            opt_wrist_pos_seq=opt_wrist_pos_seq,
            opt_wrist_rot_seq=opt_wrist_rot_seq,
            opt_dof_pos_seq=opt_dof_pos_seq,
            sigma=_parser.smooth_sigma,
            method=_parser.smooth_method
        )
        print("Temporal smoothing completed.")
    elif _parser.smooth and len(opt_wrist_pos_seq) <= 3:
        print("Warning: Sequence too short for smoothing (need > 3 frames)")
    
    # 전체 프레임에 대해 비디오 생성
    # Create filename prefix with smoothing info
    prefix = "retargeted"
    save_subdir = "sequence_videos"
    if _parser.smooth:
        prefix += f"_smooth_{_parser.smooth_method}_sigma{_parser.smooth_sigma}"
        save_subdir += "_smooth"
    
    render_multiview_keypoints_sequence(
        opt_wrist_pos_seq=opt_wrist_pos_seq,
        opt_wrist_rot_seq=opt_wrist_rot_seq,
        opt_dof_pos_seq=opt_dof_pos_seq,
        chain=mano2inspire.chain,
        joint_names=dexhand.body_names,
        save_dir=f"renders/{save_subdir}",
        prefix=prefix,
        fps=_parser.fps
    )
