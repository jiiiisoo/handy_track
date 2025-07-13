import os
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import json
import numpy as np
# import 
import torch
from termcolor import cprint
import pickle
from typing import Dict, List, Tuple, Optional

# Isaac Gym 의존성을 피하기 위해 조건부 임포트
BASE_CLASS = None
try:
    from main.dataset.base import ManipData
    from main.dataset.transform import aa_to_rotmat, rotmat_to_aa
    BASE_CLASS = ManipData
except ImportError:
    # Isaac Gym 관련 의존성이 없는 경우 기본 Dataset 클래스 사용
    from torch.utils.data import Dataset
    BASE_CLASS = Dataset


class GigaHandsDataset(BASE_CLASS):
    """
    GigaHands 데이터셋을 ManipTrans에서 사용할 수 있도록 하는 데이터셋 클래스
    """
    
    def __init__(
        self,
        data_dir: str = None,
        split: str = "all",
        skip: int = 2,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        verbose=True,
        data_indices: List[str] = None,
        max_motion_length: int = 51,
        embodiment: str = "inspire",
        side: str = "right",
        **kwargs,
    ):
        # 부모 클래스 초기화 (ManipData인 경우에만)
        if hasattr(BASE_CLASS, '__name__') and BASE_CLASS.__name__ == 'ManipData':
            try:
                super().__init__(
                    data_dir=data_dir,
                    split=split,
                    skip=skip,
                    device=device,
                    mujoco2gym_transf=mujoco2gym_transf,
                    max_seq_len=max_seq_len,
                    dexhand=dexhand,
                    verbose=verbose,
                    **kwargs,
                )
            except Exception as e:
                if verbose:
                    cprint(f"Warning: Failed to initialize parent class: {e}", "yellow")
        else:
            # 기본 Dataset 클래스인 경우
            try:
                super().__init__()
            except Exception:
                pass  # 초기화 실패 시 무시
        
        # 기본 속성 설정
        self.data_dir = data_dir
        self.device = device
        self.verbose = verbose
        self.max_motion_length = max_motion_length
        self.data_indices = data_indices if data_indices else []
        self.embodiment = embodiment
        self.side = side
        self.mujoco2gym_transf = mujoco2gym_transf
        
        # 기본값들 초기화
        self.data_pathes = []
        self.sequence_info = []
        
        # Load annotations
        self.text_file = os.path.join(data_dir, '../annotations_v2.jsonl')
        
        # Load mean and std for normalization
        self.mean_path = os.path.join(data_dir, '../giga_mean_kp.npy')
        self.std_path = os.path.join(data_dir, '../giga_std_kp.npy')
        
        if os.path.exists(self.mean_path) and os.path.exists(self.std_path):
            self.mean = torch.tensor(np.load(self.mean_path), device=device)
            self.std = torch.tensor(np.load(self.std_path), device=device)
        else:
            if verbose:
                cprint(f"Warning: Mean/std files not found. Using default normalization.", "yellow")
            self.mean = torch.zeros(126, device=device)  # 42*3 for keypoints
            self.std = torch.ones(126, device=device)
        
        # Load data paths based on indices
        self._load_data_paths()
        
        if verbose:
            cprint(f"Loaded GigaHands dataset with {len(self.data_pathes)} sequences", "green")

    def _load_data_paths(self):
        """데이터 경로를 로드합니다."""
        self.data_pathes = []
        self.sequence_info = []
        
        with open(self.text_file, 'r', encoding='utf-8') as file:
            all_sequences = []
            for line in file:
                script_info = json.loads(line)
                all_sequences.append(script_info)
        
        # 특정 indices가 지정된 경우 해당 데이터만 로드
        if self.data_indices:
            if self.verbose:
                cprint(f"Loading data indices: {self.data_indices}", "cyan")
            for data_idx in self.data_indices:
                if '@' in data_idx:
                    seq_id, frame_idx = data_idx.split('@')
                    frame_idx = int(frame_idx)
                else:
                    seq_id = data_idx
                    frame_idx = 0
                
                if self.verbose:
                    cprint(f"Searching for sequence: {seq_id}", "cyan")
                
                # 시퀀스 찾기
                found = False
                for script_info in all_sequences:
                    seq = script_info['sequence'][0] if isinstance(script_info['sequence'], list) else script_info['sequence']
                    # 정확한 매칭 또는 부분 매칭
                    if seq == seq_id or seq.startswith(seq_id) or seq_id in seq:
                        sf, ef = script_info['start_frame_id'], script_info['end_frame_id']
                        scene_name = script_info['scene']
                        script_text = script_info['clarify_annotation']
                        
                        # keypoints 경로 생성
                        v_path = os.path.join(self.data_dir, scene_name, 'keypoints_3d_mano', seq + '.json')
                        
                        if os.path.exists(v_path) and script_text != 'None' and script_text != 'Buggy':
                            self.data_pathes.append(v_path)
                            self.sequence_info.append({
                                'path': v_path,
                                'chosen_frames': (sf, ef),
                                'sequence_id': seq,
                                'scene': scene_name,
                                'caption': script_text,
                                'frame_offset': frame_idx
                            })
                            if self.verbose:
                                cprint(f"Added sequence: {seq} from {scene_name}", "blue")
                            found = True
                            break
                
                if not found and self.verbose:
                    cprint(f"Warning: No sequence found for index: {seq_id}", "yellow")
            
        else:
            # 모든 시퀀스 로드
            for script_info in all_sequences:
                seq = script_info['sequence'][0] if isinstance(script_info['sequence'], list) else script_info['sequence']
                sf, ef = script_info['start_frame_id'], script_info['end_frame_id']
                scene_name = script_info['scene']
                script_text = script_info['clarify_annotation']
                
                v_path = os.path.join(self.data_dir, scene_name, 'keypoints_3d_mano', seq + '.json')
                
                if os.path.exists(v_path) and script_text != 'None' and script_text != 'Buggy':
                    self.data_pathes.append(v_path)
                    self.sequence_info.append({
                        'path': v_path,
                        'chosen_frames': (sf, ef),
                        'sequence_id': seq,
                        'scene': scene_name,
                        'caption': script_text,
                        'frame_offset': 0
                    })

    def __len__(self):
        return max(len(self.data_pathes), 1)  # 최소 1개는 반환
    
    def _load_motion_data(self, idx: int) -> Tuple[torch.Tensor, int]:
        """모션 데이터를 로드하고 정규화합니다."""
        if idx >= len(self.sequence_info):
            # 인덱스가 범위를 벗어나면 더미 데이터 반환
            T = self.max_motion_length
            return torch.zeros(T, 126, device=self.device), T
            
        seq_info = self.sequence_info[idx]
        data_path = seq_info['path']
        
        if not data_path or not os.path.exists(data_path):
            # 파일이 없는 경우 더미 데이터 반환
            T = self.max_motion_length
            return torch.zeros(T, 126, device=self.device), T
        
        try:
            with open(data_path, "r") as f:
                mano_kp = json.load(f)  # [F, 42*3]
        except Exception as e:
            if self.verbose:
                cprint(f"Warning: Failed to load motion data from {data_path}: {e}", "yellow")
            T = self.max_motion_length
            return torch.zeros(T, 126, device=self.device), T
        
        start, end = seq_info['chosen_frames']
        frame_offset = seq_info.get('frame_offset', 0)
        
        # 프레임 범위 조정
        if end == -1:
            motion = np.array(mano_kp)[start:]
        else:
            motion = np.array(mano_kp)[start:end+1]
        
        # 특정 프레임부터 시작하는 경우
        if frame_offset > 0 and frame_offset < len(motion):
            motion = motion[frame_offset:]
        
        motion = torch.tensor(motion, dtype=torch.float32, device=self.device)
        
        # 정규화
        motion = (motion - self.mean) / self.std
        
        m_length = motion.shape[0]
        
        # 길이 제한
        if m_length > self.max_motion_length:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length
        
        # 패딩
        if m_length < self.max_motion_length:
            padding = torch.zeros((self.max_motion_length - m_length, motion.shape[1]), device=self.device)
            motion = torch.cat([motion, padding], dim=0)
        
        return motion, m_length
    
    def _convert_to_manip_format(self, motion: torch.Tensor, seq_info: Dict) -> Dict:
        """GigaHands 모션 데이터를 ManipTrans 형식으로 변환합니다."""
        # motion: [T, 126] -> [T, 42, 3] (42개 키포인트, 각각 x,y,z)
        motion_reshaped = motion.view(-1, 42, 3)
        if self.side == "right" :
            motion_reshaped = motion_reshaped[:,:21,:]
        else : 
            motion_reshaped = motion_reshaped[:,21:,:]
        
        # MANO 키포인트에서 필요한 정보 추출
        # 손목 위치 (첫 번째 키포인트를 손목으로 가정)
        wrist_pos = motion_reshaped[:, 0, :]  # [T, 3]
        
        # 손가락 끝 키포인트들 (MANO 모델 기준)
        # MANO 키포인트 순서에 따라 조정 필요
        tip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
        finger_tips = motion_reshaped[:, tip_indices, :]  # [T, 5, 3]
        
        # 전체 조인트 정보
        # mano_joints = {}
        # joint_names = [
        #     "wrist", "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
        #     "index_mcp", "index_pip", "index_dip", "index_tip",
        #     "middle_mcp", "middle_pip", "middle_dip", "middle_tip", 
        #     "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        #     "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        # ]
        gigahands_mano_joints = {
            "wrist": motion_reshaped[:, 0, :].detach(),           # 0
            "thumb_proximal": motion_reshaped[:, 1, :].detach(),  # 1 (mcp)
            "thumb_intermediate": motion_reshaped[:, 2, :].detach(), # 2 (pip)
            "thumb_distal": motion_reshaped[:, 3, :].detach(),    # 3 (dip)
            "thumb_tip": motion_reshaped[:, 4, :].detach(),       # 4
            "index_proximal": motion_reshaped[:, 5, :].detach(),  # 5 (mcp)
            "index_intermediate": motion_reshaped[:, 6, :].detach(), # 6 (pip)
            "index_distal": motion_reshaped[:, 7, :].detach(),    # 7 (dip)
            "index_tip": motion_reshaped[:, 8, :].detach(),       # 8
            "middle_proximal": motion_reshaped[:, 9, :].detach(), # 9 (mcp)
            "middle_intermediate": motion_reshaped[:, 10, :].detach(), # 10 (pip)
            "middle_distal": motion_reshaped[:, 11, :].detach(),  # 11 (dip)
            "middle_tip": motion_reshaped[:, 12, :].detach(),
            "ring_proximal": motion_reshaped[:, 13, :].detach(),  # 13 (mcp)
            "ring_intermediate": motion_reshaped[:, 14, :].detach(), # 14 (pip)
            "ring_distal": motion_reshaped[:, 15, :].detach(),    # 15 (dip)
            "ring_tip": motion_reshaped[:, 16, :].detach(),       # 16
            "pinky_proximal": motion_reshaped[:, 17, :].detach(), # 17 (mcp)
            "pinky_intermediate": motion_reshaped[:, 18, :].detach(), # 18 (pip)
            "pinky_distal": motion_reshaped[:, 19, :].detach(),   # 19 (dip)
            "pinky_tip": motion_reshaped[:, 20, :].detach()
        }
        # 손목 회전 (더미 값)
        # wrist_rot = torch.zeros(T, 3, device=self.device)
        print(f'motion reshaped: {motion_reshaped.shape}')
        transform_abs = torch.load("/workspace/ManipTrans/transform_abs.pt")
        wrist_pos = motion_reshaped[:, 0, :].detach()
        middle_pos = motion_reshaped[:, 9, :].detach()
        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # ? hack for wrist position
        dexhand = DexHandFactory.create_hand(dexhand_type="inspire", side="right")
        wrist_pos += torch.tensor(dexhand.relative_translation, device=self.device)
        mano_rot_offset = dexhand.relative_rotation
        wrist_rot_matrix = transform_abs[:, 0, :3, :3].detach() @ np.repeat(mano_rot_offset[None], transform_abs.shape[0], axis=0)
        wrist_rot_tensor = torch.tensor(wrist_rot_matrix, dtype=torch.float32)
        wrist_rot = rotmat_to_aa(wrist_rot_tensor).detach()  # [T, 3] angle-axis 형태
        # wrist_rot = torch.zeros(T, 3, device=self.device)
        if self.side == "left":
            for name in gigahands_mano_joints:
                gigahands_mano_joints[name] = gigahands_mano_joints[name].clone()
                gigahands_mano_joints[name][:, 1] *= -1  # Y축 반전
            wrist_pos = wrist_pos.clone()
            wrist_pos[:, 1] *= -1
            wrist_rot = wrist_rot.clone()
        
        data = {
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": gigahands_mano_joints,
            "sequence_info": seq_info,
            "motion_length": motion.shape[0]
        }
        
        return data

    def __getitem__(self, index):
        """데이터셋에서 하나의 시퀀스를 가져옵니다."""
        
        # 문자열 인덱스 처리 (예: "20aed@0")
        if isinstance(index, str):
            if "@" in index:
                seq_id, frame_offset = index.split("@")
                frame_offset = int(frame_offset)
            else:
                seq_id = index
                frame_offset = 0
                
            # 해당 시퀀스 찾기
            for i, seq_info in enumerate(self.sequence_info):
                if seq_id in seq_info['sequence_id'] or seq_info['sequence_id'].startswith(seq_id):
                    # 프레임 오프셋 적용
                    seq_info = seq_info.copy()
                    seq_info['frame_offset'] = frame_offset
                    
                    # 모션 데이터 로드
                    motion, motion_length = self._load_motion_data_with_info(seq_info)
                    
                    # ManipTrans 형식으로 변환
                    data = self._convert_to_manip_format(motion, seq_info)
                    
                    # 처리 및 반환
                    return self._finalize_data(data, i, seq_info)
                    # return data
            
        
        else:
            raise ValueError(f"Unsupported index type: {type(index)}")
    
    def _load_motion_data_with_info(self, seq_info: Dict) -> Tuple[torch.Tensor, int]:
        """시퀀스 정보를 사용하여 모션 데이터를 로드합니다."""
        data_path = seq_info['path']
        
        if not data_path or not os.path.exists(data_path):
            # 파일이 없는 경우 더미 데이터 반환
            T = self.max_motion_length
            return torch.zeros(T, 126, device=self.device), T
        
        try:
            with open(data_path, "r") as f:
                mano_kp = json.load(f)  # [F, 42*3]
        except Exception as e:
            if self.verbose:
                cprint(f"Warning: Failed to load motion data: {e}", "yellow")
            T = self.max_motion_length
            return torch.zeros(T, 126, device=self.device), T
        
        start, end = seq_info['chosen_frames']
        frame_offset = seq_info.get('frame_offset', 0)
        
        # 프레임 범위 조정
        if end == -1:
            motion = np.array(mano_kp)[start:]
        else:
            motion = np.array(mano_kp)[start:end+1]
        
        # 특정 프레임부터 시작하는 경우
        if frame_offset > 0 and frame_offset < len(motion):
            motion = motion[frame_offset:]
        
        motion = torch.tensor(motion, dtype=torch.float32, device=self.device)
        
        # 정규화
        motion = (motion - self.mean) / self.std
        
        m_length = motion.shape[0]
        
        # 길이 제한
        if m_length > self.max_motion_length:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length
        
        # 패딩
        if m_length < self.max_motion_length:
            padding = torch.zeros((self.max_motion_length - m_length, motion.shape[1]), device=self.device)
            motion = torch.cat([motion, padding], dim=0)
        
        return motion, m_length
    
    def _finalize_data(self, data: Dict, idx: int, seq_info: Dict) -> Dict:
        """데이터 처리를 완료하고 반환합니다."""
        # 더미 오브젝트 포인트 클라우드 (실제 객체가 없으므로)
        rs_verts_obj = torch.randn(1000, 3, device=self.device) * 0.1  # 작은 더미 포인트 클라우드
        
        # 기본 처리 적용 (ManipData가 있는 경우에만)
        if hasattr(self, 'process_data'):
            try:
                self.process_data(data, idx, rs_verts_obj)
            except Exception as e:
                if self.verbose:
                    cprint(f"Warning: Failed to process data at index {idx}: {e}", "yellow")
        
        # 리타겟팅된 데이터 로드 시도
        seq_id = seq_info['sequence_id']
        retargeted_data_path = os.path.join(
            self.data_dir, 'retargeted', f"{seq_id}_retargeted.pkl"
        )
        
        if hasattr(self, 'load_retargeted_data'):
            self.load_retargeted_data(data, retargeted_data_path)
        else:
            # 기본 리타겟팅 데이터 생성
            T = data["wrist_pos"].shape[0]
            data.update({
                "opt_wrist_pos": data["wrist_pos"],
                "opt_wrist_rot": data["wrist_rot"],
                "opt_dof_pos": torch.zeros([T, 20], device=self.device),  # 기본 DOF 수
                "opt_wrist_velocity": torch.zeros_like(data["wrist_pos"]),
                "opt_wrist_angular_velocity": torch.zeros_like(data["wrist_rot"]),
                "opt_dof_velocity": torch.zeros([T, 20], device=self.device),
            })
        
        return data

    
    def inv_transform(self, data: torch.Tensor) -> torch.Tensor:
        """정규화를 역변환합니다."""
        return data * self.std + self.mean


def create_gigahands_dataset(
    data_dir: str = None,
    data_indices: List[str] = None,
    device: str = "cuda:0",
    **kwargs
) -> GigaHandsDataset:
    """GigaHands 데이터셋을 생성하는 팩토리 함수"""
    
    # Mujoco to Gym transformation matrix (예시)
    mujoco2gym_transf = torch.eye(4, dtype=torch.float32, device=device)
    
    dataset = GigaHandsDataset(
        data_dir=data_dir,
        data_indices=data_indices,
        device=device,
        mujoco2gym_transf=mujoco2gym_transf,
        **kwargs
    )
    
    return dataset


# 조건부 데이터셋 등록 (Isaac Gym 환경이 있는 경우에만)
try:
    from main.dataset.factory import ManipDataFactory
    ManipDataFactory.register("gigahands", GigaHandsDataset)
except ImportError:
    # Isaac Gym 환경이 없는 경우 등록하지 않음
    pass 