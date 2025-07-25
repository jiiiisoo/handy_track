import os
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
import json
import numpy as np
# import 
import torch
from termcolor import cprint
import pickle
from typing import Dict, List, Tuple, Optional

from main.dataset.base import ManipData
from main.dataset.transform import aa_to_rotmat, rotmat_to_aa

from manopth.manolayer import ManoLayer


class GigaHandsDataset(ManipData):
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
        verbose=False,
        data_indices: List[str] = None,
        max_motion_length: int = 100,
        embodiment: str = "inspire",
        side: str = "right",
        **kwargs,
    ):
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
        
        # 기본 속성 설정
        self.data_dir = data_dir
        self.device = device
        self.verbose = verbose
        self.max_motion_length = max_motion_length
        self.data_indices = data_indices if data_indices else []
        self.embodiment = embodiment
        self.side = side
        # Ensure mujoco2gym_transf is on the correct device
        if mujoco2gym_transf is None:
            self.mujoco2gym_transf = torch.eye(4, dtype=torch.float32, device=device)
        else:
            self.mujoco2gym_transf = mujoco2gym_transf.to(device) if hasattr(mujoco2gym_transf, 'to') else mujoco2gym_transf
        
        # 기본값들 초기화
        self.data_pathes = []
        self.sequence_info = []
        
        # Load annotations
        self.json_file = os.path.join(data_dir, '../annotations_v2.jsonl')
        
        # # Load mean and std for normalization
        # self.mean_path = os.path.join(data_dir, '../giga_mean_kp.npy')
        # self.std_path = os.path.join(data_dir, '../giga_std_kp.npy')

        # self.mean = torch.tensor(np.load(self.mean_path), device=device)
        # self.std = torch.tensor(np.load(self.std_path), device=device)
        
        # Load data paths based on indices
        self._load_data_paths()
        
        if verbose:
            cprint(f"Loaded GigaHands dataset with {len(self.data_pathes)} sequences", "green")

    def _load_data_paths(self):
        """데이터 경로를 로드합니다."""
        self.data_pathes = []
        self.sequence_info = []

                # JSONL 형식으로 로드 (각 라인이 별도의 JSON 객체)
        with open(self.json_file, 'r', encoding='utf-8') as file:
            all_sequences = []
            # data_indices가 주어진 경우, 해당 시퀀스들을 찾기 위해 필요한 만큼 로드
            if self.data_indices:
                needed_sequences = set()
                for data_idx in self.data_indices:
                    if '@' in data_idx:
                        seq_id, _ = data_idx.split('@')
                    else:
                        seq_id = data_idx
                    needed_sequences.add(seq_id)
                
                found_sequences = set()
                for i, line in enumerate(file):
                    line = line.strip()
                    if line:  # 빈 라인 스킵
                        script_info = json.loads(line)
                        all_sequences.append(script_info)
                        
                        # 현재 시퀀스가 필요한 시퀀스 중 하나인지 확인
                        current_seq_id = script_info.get('sequence_id', '')
                        if current_seq_id in needed_sequences:
                            found_sequences.add(current_seq_id)
                        
                        # 모든 필요한 시퀀스를 찾았으면 로딩 중단
                        if len(found_sequences) >= len(needed_sequences):
                            break
            else:
                for i, line in enumerate(file):
                    line = line.strip()
                    if line:  # 빈 라인 스킵
                        script_info = json.loads(line)
                        all_sequences.append(script_info)

        if self.verbose:
            cprint(f"Loaded annotations as JSONL format: {len(all_sequences)} sequences", "green")
        
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
                
                # 시퀀스 찾기 - 새로운 형식 지원
                found = False
                
                # scene_sequence 형태인지 확인 (예: "p019-makeup_063")
                if '_' in seq_id and not '/' in seq_id:
                    target_scene, target_seq = seq_id.rsplit('_', 1)  # 마지막 '_'로 분리
                    
                    for script_info in all_sequences:
                        scene = script_info['scene']
                        seq = script_info['sequence'][0] if isinstance(script_info['sequence'], list) else script_info['sequence']
                        
                        # scene과 sequence가 모두 일치하는지 확인
                        if scene == target_scene and seq == target_seq:
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
                                break  # 첫 번째 매칭되는 것만 사용
                else:
                    # 기존 방식 (하위 호환성)
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
        
        with open(data_path, "r") as f:
            mano_kp = json.load(f)  # [F, 42*3]

        # params_path: .../keypoints_3d_mano/000.json -> .../params/000.json
        params_path = data_path.replace("keypoints_3d_mano", "params")
        with open(params_path, "r") as f:
            if self.side == "right":
                mano_motion = json.load(f)['right']['poses']
            else:
                mano_motion = json.load(f)['left']['poses']
        
        start, end = seq_info['chosen_frames']
        frame_offset = seq_info.get('frame_offset', 0)
        
        # 프레임 범위 조정
        if end == -1:
            motion = np.array(mano_kp)[start:]
            mano_motion = np.array(mano_motion)[start:]
        else:
            motion = np.array(mano_kp)[start:end+1]
            mano_motion = np.array(mano_motion)[start:end+1]
        
        # 특정 프레임부터 시작하는 경우
        if frame_offset > 0 and frame_offset < len(motion):
            motion = motion[frame_offset:]
        
        motion = torch.tensor(motion, dtype=torch.float32, device=self.device)
        mano_motion = torch.tensor(mano_motion, dtype=torch.float32)
        
        # 정규화
        # motion = (motion - self.mean) / self.std
        
        m_length = motion.shape[0]
        
        # 길이 제한
        if m_length > self.max_motion_length:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length
        
        # 패딩
        if m_length < self.max_motion_length:
            padding = torch.zeros((self.max_motion_length - m_length, motion.shape[1]), device=self.device)
            motion = torch.cat([motion, padding], dim=0)
        
        return motion, m_length, mano_motion
    
    def _convert_to_manip_format(self, motion: torch.Tensor, mano_motion: torch.Tensor, seq_info: Dict) -> Dict:
        """GigaHands 모션 데이터를 ManipTrans 형식으로 변환합니다."""
        # motion: [T, 126] -> [T, 42, 3] (42개 키포인트, 각각 x,y,z)
        motion_reshaped = motion.view(-1, 42, 3).to(self.device)
        if self.side == "right" :
            motion_reshaped = motion_reshaped[:,21:,:]
        else : 
            motion_reshaped = motion_reshaped[:,:21,:]
        
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
        # 손목 회전 계산
        mano_layer = ManoLayer(
            mano_root='/workspace/manopth/mano/models', 
            use_pca=True, 
            ncomps=6, 
            flat_hand_mean=True
        )
        with torch.no_grad():
            hand_verts, _, transform_abs = mano_layer(mano_motion, torch.ones(mano_motion.shape[0], 10))
        # hand_verts, hand_faces = mano_layer(
        #     pose=motion_reshaped[:, 3:66, :],
        #     betas=motion_reshaped[:, 66:66+10, :],
        #     hand_type="right",
        # )
        # print(f'hand_verts: {hand_verts.shape}')
        # transform_abs = torch.load("/scratch2/jisoo6687/handy_track/transform_abs.pt")
        ###############oak dataset
        # transform_abs = torch.ones(motion_reshaped.shape[0], 16, 4, 4)

        wrist_pos = motion_reshaped[:, 0, :].detach()
        middle_pos = motion_reshaped[:, 9, :].detach()
        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # ? hack for wrist position
        dexhand = DexHandFactory.create_hand(dexhand_type="inspire", side="right")
        wrist_pos += torch.tensor(dexhand.relative_translation, device=self.device)
        mano_rot_offset = dexhand.relative_rotation
        wrist_rot = transform_abs[:, 0, :3, :3].detach() @ np.repeat(mano_rot_offset[None], transform_abs.shape[0], axis=0)
        wrist_rot = wrist_rot.to(self.device)        
        ###############oak dataset

        # hand_rot = mano_motion[:, :3].to(self.device) # global wrist rotation
        # wrist_pos = motion_reshaped[:, 0, :].detach()
        # middle_pos = motion_reshaped[:, 9, :].detach()
        # wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # ? hack for wrist position
        # inspire_rot_offset = self.dexhand.relative_rotation
        # wrist_rot = aa_to_rotmat(hand_rot) @ torch.tensor(
        #     np.repeat(inspire_rot_offset[None], motion_reshaped.shape[0], axis=0), device=self.device
        # )

        # Object trajectory 계산: 손가락 팁들의 중심점
        finger_tips = [
            gigahands_mano_joints["thumb_tip"],
            gigahands_mano_joints["index_tip"], 
            gigahands_mano_joints["middle_tip"],
            gigahands_mano_joints["ring_tip"],
            gigahands_mano_joints["pinky_tip"]
        ]
        
        # 손가락 팁들의 평균 위치를 object center로 계산
        finger_tips_stack = torch.stack(finger_tips, dim=1)  # [T, 5, 3]
        obj_center_pos = torch.mean(finger_tips_stack, dim=1)  # [T, 3]
        
        # Object trajectory를 4x4 transformation matrix 형태로 생성
        T = obj_center_pos.shape[0]
        obj_trajectory = torch.eye(4, device=self.device).unsqueeze(0).repeat(T, 1, 1)  # [T, 4, 4]
        obj_trajectory[:, :3, 3] = obj_center_pos  # position 설정
        
        # 더미 object mesh (작은 구형태)
        obj_id = f"dummy_object_{seq_info['sequence_id']}"
        obj_mesh_path = f"dummy_path_{seq_info['sequence_id']}.obj"
        
        if self.side == "left":
            for name in gigahands_mano_joints:
                gigahands_mano_joints[name] = gigahands_mano_joints[name].clone()
                gigahands_mano_joints[name][:, 1] *= -1  # Y축 반전
            wrist_pos = wrist_pos.clone()
            wrist_pos[:, 1] *= -1
            wrist_rot = wrist_rot.clone()
            # Object trajectory도 Y축 반전
            obj_trajectory[:, :3, 3][:, 1] *= -1
        
        data = {
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": gigahands_mano_joints,
            "obj_id": obj_id,
            "obj_mesh_path": obj_mesh_path,
            "obj_trajectory": obj_trajectory,
            "sequence_info": seq_info,
            "motion_length": motion.shape[0]
        }
        
        return data

    def __getitem__(self, index):
        """데이터셋에서 하나의 시퀀스를 가져옵니다."""
        
        # 데이터 변수 초기화
        data = None
        
        # 문자열 인덱스 처리 (예: "20aed@0" 또는 "p019-makeup_063")
        if isinstance(index, str):
            if "@" in index:
                seq_id, frame_offset = index.split("@")
                frame_offset = int(frame_offset)
            else:
                seq_id = index
                frame_offset = None  # frame_offset이 지정되지 않으면 None으로 설정
            
            # scene_sequence 형식 처리 (예: "p019-makeup_063")
            if "_" in seq_id:
                target_scene, target_seq = seq_id.rsplit("_", 1)
                
                # 해당 시퀀스 찾기 - scene과 sequence 모두 일치하는지 확인
                for i, seq_info in enumerate(self.sequence_info):
                    if seq_info['scene'] == target_scene and seq_info['sequence_id'] == target_seq:
                        # 프레임 오프셋 적용
                        seq_info = seq_info.copy()
                        seq_info['frame_offset'] = frame_offset
                        
                        # 모션 데이터 로드
                        motion, motion_length, mano_motion = self._load_motion_data_with_info(seq_info)
                        
                        # ManipTrans 형식으로 변환
                        data = self._convert_to_manip_format(motion, mano_motion, seq_info)
                        
                        # 처리 및 반환
                        self._finalize_data(data, i, seq_info)
                        return data  # 찾으면 즉시 반환
            else:
                # 기존 방식 (하위 호환성)
                for i, seq_info in enumerate(self.sequence_info):
                    if seq_id in seq_info['sequence_id'] or seq_info['sequence_id'].startswith(seq_id):
                        # 프레임 오프셋 적용
                        seq_info = seq_info.copy()
                        seq_info['frame_offset'] = frame_offset
                        
                        # 모션 데이터 로드
                        motion, motion_length, mano_motion = self._load_motion_data_with_info(seq_info)
                        
                        # ManipTrans 형식으로 변환
                        data = self._convert_to_manip_format(motion, mano_motion, seq_info)
                        
                        # 처리 및 반환
                        self._finalize_data(data, i, seq_info)
                        return data  # 찾으면 즉시 반환
            
            # 시퀀스를 찾지 못한 경우 예외 발생시켜 건너뛰기
            if data is None:
                if self.verbose:
                    cprint(f"Warning: Sequence '{seq_id}' not found, skipping...", "yellow")
                raise KeyError(f"Sequence '{seq_id}' not found in dataset")
                    
        elif isinstance(index, int):
            # 정수 인덱스 처리
            if 0 <= index < len(self.sequence_info):
                seq_info = self.sequence_info[index]
                motion, motion_length, mano_motion = self._load_motion_data_with_info(seq_info)
                data = self._convert_to_manip_format(motion, mano_motion, seq_info)
                self._finalize_data(data, index, seq_info)
            else:
                if self.verbose:
                    cprint(f"Warning: Index {index} out of range, skipping...", "yellow")
                raise IndexError(f"Index {index} out of range for dataset with {len(self.sequence_info)} sequences")
        else:
            raise ValueError(f"Unsupported index type: {type(index)}")
            
        return data


    def _load_motion_data_with_info(self, seq_info: Dict) -> Tuple[torch.Tensor, int]:
        """시퀀스 정보를 사용하여 모션 데이터를 로드합니다."""
        data_path = seq_info['path']
        params_path = data_path.replace("keypoints_3d_mano", "params")
        with open(data_path, "r") as f:
            mano_kp = json.load(f)  # [F, 42*3]

        with open(params_path, "r") as f:
            if self.side == "right":
                mano_motion = json.load(f)['right']['poses']
                dexhand = DexHandFactory.create_hand(dexhand_type="inspire", side="right")
            else:
                mano_motion = json.load(f)['left']['poses']
                dexhand = DexHandFactory.create_hand(dexhand_type="inspire", side="left")
        
        start, end = seq_info['chosen_frames']
        frame_offset = seq_info.get('frame_offset', None)
        
        # 프레임 범위 조정
        if end == -1:
            motion = np.array(mano_kp)[start:]
            mano_motion = np.array(mano_motion)[start:]
        else:
            motion = np.array(mano_kp)[start:end+1]
            mano_motion = np.array(mano_motion)[start:end+1]
        
        original_length = len(motion)
        
        # frame_offset이 지정된 경우: 해당 지점부터 시작
        if frame_offset is not None and frame_offset > 0:
            if frame_offset < original_length:
                motion = motion[frame_offset:]
        
        # frame_offset이 지정되지 않은 경우: 랜덤하게 시작점 선택
        elif frame_offset is None:
            if original_length > self.max_motion_length:
                # 시퀀스가 max_motion_length보다 길면 랜덤한 시작점에서 max_motion_length만큼 추출
                max_start_idx = original_length - self.max_motion_length
                random_start = np.random.randint(0, max_start_idx + 1)
                motion = motion[random_start:random_start + self.max_motion_length]
                mano_motion = mano_motion[random_start:random_start + self.max_motion_length]
                if self.verbose:
                    cprint(f"Random sampling: start={random_start}, length={self.max_motion_length}, total={original_length}", "cyan")
            else:
                # 시퀀스가 max_motion_length보다 작거나 같으면 전체 시퀀스 사용
                if self.verbose:
                    cprint(f"Using full sequence: length={original_length}, max={self.max_motion_length}", "cyan")
        
        motion = torch.tensor(motion, dtype=torch.float32, device=self.device)
        mano_motion = torch.tensor(mano_motion, dtype=torch.float32)
        
        # 정규화
        # motion = (motion - self.mean) / self.std
        
        m_length = motion.shape[0]
        
        # 길이 제한 (frame_offset=None이고 길이가 max보다 큰 경우는 이미 위에서 처리됨)
        if m_length > self.max_motion_length:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length
        
        # 패딩 (시퀀스가 max_motion_length보다 짧은 경우)
        if m_length < self.max_motion_length:
            padding = torch.zeros((self.max_motion_length - m_length, motion.shape[1]), device=self.device)
            motion = torch.cat([motion, padding], dim=0)
        
        return motion, m_length, mano_motion
    
    def _finalize_data(self, data: Dict, idx: int, seq_info: Dict) -> Dict:
        """데이터 처리를 완료하고 반환합니다."""
        # Object center 기반으로 더미 object point cloud 생성
        obj_center_pos = data["obj_trajectory"][:, :3, 3]  # [T, 3]
        mean_center = torch.mean(obj_center_pos, dim=0)  # [3]
        
        # 더미 오브젝트 포인트 클라우드 (object center 주변에 작은 구 형태)
        rs_verts_obj = torch.randn(1000, 3, device=self.device) * 0.05 + mean_center  # object center 주변
        
        # OakInk2 형식에 맞게 추가 데이터 설정
        data.update({
            "data_path": seq_info['path'],
            "obj_verts": rs_verts_obj,
            "scene_objs": [],  # 빈 리스트
            "obj_urdf_path": f"dummy_urdf_path_{seq_info['sequence_id']}.urdf"
        })

        self.process_data(data, idx, rs_verts_obj)

        seq_id = seq_info['sequence_id']
        retargeted_data_path = os.path.join(
            self.data_dir, 'retargeted', f"{seq_id}_retargeted.pkl"
        )

        self.load_retargeted_data(data, retargeted_data_path)
        
        return data

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