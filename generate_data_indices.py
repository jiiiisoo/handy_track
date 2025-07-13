#!/usr/bin/env python3
"""
GigaHands 데이터셋의 모든 시퀀스와 프레임을 찾아서 
ManipTrans에서 사용할 수 있는 dataIndices 형식으로 출력하는 스크립트
"""

import os
import json
import glob
from pathlib import Path

def find_all_sequences(data_root="/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose"):
    """
    모든 시퀀스와 해당하는 프레임을 찾습니다.
    
    Returns:
        list: dataIndices 형식의 리스트 (seq_id@frame_offset)
    """
    data_indices = []
    
    if not os.path.exists(data_root):
        print(f"데이터 경로가 존재하지 않습니다: {data_root}")
        return data_indices
    
    # 모든 시퀀스 디렉토리 찾기
    sequence_dirs = [d for d in os.listdir(data_root) 
                    if os.path.isdir(os.path.join(data_root, d)) and d.startswith('p')]
    
    sequence_dirs.sort()
    print(f"총 {len(sequence_dirs)}개의 시퀀스를 발견했습니다.")
    
    for seq_dir in sequence_dirs:
        mano_path = os.path.join(data_root, seq_dir, "keypoints_3d_mano")
        
        if not os.path.exists(mano_path):
            print(f"MANO 데이터가 없습니다: {seq_dir}")
            continue
            
        # JSON 파일들 찾기
        json_files = glob.glob(os.path.join(mano_path, "*.json"))
        json_files.sort()
        
        if not json_files:
            print(f"JSON 파일이 없습니다: {seq_dir}")
            continue
            
        print(f"{seq_dir}: {len(json_files)}개 프레임")
        
        # 시퀀스 ID 추출 (p001-folder -> p001folder)
        seq_id = seq_dir.replace('-', '')
        
        # 각 프레임에 대해 인덱스 생성
        for json_file in json_files:
            frame_num = int(os.path.basename(json_file).split('.')[0])
            data_indices.append(f"{seq_id}@{frame_num}")
    
    return data_indices

def sample_data_indices(data_indices, max_samples=None, every_n_frames=1):
    """
    데이터 인덱스를 샘플링합니다.
    
    Args:
        data_indices: 전체 데이터 인덱스 리스트
        max_samples: 최대 샘플 개수 (None이면 모두 사용)
        every_n_frames: N 프레임마다 하나씩 샘플링
    """
    if every_n_frames > 1:
        # 각 시퀀스에서 N 프레임마다 샘플링
        sampled = []
        current_seq = None
        frame_count = 0
        
        for idx in data_indices:
            seq_id = idx.split('@')[0]
            if seq_id != current_seq:
                current_seq = seq_id
                frame_count = 0
            
            if frame_count % every_n_frames == 0:
                sampled.append(idx)
            frame_count += 1
        
        data_indices = sampled
    
    if max_samples and len(data_indices) > max_samples:
        # 균등하게 샘플링
        step = len(data_indices) // max_samples
        data_indices = data_indices[::step][:max_samples]
    
    return data_indices

def generate_train_command(data_indices, max_samples=None, every_n_frames=1):
    """
    훈련 명령어를 생성합니다.
    """
    sampled_indices = sample_data_indices(data_indices, max_samples, every_n_frames)
    
    # dataIndices 문자열 생성
    indices_str = "[" + ",".join(sampled_indices) + "]"
    
    command = f"""python main/rl/train.py \\
    dexhand=inspire \\
    task=DexHandImitator \\
    side=RH \\
    headless=true \\
    num_envs=5120 \\
    test=false \\
    randomStateInit=true \\
    dataIndices={indices_str} \\
    learning_rate=2e-4 \\
    actionsMovingAverage=0.4 \\
    usePIDControl=False"""
    
    return command, len(sampled_indices)

if __name__ == "__main__":
    print("=== GigaHands 데이터셋 인덱스 생성기 ===")
    
    # 모든 데이터 인덱스 찾기
    all_indices = find_all_sequences()
    
    print(f"\n총 발견된 데이터 포인트: {len(all_indices)}")
    
    if len(all_indices) == 0:
        print("데이터를 찾을 수 없습니다. 경로를 확인해주세요.")
        exit(1)
    
    # 샘플 인덱스 몇 개 출력
    print(f"\n처음 10개 인덱스 예시:")
    for idx in all_indices[:10]:
        print(f"  {idx}")
    
    # 다양한 샘플링 옵션 제공
    options = [
        ("모든 데이터 사용", None, 1),
        ("10000개 샘플", 10000, 1),
        ("5000개 샘플", 5000, 1),
        ("매 2프레임마다 샘플링", None, 2),
        ("매 5프레임마다 샘플링", None, 5),
        ("매 2프레임마다 + 5000개 샘플", 5000, 2),
    ]
    
    print(f"\n=== 훈련 명령어 옵션 ===")
    
    for i, (desc, max_samples, every_n) in enumerate(options, 1):
        command, count = generate_train_command(all_indices, max_samples, every_n)
        print(f"\n{i}. {desc} ({count}개 데이터 포인트):")
        print(command)
        print()
    
    # 권장 사항
    print("=== 권장 사항 ===")
    print("1. 처음 테스트할 때는 '매 5프레임마다 + 5000개 샘플' 사용")
    print("2. 충분한 GPU 메모리가 있다면 '매 2프레임마다 샘플링' 사용")
    print("3. 전체 데이터를 사용하려면 '모든 데이터 사용' (매우 오래 걸림)") 