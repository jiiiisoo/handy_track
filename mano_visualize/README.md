# MANO 시각화 도구

GigaHands 데이터셋의 MANO 파라미터를 시각화하는 도구입니다. 메쉬와 키포인트 렌더링, 애니메이션 생성, 궤적 분석 기능을 제공합니다.

## 📁 파일 구조

```
mano_visualize/
├── mano_mesh_renderer.py         # 메쉬 렌더링 클래스
├── mano_keypoint_renderer.py     # 키포인트 렌더링 클래스
├── run_mano_visualization.py     # 통합 실행 스크립트
└── README.md                     # 사용법 가이드
```

## 🚀 빠른 시작

### 1. 기본 사용법

```bash
# 모든 시각화 실행 (메쉬 + 키포인트 + 애니메이션 + 궤적 분석)
python run_mano_visualization.py --all

# 키포인트만 렌더링
python run_mano_visualization.py --render_keypoints

# 메쉬만 렌더링  
python run_mano_visualization.py --render_mesh

# 애니메이션만 생성
python run_mano_visualization.py --create_animation

# 궤적 분석만 실행
python run_mano_visualization.py --analyze_trajectory
```

### 2. 고급 옵션

```bash
# 사용자 정의 JSON 파일과 프레임 범위
python run_mano_visualization.py \
    --json_path /path/to/your/data.json \
    --hand_type right \
    --start_frame 10 \
    --end_frame 50 \
    --all

# 특정 관절의 궤적 분석
python run_mano_visualization.py \
    --analyze_trajectory \
    --joint_idx 12 \
    --hand_type left

# 출력 디렉토리 지정
python run_mano_visualization.py \
    --all \
    --output_dir ./my_output
```

## 📊 출력 결과

### 1. 메쉬 렌더링
- **정적 이미지**: `mesh_images/` 폴더에 각 프레임별 메쉬 이미지
- **애니메이션**: `{hand_type}_mesh_animation.gif`

### 2. 키포인트 렌더링  
- **정적 이미지**: `keypoint_images/` 폴더에 각 프레임별 키포인트 이미지
- **애니메이션**: `{hand_type}_keypoint_animation.gif`

### 3. 궤적 분석
- **분석 결과**: `{hand_type}_joint_{joint_idx}_trajectory.png`

## 🎯 주요 기능

### 1. 메쉬 렌더링 (MANOMeshRenderer)

```python
from mano_mesh_renderer import MANOMeshRenderer

renderer = MANOMeshRenderer()

# 시퀀스 렌더링
renderer.render_mesh_sequence(
    json_path="path/to/data.json",
    hand_type='left',
    frame_range=(0, 30),
    output_dir="output"
)

# 애니메이션 생성
renderer.create_mesh_animation(
    json_path="path/to/data.json",
    hand_type='left',
    output_path="animation.gif",
    fps=10
)
```

**특징:**
- 3D 메쉬 정점 시각화
- 관절 위치와 스켈레톤 표시
- 3D 뷰와 2D 투영 동시 표시
- GIF 애니메이션 생성

### 2. 키포인트 렌더링 (MANOKeypointRenderer)

```python
from mano_keypoint_renderer import MANOKeypointRenderer

renderer = MANOKeypointRenderer()

# 키포인트 시퀀스 렌더링
renderer.render_keypoint_sequence(
    json_path="path/to/data.json",
    hand_type='left',
    frame_range=(0, 30),
    output_dir="output"
)

# 궤적 분석
renderer.analyze_keypoint_trajectory(
    json_path="path/to/data.json",
    joint_idx=8,  # index fingertip
    output_path="trajectory.png"
)
```

**특징:**
- 21개 MANO 관절 시각화
- 손가락별 색상 구분
- 3D, XY, XZ 뷰 제공
- 궤적 분석 및 속도 계산

## 🎨 시각화 예시

### 메쉬 렌더링
- **좌측**: 3D 메쉬 (정점 + 관절 + 스켈레톤)
- **우측**: 2D 투영 (XY 평면)

### 키포인트 렌더링
- **좌측**: 3D 키포인트 (손가락별 색상)
- **중앙**: XY 투영
- **우측**: XZ 투영

### 궤적 분석
- **좌상**: 3D 궤적
- **우상**: XY 투영
- **좌하**: 시간에 따른 위치 변화
- **우하**: 속도 변화

## 🔧 관절 인덱스 참조

```python
joint_names = [
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
```

## 📋 명령행 옵션

### 필수 옵션
- `--json_path`: GigaHands JSON 파일 경로 (기본값: `/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose/p001-folder/params/000.json`)

### 선택 옵션
- `--hand_type`: 손 타입 (`left` 또는 `right`, 기본값: `left`)
- `--start_frame`: 시작 프레임 (기본값: `0`)
- `--end_frame`: 종료 프레임 (기본값: `30`)
- `--output_dir`: 출력 디렉토리 (기본값: `mano_visualization_outputs`)

### 실행 옵션
- `--render_mesh`: 메쉬 렌더링 활성화
- `--render_keypoints`: 키포인트 렌더링 활성화
- `--create_animation`: 애니메이션 생성 활성화
- `--analyze_trajectory`: 궤적 분석 활성화
- `--all`: 모든 기능 실행

### 고급 옵션
- `--mano_path`: MANO 모델 파일 경로 (옵션)
- `--joint_idx`: 궤적 분석할 관절 인덱스 (기본값: `8`)
- `--fps`: 애니메이션 FPS (기본값: `10`)

## 📦 의존성

### 필수 패키지
```bash
pip install torch numpy matplotlib pillow
```

### 선택적 패키지 (실제 MANO 모델 사용시)
```bash
# MANO 모델 (manopth)
pip install git+https://github.com/hassony2/manopth.git

# 또는 로컬 설치
git clone https://github.com/hassony2/manopth.git
cd manopth
pip install -e .
```

## ⚠️ 주의사항

1. **MANO 모델 없이도 실행 가능**: Mock 구현을 통해 MANO 모델 없이도 데모 실행 가능
2. **메모리 사용량**: 긴 시퀀스는 많은 메모리를 사용할 수 있으므로 프레임 범위를 조절하세요
3. **애니메이션 생성**: Pillow 패키지가 필요하며, 큰 시퀀스는 시간이 오래 걸릴 수 있습니다

## 🐛 문제 해결

### 1. Import Error
```bash
# 패키지 설치 확인
pip install torch numpy matplotlib pillow

# Python 경로 확인
export PYTHONPATH=$PYTHONPATH:/path/to/mano_visualize
```

### 2. JSON 파일 오류
```bash
# 파일 존재 확인
ls -la /mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose/p001-folder/params/000.json

# 권한 확인
chmod 644 /path/to/json/file
```

### 3. 메모리 부족
```bash
# 프레임 수 줄이기
python run_mano_visualization.py --end_frame 10 --all

# 또는 단계별 실행
python run_mano_visualization.py --render_keypoints
```

## 📈 예시 실행 결과

```bash
$ python run_mano_visualization.py --all

============================================================
        GigaHands MANO Visualization Tool
============================================================
JSON 파일: /mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose/p001-folder/params/000.json
손 타입: left
프레임 범위: 0 - 30
출력 디렉토리: mano_visualization_outputs

실행할 작업:
  ✅ Mesh
  ✅ Keypoints
  ✅ Animation
  ✅ Trajectory

🔧 1. 메쉬 렌더링 시작...
Using mock MANO model
Rendering 30 frames for left hand...
✅ 메쉬 이미지 저장 완료: mano_visualization_outputs/mesh_images

🎯 2. 키포인트 렌더링 시작...
Using mock MANO model for keypoints
Rendering 30 keypoint frames for left hand...
✅ 키포인트 이미지 저장 완료: mano_visualization_outputs/keypoint_images

🎬 3. 애니메이션 생성 시작...
✅ 메쉬 애니메이션 저장 완료: mano_visualization_outputs/left_mesh_animation.gif
✅ 키포인트 애니메이션 저장 완료: mano_visualization_outputs/left_keypoint_animation.gif

📈 4. 궤적 분석 시작...
Computing trajectory for joint 8 (index_tip)...
✅ 궤적 분석 저장 완료: mano_visualization_outputs/left_joint_8_trajectory.png

============================================================
        실행 결과 요약
============================================================
성공: 4/4 작업
🎉 모든 작업이 성공적으로 완료되었습니다!

📁 출력 파일들을 확인하세요: mano_visualization_outputs
```

## 🤝 기여하기

버그 리포트나 기능 제안은 이슈를 통해 남겨주세요!

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 