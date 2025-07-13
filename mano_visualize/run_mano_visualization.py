#!/usr/bin/env python3
"""
GigaHands MANO 파라미터 시각화 통합 스크립트

메쉬와 키포인트 렌더링을 모두 실행하는 메인 스크립트입니다.

Usage:
    python run_mano_visualization.py [options]
"""

import argparse
import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from mano_mesh_renderer import MANOMeshRenderer
    from mano_keypoint_renderer import MANOKeypointRenderer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure mano_mesh_renderer.py and mano_keypoint_renderer.py are in the same directory")
    sys.exit(1)

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="GigaHands MANO 파라미터 시각화",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 입력 파일
    parser.add_argument(
        '--json_path', 
        type=str, 
        default="/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose/p001-folder/params/000.json",
        help="GigaHands JSON 파일 경로"
    )
    
    # 손 타입
    parser.add_argument(
        '--hand_type', 
        type=str, 
        choices=['left', 'right'], 
        default='right',
        help="렌더링할 손 (left 또는 right)"
    )
    
    # 프레임 범위
    parser.add_argument(
        '--start_frame', 
        type=int, 
        default=0,
        help="시작 프레임"
    )
    
    parser.add_argument(
        '--end_frame', 
        type=int, 
        default=30,
        help="종료 프레임"
    )
    
    # 출력 설정
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="mano_visualization_outputs",
        help="출력 디렉토리"
    )
    
    # 렌더링 옵션
    parser.add_argument(
        '--render_mesh', 
        action='store_true',
        help="메쉬 렌더링 활성화"
    )
    
    parser.add_argument(
        '--render_keypoints', 
        action='store_true',
        help="키포인트 렌더링 활성화"
    )
    
    parser.add_argument(
        '--create_animation', 
        action='store_true',
        help="애니메이션 생성 활성화"
    )
    
    parser.add_argument(
        '--analyze_trajectory', 
        action='store_true',
        help="궤적 분석 활성화"
    )
    
    # 전체 실행
    parser.add_argument(
        '--all', 
        action='store_true',
        help="모든 시각화 실행 (mesh + keypoints + animation + trajectory)"
    )
    
    # MANO 모델 경로
    parser.add_argument(
        '--mano_path', 
        type=str, 
        default=None,
        help="MANO 모델 파일 경로 (옵션)"
    )
    
    # 궤적 분석 옵션
    parser.add_argument(
        '--joint_idx', 
        type=int, 
        default=8,
        help="궤적 분석할 관절 인덱스 (기본: 8 = index_tip)"
    )
    
    # 애니메이션 설정
    parser.add_argument(
        '--fps', 
        type=int, 
        default=10,
        help="애니메이션 FPS"
    )
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    args = parse_args()
    
    print("=" * 60)
    print("        GigaHands MANO Visualization Tool")
    print("=" * 60)
    print(f"JSON 파일: {args.json_path}")
    print(f"손 타입: {args.hand_type}")
    print(f"프레임 범위: {args.start_frame} - {args.end_frame}")
    print(f"출력 디렉토리: {args.output_dir}")
    print()
    
    # 입력 파일 확인
    if not os.path.exists(args.json_path):
        print(f"❌ 오류: JSON 파일을 찾을 수 없습니다: {args.json_path}")
        return 1
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 프레임 범위 설정
    frame_range = (args.start_frame, args.end_frame)
    
    # 실행할 작업 결정
    tasks = {
        'mesh': args.render_mesh or args.all,
        'keypoints': args.render_keypoints or args.all,
        'animation': args.create_animation or args.all,
        'trajectory': args.analyze_trajectory or args.all
    }
    
    if not any(tasks.values()):
        print("⚠️  경고: 실행할 작업이 선택되지 않았습니다.")
        print("   --render_mesh, --render_keypoints, --create_animation, --analyze_trajectory 중 하나 이상을 선택하거나")
        print("   --all을 사용하여 모든 작업을 실행하세요.")
        return 1
    
    print("실행할 작업:")
    for task, enabled in tasks.items():
        status = "✅" if enabled else "⏸️"
        print(f"  {status} {task.capitalize()}")
    print()
    
    success_count = 0
    total_count = sum(tasks.values())
    
    # 1. 메쉬 렌더링
    if tasks['mesh']:
        print("🔧 1. 메쉬 렌더링 시작...")
        renderer = MANOMeshRenderer(model_path=args.mano_path)
        
        # 정적 이미지
        mesh_output_dir = os.path.join(args.output_dir, "mesh_images")
        renderer.render_mesh_sequence(
            json_path=args.json_path,
            hand_type=args.hand_type,
            frame_range=frame_range,
            output_dir=mesh_output_dir
        )
        
        print(f"✅ 메쉬 이미지 저장 완료: {mesh_output_dir}")
        success_count += 1
    
    # 2. 키포인트 렌더링
    if tasks['keypoints']:
        print("\n🎯 2. 키포인트 렌더링 시작...")
        try:
            renderer = MANOKeypointRenderer(model_path=args.mano_path)
            
            # 정적 이미지
            keypoint_output_dir = os.path.join(args.output_dir, "keypoint_images")
            renderer.render_keypoint_sequence(
                json_path=args.json_path,
                hand_type=args.hand_type,
                frame_range=frame_range,
                output_dir=keypoint_output_dir
            )
            
            print(f"✅ 키포인트 이미지 저장 완료: {keypoint_output_dir}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 키포인트 렌더링 오류: {e}")
    
    # 3. 애니메이션 생성
    if tasks['animation']:
        print("\n🎬 3. 애니메이션 생성 시작...")
        try:
            # 메쉬 애니메이션
            if tasks['mesh']:
                mesh_renderer = MANOMeshRenderer(model_path=args.mano_path)
                mesh_anim_path = os.path.join(args.output_dir, f"{args.hand_type}_mesh_animation.gif")
                mesh_renderer.create_mesh_animation(
                    json_path=args.json_path,
                    hand_type=args.hand_type,
                    frame_range=frame_range,
                    output_path=mesh_anim_path,
                    fps=args.fps
                )
                print(f"✅ 메쉬 애니메이션 저장 완료: {mesh_anim_path}")
            
            # 키포인트 애니메이션
            if tasks['keypoints']:
                keypoint_renderer = MANOKeypointRenderer(model_path=args.mano_path)
                keypoint_anim_path = os.path.join(args.output_dir, f"{args.hand_type}_keypoint_animation.gif")
                keypoint_renderer.create_keypoint_animation(
                    json_path=args.json_path,
                    hand_type=args.hand_type,
                    frame_range=frame_range,
                    output_path=keypoint_anim_path,
                    fps=args.fps
                )
                print(f"✅ 키포인트 애니메이션 저장 완료: {keypoint_anim_path}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ 애니메이션 생성 오류: {e}")
    
    # 4. 궤적 분석
    if tasks['trajectory']:
        print("\n📈 4. 궤적 분석 시작...")
        try:
            renderer = MANOKeypointRenderer(model_path=args.mano_path)
            trajectory_path = os.path.join(args.output_dir, f"{args.hand_type}_joint_{args.joint_idx}_trajectory.png")
            
            renderer.analyze_keypoint_trajectory(
                json_path=args.json_path,
                hand_type=args.hand_type,
                joint_idx=args.joint_idx,
                output_path=trajectory_path
            )
            
            print(f"✅ 궤적 분석 저장 완료: {trajectory_path}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 궤적 분석 오류: {e}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("        실행 결과 요약")
    print("=" * 60)
    print(f"성공: {success_count}/{total_count} 작업")
    
    if success_count == total_count:
        print("🎉 모든 작업이 성공적으로 완료되었습니다!")
        print(f"\n📁 출력 파일들을 확인하세요: {args.output_dir}")
        
        # 생성된 파일들 나열
        print("\n생성된 파일들:")
        for root, dirs, files in os.walk(args.output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, args.output_dir)
                print(f"  📄 {rel_path}")
        
        return 0
    else:
        print(f"⚠️  일부 작업이 실패했습니다. ({total_count - success_count}개 실패)")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 