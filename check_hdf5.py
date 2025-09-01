#!/usr/bin/env python3
import h5py
import numpy as np

def check_hdf5_structure(filepath):
    """HDF5 파일 구조 확인"""
    print(f"🔍 Checking file: {filepath}")
    print("="*60)
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("📁 Root level keys:")
            for key in f.keys():
                print(f"  - {key}")
            
            print("\n📊 Detailed structure:")
            def print_structure(name, obj):
                indent = "  " * (name.count('/') + 1)
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}📊 Dataset: {name}")
                    print(f"{indent}   Shape: {obj.shape}")
                    print(f"{indent}   Type: {obj.dtype}")
                    if obj.size > 0 and obj.size < 10:  # 작은 데이터셋은 값 출력
                        print(f"{indent}   Values: {obj[:]}")
                    print()
                else:
                    print(f"{indent}📁 Group: {name}/")
                    # 그룹 안의 키들 출력
                    try:
                        group_keys = list(obj.keys())
                        if group_keys:
                            print(f"{indent}   Contains: {group_keys}")
                        print()
                    except:
                        print(f"{indent}   (Cannot list contents)")
                        print()
            
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    return True

if __name__ == "__main__":
    hdf5_path = "/scratch2/jisoo6687/handy_track/dumps/dump_DexHandImitator__07-17-14-56-30/rollouts.hdf5"
    check_hdf5_structure(hdf5_path) 