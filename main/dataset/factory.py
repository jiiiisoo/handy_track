import os
import importlib
from .base import ManipData


class ManipDataFactory:
    _registry = {}

    @classmethod
    def register(cls, manipdata_type: str, data_class):
        """Register a new data type."""
        cls._registry[manipdata_type] = data_class

    @classmethod
    def create_data(cls, manipdata_type: str, side: str, *args, **kwargs) -> ManipData:
        assert side in ["left", "right"], f"Invalid side '{side}', must be 'left' or 'right'."
        """Create a data instance by type."""
        
        # data_type argument로 명시적 데이터셋 타입 선택 가능
        data_type = kwargs.pop("data_type", None)  # kwargs에서 제거, 없으면 None
        if data_type is not None:
            manipdata_type = data_type

        
        # GigaHands는 양손 데이터를 모두 포함하므로 side 구분 없이 처리
        if manipdata_type == "gigahands":
            # GigaHands 데이터셋을 직접 로드
            try:
                from .gigahands_dataset import GigaHandsDataset
                return GigaHandsDataset(*args, side=side, **kwargs)
            except ImportError as e:
                raise ValueError(f"Failed to import GigaHands dataset: {e}")
        elif manipdata_type == "oakink2":
            # OakInk2 데이터셋 처리
            manipdata_type += "_rh" if side == "right" else "_lh"
        else:
            # 기타 데이터셋들
            manipdata_type += "_rh" if side == "right" else "_lh"
            
        if manipdata_type not in cls._registry:
            raise ValueError(f"Data type '{manipdata_type}' not registered.")
        return cls._registry[manipdata_type](*args, **kwargs)

    @classmethod
    def auto_register_data(cls, directory: str, base_package: str):
        """Automatically import all data modules in the directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != "__init__.py":
                # GigaHands 데이터셋은 조건부 로딩으로 건너뜀
                if filename == "gigahands_dataset.py":
                    continue
                module_name = f"{base_package}.{filename[:-3]}"
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    print(f"Warning: Failed to import {module_name}: {e}")
                    continue

    @staticmethod
    def dataset_type(index):
        """We use the index to get the dataset type"""
        """Define your own dataset type and index format here"""

        if type(index) == str and index.endswith("M"):
            # !!! The right hand mirrored dataset comes from the original left hand dataset
            is_mirrored = True
            index = index[:-1]
        else:
            is_mirrored = False

        if type(index) == str and "@" in index:
            # GigaHands 데이터셋 인덱스 형식: seq_id@frame_offset
            # 3자리 숫자 형태도 GigaHands로 인식하도록 수정
            # seq_part = index.split("@")[0]
            # if any(char.isalpha() for char in seq_part) or (seq_part.isdigit() and len(seq_part) == 3):
            #     dtype = "gigahands"
            # else:
            dtype = "oakink2"
        elif type(index) == str and "_" in index and index.startswith("p"):
            # 새로운 GigaHands 형식: "p019-makeup_063"
            # p로 시작하고 언더스코어를 포함하는 경우 GigaHands로 인식
            dtype = "gigahands"
        elif type(index) == str and index.startswith("g"):
            dtype = "grabdemo"
        elif type(index) == str and index.startswith("v"):
            dtype = "visionpro"
        else:
            dtype = "favor"

        if is_mirrored:
            dtype += "_mirrored"
        return dtype

# GigaHands 데이터셋을 수동으로 등록 시도
try:
    from .gigahands_dataset import GigaHandsDataset
    ManipDataFactory.register("gigahands", GigaHandsDataset)
    print("GigaHands dataset registered successfully")
except ImportError:
    print("GigaHands dataset not registered due to missing dependencies")
    pass
