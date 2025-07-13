import os
import pickle
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
import json
from DexManipNet.dexmanipnet import DexManipNet


class DexManipNetFAVOR(DexManipNet):
    def __init__(
        self,
        *,
        data_dir: str,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.color_dict = {
            "red": [1, 0, 0],
            "green": [0, 1, 0],
            "blue": [0, 0, 1],
            "yellow": [1, 1, 0],
            "purple": [1, 0, 1],
            "cyan": [0, 1, 1],
        }

    def __getitem__(self, idx):

        data = super().__getitem__(idx)
        seq_info = json.load(open(data["seq_info_path"], "r"))
        data["oid_rh"] = seq_info["oid_rh"]
        data["obj_rh_path"] = os.path.join(self.data_dir, seq_info["obj_rh_path"])
        data["oid_lh"] = None
        data["obj_lh_path"] = None
        data["description"] = seq_info["description"]
        data["scene_objs"] = seq_info["scene_objects"]
        for scene_obj in data["scene_objs"]:
            scene_obj["color"] = self.color_dict.get(scene_obj["color"], [1, 1, 1])

        return data


if __name__ == "__main__":
    # Example usage
    dataset = DexManipNetFAVOR(data_dir="data/dexmanipnet/dexmanipnet_favor", side="rh")
    print(dataset[0])  # Access the first item in the dataset
