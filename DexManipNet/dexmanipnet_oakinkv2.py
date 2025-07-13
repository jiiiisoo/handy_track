import os
import pickle
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
import json
from DexManipNet.dexmanipnet import DexManipNet


class DexManipNetOakInkV2(DexManipNet):
    def __init__(
        self,
        *,
        data_dir: str,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)

    def __getitem__(self, idx):

        data = super().__getitem__(idx)
        seq_info = json.load(open(data["seq_info_path"], "r"))

        if self.side != "lh":
            data["oid_rh"] = seq_info["oid_rh"]
            data["obj_rh_path"] = os.path.join(self.data_dir, seq_info["obj_rh_path"])
        if self.side != "rh":
            data["oid_lh"] = seq_info["oid_lh"]
            data["obj_lh_path"] = os.path.join(self.data_dir, seq_info["obj_lh_path"])
        data["scene_objs"] = []
        data["start_frame"] = seq_info["start_frame"]
        data["seq_len"] = seq_info["seq_len"]
        return data


if __name__ == "__main__":
    # Example usage
    dataset = DexManipNetOakInkV2(data_dir="data/dexmanipnet/dexmanipnet_oakinkv2", side="rh")
    print(dataset[0])  # Access the first item in the dataset
