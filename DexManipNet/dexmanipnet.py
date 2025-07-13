import os
import pickle
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
import json


class DexManipNet(Dataset):
    def __init__(
        self,
        *,
        data_dir: str,
        side: str = "all",  # "rh", "lh", "bih", "all"
        **kwargs,
    ):
        self.data_dir = data_dir

        seq_list = os.listdir(os.path.join(data_dir, "sequences"))
        self.side = side

        if side == "all":
            self.seq_list = seq_list
        else:
            self.seq_list = [s for s in seq_list if s.split("_")[1] == side]

        self.seq_list.sort()

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq_name = self.seq_list[idx]
        seq_path = os.path.join(self.data_dir, "sequences", seq_name)

        rollout_data = h5py.File(os.path.join(seq_path, "rollouts.hdf5"), "r")
        seq_info = json.load(open(os.path.join(seq_path, "seq_info.json"), "r"))

        rollouts = [r for r in rollout_data[f"rollouts/successful"]]
        rollouts_reward = np.array([rollout_data[f"rollouts/successful/{r}/reward"][:].sum() for r in rollouts])
        max_idx = rollouts[np.argmax(rollouts_reward)]  # ? Only the best rollout is used
        rollout = rollout_data[f"rollouts/successful/{max_idx}"]

        assert (
            rollout_data[f"rollouts/successful/rollout_0/reward"].shape[0]
            == rollout_data[f"rollouts/successful/rollout_0/actions"].shape[0]
            == rollout_data[f"rollouts/successful/rollout_0/dq_rh"].shape[0]
            == seq_info["seq_len"]
        )

        data = {}
        for k in rollout.keys():
            data[k] = torch.tensor(rollout[k][:], dtype=torch.float32)
        data["type"] = seq_info["type"]
        data["seq_name"] = seq_name.split("_")[0]
        data["seq_info_path"] = os.path.join(seq_path, "seq_info.json")
        data["dexhand"] = seq_info["dexhand"]
        data["primitive"] = seq_info["primitive"]
        data["primitive_rh"] = seq_info["primitive_rh"]
        data["primitive_lh"] = seq_info["primitive_lh"]
        data["interaction_mode"] = seq_info["interaction_mode"]
        return data
