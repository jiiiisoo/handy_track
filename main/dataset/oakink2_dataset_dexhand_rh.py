import json
import os
import pickle
from functools import lru_cache

import numpy as np
import smplx
import torch
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from smplx.lbs import batch_rigid_transform, batch_rodrigues
from termcolor import cprint
from torch.utils.data import Dataset
from typing import List
from main.dataset.oakink2_layer.smplx import SMPLXLayer
from main.dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa
from .base import ManipData
from .oakink2_dataset_utils import load_obj_map, as_mesh
from .decorators import register_manipdata


@register_manipdata("oakink2_rh")
class OakInk2DatasetDexHandRH(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/OakInk-v2",
        split: str = "all",
        skip: int = 2,  # OakInk2 120Hz, while DEXHAND 60Hz
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
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
            **kwargs,
        )

        pathes = os.listdir(os.path.join(data_dir, "anno_preview"))
        pathes = [os.path.join(data_dir, "anno_preview", p) for p in pathes]
        pathes.sort(key=lambda x: x.split("/")[-1])
        self.data_pathes = pathes
        # * We use the first 5 digits of hash as the index
        self.seq_hashes = {os.path.split(p)[-1].split("_")[5][:5]: i for i, p in enumerate(pathes)}

        SMPLX_ROT_MODE = "quat"
        SMPLX_DIM_SHAPE_ALL = 300

        self.smplx_layer = SMPLXLayer(
            "data/body_utils/body_models/smplx",
            dtype=torch.float32,
            rot_mode=SMPLX_ROT_MODE,
            num_betas=SMPLX_DIM_SHAPE_ALL,
            gender="neutral",
            use_body_upper_asset="data/smplx_extra/body_upper_idx.pt",
        ).to(self.device)
        self.smplx_faces_np = self.smplx_layer.body_upper_faces.detach().clone().cpu().numpy()
        self.smplx_body_upper_idx = self.smplx_layer.body_upper_vert_idx.detach().clone().cpu().numpy()
        self.smplx_layer = self.smplx_layer.to(self.device)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):

        if type(index) == str:
            index = (index.split("@")[0], int(index.split("@")[1]))

        assert (
            type(index) == tuple and len(index) == 2 and type(index[0]) == str and type(index[1]) == int
        ), "index error"
        assert (
            index[0] in self.seq_hashes
        ), f"index {index[0]} not found, please check the 5 digits hash (first 5 digits of the sequence hash) in the data_pathes"
        idx = self.seq_hashes[index[0]]
        stage = index[1]

        anno = self.data_pathes[idx]
        anno = pickle.load(open(anno, "rb"))

        frame_id_list = anno["mocap_frame_id_list"]  # ! 120HZ
        frame_id_list = frame_id_list[:: self.skip]

        program_filepath = os.path.join(
            self.data_dir,
            "program",
            "program_info",
            f"{os.path.splitext(os.path.split(self.data_pathes[idx])[1])[0]}.json",
        )
        program_info = {}
        with open(program_filepath, "r") as ifs:
            _program_info = json.load(ifs)
            for k, v in _program_info.items():
                seg_pair_def = eval(k)
                program_info[seg_pair_def] = v

        left_hand_range = list(program_info.keys())[stage][0]
        right_hand_range = list(program_info.keys())[stage][1]

        def intersection(lst1, lst2):
            begin = max(lst1[0], lst2[0])
            end = min(lst1[1], lst2[1])
            if begin < end:
                return [begin, end]
            else:
                assert False, f"no intersection between {lst1} and {lst2}"

        assert (
            right_hand_range is not None
        ), f"Right hand data is empty. Please check if {index} is a left-hand-only task."

        if left_hand_range is not None:
            right_hand_range = intersection(left_hand_range, right_hand_range)

        program_info_selected = program_info[list(program_info.keys())[stage]]

        frame_id_list = [f for f in frame_id_list if right_hand_range[0] <= f <= right_hand_range[1]]

        object_list = anno["obj_list"]

        length = len(frame_id_list)

        obj_transf_map = {}
        smplx_result = []
        for frame_id in frame_id_list:
            for obj in object_list:
                if obj not in obj_transf_map:
                    obj_transf_map[obj] = [anno["obj_transf"][obj][frame_id]]
                else:
                    obj_transf_map[obj].append(anno["obj_transf"][obj][frame_id])
            smplx_result.append(anno["raw_smplx"][frame_id])

        for o in obj_transf_map.keys():
            obj_transf_map[o] = torch.tensor(np.stack(obj_transf_map[o]).astype(np.float32), device=self.device)
        smplx_data = {k: [] for k in smplx_result[0].keys()}
        for smplx_k in smplx_result[0].keys():
            for d in smplx_result:
                smplx_data[smplx_k].append(d[smplx_k])
        smplx_data = {k: torch.cat(v).to(self.device) for k, v in smplx_data.items()}

        smplx_results = self.smplx_layer(**smplx_data)

        mano_joints = {
            "index_proximal": smplx_results.joints[:, 40].detach(),
            "index_intermediate": smplx_results.joints[:, 41].detach(),
            "index_distal": smplx_results.joints[:, 42].detach(),
            "index_tip": smplx_results.vertices[:, 7706].detach(),  # reselect tip
            "middle_proximal": smplx_results.joints[:, 43].detach(),
            "middle_intermediate": smplx_results.joints[:, 44].detach(),
            "middle_distal": smplx_results.joints[:, 45].detach(),
            "middle_tip": smplx_results.vertices[:, 7818].detach(),  # reselect tip
            "pinky_proximal": smplx_results.joints[:, 46].detach(),
            "pinky_intermediate": smplx_results.joints[:, 47].detach(),
            "pinky_distal": smplx_results.joints[:, 48].detach(),
            "pinky_tip": smplx_results.vertices[:, 8046].detach(),  # reselect tip
            "ring_proximal": smplx_results.joints[:, 49].detach(),
            "ring_intermediate": smplx_results.joints[:, 50].detach(),
            "ring_distal": smplx_results.joints[:, 51].detach(),
            "ring_tip": smplx_results.vertices[:, 7929].detach(),  # reselect tip
            "thumb_proximal": smplx_results.joints[:, 52].detach(),
            "thumb_intermediate": smplx_results.joints[:, 53].detach(),
            "thumb_distal": smplx_results.joints[:, 54].detach(),
            "thumb_tip": smplx_results.vertices[:, 8096].detach(),  # reselect tip
        }

        wrist_pos = smplx_results.joints[:, 21].detach()
        middle_pos = mano_joints["middle_proximal"]
        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # ? hack for wrist position
        wrist_pos += torch.tensor(self.dexhand.relative_translation, device=self.device)
        mano_rot_offset = self.dexhand.relative_rotation
        wrist_rot = smplx_results.transform_abs[:, 21, :3, :3].detach() @ torch.tensor(
            np.repeat(mano_rot_offset[None], smplx_results.transform_abs.shape[0], axis=0), device=self.device
        )

        obj_mesh = load_obj_map(os.path.join(self.data_dir, "object_preview", "align_ds"), object_list)
        obj_id = program_info_selected["obj_list_rh"][0]
        obj_mesh_trimesh = obj_mesh[program_info_selected["obj_list_rh"][0]][0]
        obj_mesh_path = obj_mesh[program_info_selected["obj_list_rh"][0]][1]

        mesh = Meshes(
            verts=torch.from_numpy(obj_mesh_trimesh.vertices[None, ...].astype(np.float32)),
            faces=torch.from_numpy(obj_mesh_trimesh.faces[None, ...].astype(np.float32)),
        )

        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": self.data_pathes[idx],
            "obj_id": obj_id,
            "obj_mesh_path": obj_mesh_path,
            "obj_verts": rs_verts_obj,
            "obj_trajectory": obj_transf_map[obj_id],
            "scene_objs": [],  #! BUG: NO TRANSFER
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }
        obj_mesh_path_dir, obj_mesh_path_file = os.path.split(앷)
        obj_urdf_path_dir = obj_mesh_path_dir.replace(
            "data/OakInk-v2/object_preview", "data/OakInk-v2/coacd_object_preview"
        )
        obj_urdf_path_file = obj_mesh_path_file.replace(".obj", ".urdf").replace(".ply", ".urdf")
        data["obj_urdf_path"] = os.path.join(obj_urdf_path_dir, obj_urdf_path_file)

        self.process_data(data, idx, rs_verts_obj)

        # todo load retargeted data
        OPT_INSPIRE_PATH = f"data/retargeting/OakInk-v2/mano2{str(self.dexhand)}"
        opt_path = os.path.join(
            OPT_INSPIRE_PATH, os.path.split(self.data_pathes[idx])[-1].replace(".pkl", f"@{stage}.pkl")
        )

        self.load_retargeted_data(data, opt_path)

        return data


if __name__ == "__main__":
    fdata = OakInk2DatasetDexHandRH(data_dir="data/OakInk-v2", mujoco2gym_transf=torch.eye(4, device="cuda:0"))
    print(fdata[34])
