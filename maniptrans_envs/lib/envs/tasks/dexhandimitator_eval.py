from __future__ import annotations

import os
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from gym import spaces
from typing import Dict, List, Tuple

from .dexhandimitator import (
    DexHandImitatorRHEnv,
    DexHandImitatorLHEnv,
    aa_to_quat,
    rotmat_to_quat,
    quat_conjugate,
    quat_mul,
    quat_to_angle_axis,
)
from ...utils import torch_jit_utils as torch_jit_utils
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass


class DexHandImitatorEvalManipRHEnv(DexHandImitatorRHEnv):
    """
    Evaluation-only environment: keep the Imitator observation/action interface and policy,
    but spawn and track a manipulation object like ResDexHand. Compute success/failure and
    additional reward_dict terms based on the object trajectory tracking performance.
    """

    def _post_create_actors(self, env_ptr, env_id):
        # Add object actor per env during creation to satisfy actor ordering
        if not hasattr(self, "obj_handles"):
            self.obj_handles = []
            self.manip_obj_mass = []
            self.manip_obj_com = []
        if "obj_trajectory" not in self.demo_data:
            if len(self.obj_handles) < env_id + 1:
                self.obj_handles.append(None)
            return

        obj_tf_0 = self.demo_data["obj_trajectory"][env_id, 0]
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(obj_tf_0[0, 3]), float(obj_tf_0[1, 3]), float(obj_tf_0[2, 3]))
        obj_quat = rotmat_to_quat(obj_tf_0[:3, :3])
        obj_quat = obj_quat[[1, 2, 3, 0]]
        pose.r = gymapi.Quat(obj_quat[0].item(), obj_quat[1].item(), obj_quat[2].item(), obj_quat[3].item())

        obj_asset = None
        if "obj_urdf_path" in self.demo_data:
            asset_options = gymapi.AssetOptions()
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            obj_asset = self.gym.load_asset(self.sim, *os.path.split(self.demo_data["obj_urdf_path"][env_id]), asset_options)
            # Fallback if asset has no rigid bodies
            try:
                if self.gym.get_asset_rigid_body_count(obj_asset) == 0:
                    fallback = gymapi.AssetOptions()
                    fallback.fix_base_link = False
                    obj_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, fallback)
            except Exception:
                pass
        else:
            raise RuntimeError("no obj urdf path in demo")

        obj_id = None
        if "obj_id" in self.demo_data:
            oid = self.demo_data["obj_id"][env_id]
            if isinstance(oid, torch.Tensor):
                try:
                    obj_id = str(oid.item())
                except Exception:
                    obj_id = str(oid)
            else:
                obj_id = str(oid)
        # obj_actor = self.gym.create_actor(env_ptr, obj_asset, pose, "manip_obj", env_id + 2 * self.num_envs, 0)
        # make hand, object, table in same group
        obj_actor = self.gym.create_actor(env_ptr, obj_asset, pose, "manip_obj",
                                  env_id, 0)
        if obj_id is not None and obj_id in oakink2_obj_scale:
            self.gym.set_actor_scale(env_ptr, obj_actor, float(oakink2_obj_scale[obj_id]))
        obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, obj_actor)
        if len(obj_props) > 0:
            # Adjust mass on base link conservatively
            try:
                base_mass = obj_props[0].mass
            except Exception:
                base_mass = 0.1
            new_mass = min(0.5, base_mass)
            if obj_id is not None and obj_id in oakink2_obj_mass and oakink2_obj_mass[obj_id] is not None:
                new_mass = float(oakink2_obj_mass[obj_id])
            obj_props[0].mass = new_mass
            self.gym.set_actor_rigid_body_properties(env_ptr, obj_actor, obj_props)
            self.manip_obj_mass.append(new_mass)
            try:
                self.manip_obj_com.append(torch.tensor([obj_props[0].com.x, obj_props[0].com.y, obj_props[0].com.z]))
            except Exception:
                self.manip_obj_com.append(torch.zeros(3))
        else:
            # No rigid bodies: append sensible defaults
            self.manip_obj_mass.append(0.1)
            self.manip_obj_com.append(torch.zeros(3))
        rb_props = self.gym.get_actor_rigid_shape_properties(env_ptr, obj_actor)
        if len(rb_props) > 0:
            for element in rb_props:
                try:
                    element.filter = 0
                except Exception:
                    pass
                element.friction = 2.0
                element.rolling_friction = 0.05
                element.torsion_friction = 0.05
            self.gym.set_actor_rigid_shape_properties(env_ptr, obj_actor, rb_props)
        self.obj_handles.append(obj_actor)

    def init_data(self):
        # Use parent behavior, then bind object state views if present
        super().init_data()
        if hasattr(self, "obj_handles") and len(self.obj_handles) == self.num_envs and any(h is not None for h in self.obj_handles):
            env0 = self.envs[0]
            self._manip_obj_handle = self.gym.find_actor_handle(env0, "manip_obj")
            self._manip_obj_root_state = self._root_state[:, self._manip_obj_handle, :]
            CONTACT_HISTORY_LEN = 3
            self.tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()
            if len(self.manip_obj_mass) == self.num_envs:
                self.manip_obj_mass = torch.tensor(self.manip_obj_mass, device=self.device)
                self.manip_obj_com = torch.stack(self.manip_obj_com, dim=0).to(self.device)

    def _update_states(self):
        super()._update_states()
        if hasattr(self, "_manip_obj_handle") and self._manip_obj_handle is not None:
            self.states.update(
                {
                    "manip_obj_pos": self._manip_obj_root_state[:, :3],
                    "manip_obj_quat": self._manip_obj_root_state[:, 3:7],
                    "manip_obj_vel": self._manip_obj_root_state[:, 7:10],
                    "manip_obj_ang_vel": self._manip_obj_root_state[:, 10:13],
                }
            )

    def compute_reward(self, actions):
        # call base to compute imitation rewards and success/failure for hand-only
        super().compute_reward(actions)

        # additionally evaluate object tracking if object exists and demo has trajectories
        if getattr(self, "_manip_obj_handle", None) is None or "obj_trajectory" not in self.demo_data:
            return

        max_length = torch.clip(self.demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf
        cur_obj_tf = self.demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        target_obj_pos = cur_obj_tf[:, :3, 3]
        target_obj_quat = rotmat_to_quat(cur_obj_tf[:, :3, :3])[:, [1, 2, 3, 0]]

        cur_obj_pos = self.states.get("manip_obj_pos")
        cur_obj_quat = self.states.get("manip_obj_quat")
        if cur_obj_pos is None or cur_obj_quat is None:
            return

        diff_pos = torch.norm(target_obj_pos - cur_obj_pos, dim=-1)
        diff_rot_angle = quat_to_angle_axis(quat_mul(target_obj_quat, quat_conjugate(cur_obj_quat)))[0]
        diff_rot = diff_rot_angle.abs()

        # fingertip contact forces and history
        tip_force = torch.stack(
            [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
            axis=1,
        )
        self.tips_contact_history = torch.cat(
            [self.tips_contact_history[:, 1:], (torch.norm(tip_force, dim=-1) > 0)[:, None]], dim=1
        )

        # approximate fingertip distance to object (min over tips)
        tips_pos = torch.stack(
            [self._rigid_body_state[:, self.dexhand_handles[k], :][:, :3] for k in self.dexhand.contact_body_names],
            dim=1,
        )  # (N, TIPS, 3)
        tips_dist = torch.norm(tips_pos - cur_obj_pos[:, None, :], dim=-1)  # (N, TIPS)

        # Error buffer similar to ResDexHand
        current_eef_vel = self.states["base_state"][:, 7:10]
        current_eef_ang_vel = self.states["base_state"][:, 10:13]
        joints_vel = self.states["joints_state"][:, 1:, 7:10]
        current_dof_vel = self.states["dq"]
        current_obj_vel = self.states["manip_obj_vel"]
        current_obj_ang_vel = self.states["manip_obj_ang_vel"]
        error_buf = (
            (torch.norm(current_eef_vel, dim=-1) > 100)
            | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
            | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
            | (torch.abs(current_dof_vel).mean(-1) > 200)
            | (torch.norm(current_obj_vel, dim=-1) > 100)
            | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
        )

        # Failure rule similar to ResDexHand
        scale_factor = 1.0
        obj_pos_fail = diff_pos > (0.02 / 0.343) * (scale_factor ** 3)
        obj_rot_fail = (diff_rot / np.pi * 180) > (30 / 0.343) * (scale_factor ** 3)
        contact_cond_fail = torch.any((tips_dist < 0.005) & ~(self.tips_contact_history.any(1)), dim=-1)

        failed_execute = (
            (obj_pos_fail | obj_rot_fail | contact_cond_fail) & (self.running_progress_buf >= 8)
        ) | error_buf

        # Success if reached end and not failed
        reached_end = (self.progress_buf + 1 + 3 >= max_length)
        obj_succeeded = reached_end & ~failed_execute
        obj_failed = failed_execute

        # Merge into buffers (do not overwrite hand-only result, just OR)
        self.success_buf |= obj_succeeded
        self.failure_buf |= obj_failed

        # Extend reward_dict for logging
        self.reward_dict = self.reward_dict if isinstance(self.reward_dict, dict) else {}
        self.reward_dict.update(
            {
                "eval_obj_pos_err": diff_pos,
                "eval_obj_rot_err_deg": (diff_rot / np.pi * 180),
                "eval_tip_min_dist": tips_dist.min(dim=1).values,
                "eval_tip_any_contact": self.tips_contact_history.any(1).float(),
            }
        )


class DexHandImitatorEvalManipLHEnv(DexHandImitatorEvalManipRHEnv, DexHandImitatorLHEnv):
    side = "left"



