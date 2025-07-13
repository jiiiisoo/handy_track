from maniptrans_envs.lib.envs.core.vec_task import VecTask
import torch
import numpy as np
from tqdm import tqdm
from main.dataset.factory import ManipDataFactory
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.transform import aa_to_rotmat
from isaacgym import gymapi, gymtorch
import os


ROBOT_HEIGHT = 0.05  # 로봇 높이 설정


class DexHandImitatorHandOnlyRHEnv(VecTask):
    """
    Hand-only DexHandImitator environment (no object data required)
    """
    side = "right"
    
    def __init__(self, cfg, **kwargs):
        # 설정 값들 초기화
        self.cfg = cfg
        self.dexhand = DexHandFactory.create_hand(cfg["env"]["dexhand"], self.side)
        self.max_episode_length = cfg["env"]["episodeLength"]
        self.training = cfg["env"]["training"]
        self.rollout_state_init = cfg["env"]["rolloutStateInit"]
        self.random_state_init = cfg["env"]["randomStateInit"]
        self.dataIndices = cfg["env"]["dataIndices"]
        self.obs_future_length = cfg["env"]["obsFutureLength"]
        self.use_pid_control = cfg["env"]["usePIDControl"]
        self.actions_moving_average = cfg["env"]["actionsMovingAverage"]
        
        # Object-related attributes (disabled)
        self.has_object_data = False
        
        # 추가 속성들 설정
        self.control_freq_inv = 1  # physics steps per control step
        self.randomize = False
        self.aggregate_mode = 3
        self._record = False
        self.camera_handlers = None
        
        # States 초기화
        self.states = {}
        self.target_states = {}
        
        # Hand-only에서는 로봇 팔 + hand DOF 제어 (객체 제어만 없음)
        use_quat_rot = cfg["env"].get("useQuatRot", False)
        # cfg["env"]["numActions"] = self.dexhand.n_dofs
        cfg["env"]["numActions"] = (
            (1 + 6 + self.dexhand.n_dofs) if use_quat_rot else (6 + self.dexhand.n_dofs)
        ) + (3 if self.use_pid_control else 0)
        # print(f"DEBUG: numActions: {cfg['env']['numActions']}")
        # 1/0
        
        # sim_device 속성 설정 (VecTask 초기화 전에 필요)
        sim_device = kwargs.get('sim_device', 0)
        self.sim_device = torch.device(sim_device)
        
        # VecTask 초기화
        super().__init__(
            config=cfg,
            rl_device=kwargs.get('rl_device', 0),
            sim_device=kwargs.get('sim_device', 0),
            graphics_device_id=kwargs.get('graphics_device_id', 0),
            display=kwargs.get('display', False),
            record=kwargs.get('record', False),
            headless=kwargs.get('headless', True),
        )
        
        print("Hand-only mode: Object data disabled")
    
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Create table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015
        table_half_width = 0.4
        self._table_surface_z = table_surface_z = table_pos.z + table_half_height
        self.dexhand_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)

        # Transformation matrix
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        # Load demo data
        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))
        self.demo_dataset_dict = {}
        for dataset_type in dataset_list:
            create_kwargs = {
                "manipdata_type": dataset_type,
                "side": self.side,
                "device": self.sim_device,
                "mujoco2gym_transf": self.mujoco2gym_transf,
                "max_seq_len": self.max_episode_length,
                "dexhand": self.dexhand,
                "embodiment": self.cfg["env"]["dexhand"],
                "data_indices": self.dataIndices,
            }
            
            if dataset_type == "gigahands":
                create_kwargs["data_dir"] = "/mnt/ssd1/jisoo6687/hoi_dataset/gigahands/handpose"
                print(f"DEBUG: Setting gigahands data_dir to: {create_kwargs['data_dir']}")
            
            self.demo_dataset_dict[dataset_type] = ManipDataFactory.create_data(**create_kwargs)

        # Load dexhand asset
        dexhand_asset_file = self.dexhand.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        dexhand_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_asset_file), asset_options)
        
        # DOF properties
        dexhand_dof_stiffness = torch.tensor([500] * self.dexhand.n_dofs, dtype=torch.float, device=self.sim_device)
        dexhand_dof_damping = torch.tensor([30] * self.dexhand.n_dofs, dtype=torch.float, device=self.sim_device)
        
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        # Set friction properties
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_props_asset)

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        # DOF properties setup
        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]
            
            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self._dexhand_effort_limits = torch.tensor(self._dexhand_effort_limits, device=self.sim_device)
        self._dexhand_dof_speed_limits = torch.tensor(self._dexhand_dof_speed_limits, device=self.sim_device)

        # Create environments and actors
        num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        num_dexhand_shapes = self.gym.get_asset_rigid_shape_count(dexhand_asset)

        self.dexhands = []
        self.envs = []

        # Load and pack demo data
        def segment_data(k):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]
            return self.demo_dataset_dict[ManipDataFactory.dataset_type(idx)][idx]

        self.demo_data = [segment_data(i) for i in tqdm(range(self.num_envs))]
        self.demo_data = self.pack_data(self.demo_data)

        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            
            # Create dexhand actor
            dexhand_actor = self.gym.create_actor(
                env_ptr, dexhand_asset, self.dexhand_pose, "dexhand", i, 
                (1 if self.dexhand.self_collision else 0)
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_actor, dexhand_dof_props)

            # Create table
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i + self.num_envs, 0b11)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            self.envs.append(env_ptr)
            self.dexhands.append(dexhand_actor)

        # Setup data
        self.init_data()
    
    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        dexhand_handle = self.gym.find_actor_handle(env_ptr, "dexhand")
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        self.dexhand_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand.body_names
        }
        
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self.dexhand_root_state = self._root_state[:, dexhand_handle, :]

        # Initialize controls and forces
        self.apply_forces = torch.zeros((self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float)
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.use_pid_control:
            self.prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_dexhand_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        # Set observation space size (dof_pos + cos + sin + base_state)
        self.num_obs = self.num_dexhand_dofs * 3 + 13  # q + cos(q) + sin(q) + base_state
        
        # Initialize states dictionary
        self.states = {}
        
        # Store initial root state for resets
        self._initial_root_state = self._root_state.clone()
        
        # Allocate buffers
        self.allocate_buffers()
        
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # Refresh tensors  
        self._refresh()
        
    def allocate_buffers(self):
        # Observation and action buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def _update_states(self):
        self.states.update({
            "q": self._q[:, :self.num_dexhand_dofs],
            "q_cos": torch.cos(self._q[:, :self.num_dexhand_dofs]),
            "q_sin": torch.sin(self._q[:, :self.num_dexhand_dofs]),
            "dq": self._qd[:, :self.num_dexhand_dofs],
            "base_state": self._base_state,
        })

    def compute_reward(self, actions):
        # Simplified hand-only reward (basic survival reward for now)
        self.rew_buf.fill_(1.0)  # Basic reward for staying alive
        self.extras = {}

    def compute_observations(self):
        # Hand-only observations
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Basic hand state
        dof_pos = self._q[:, :self.num_dexhand_dofs]
        dof_vel = self._qd[:, :self.num_dexhand_dofs]
        base_state = self._base_state

        # Create observation
        self.obs_buf = torch.cat([
            dof_pos,
            torch.cos(dof_pos),
            torch.sin(dof_pos),  
            base_state,
        ], dim=-1)
        
        return self.obs_buf

    def reset_idx(self, env_ids):
        # Reset environments
        # Use global dexhand indices like the original code
        dexhand_multi_env_ids_int32 = self._global_dexhand_indices[env_ids].flatten()
        
        # Reset DOF states
        dof_pos = torch.zeros_like(self._q[env_ids])
        dof_vel = torch.zeros_like(self._qd[env_ids])
        
        # Set random hand pose
        if self.random_state_init:
            dof_pos[:, :self.num_dexhand_dofs] = torch.rand_like(dof_pos[:, :self.num_dexhand_dofs]) * 0.5
            
        self._q[env_ids] = dof_pos
        self._qd[env_ids] = dof_vel

        # Reset root state  
        self._root_state[env_ids] = self._initial_root_state[env_ids]
        
        # Apply changes using the correct indices
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32)
        )
        
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32)
        )

        # Reset buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone()
        
        # Apply actions as position targets
        targets = self.actions * 0.1  # scale actions
        targets += self._q[:, :self.num_dexhand_dofs]  # relative to current position
        
        # Clamp to joint limits
        targets = torch.clamp(targets, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)
        
        self._pos_control[:, :self.num_dexhand_dofs] = targets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):
        self.progress_buf += 1
        self._refresh()
        self._update_states()
        
        # Check for resets
        self.reset_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def step(self, actions):
        # Apply actions
        self.pre_physics_step(actions)
        
        # Simulate physics
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        # Post-physics
        self.post_physics_step()
        
        # Compute observations and rewards
        self.compute_observations()
        self.compute_reward(actions)
        
        # Reset finished episodes
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def __post_init__(self):
        """Override post-initialization to skip object-related setup"""
        # Skip BPS initialization since we don't have object data
        print("Hand-only mode: Skipping object BPS initialization")
        
        # If you still want to keep some BPS functionality, initialize with dummy data
        # self.bps_feat_type = "dists"
        # self.bps_layer = bps_torch(...)
        # self.obj_bps = torch.zeros((self.num_envs, 128), device=self.device)  # dummy
        
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        # Refresh tensors
        self._refresh()

    def pack_data(self, data):
        """Modified pack_data without object dependency"""
        packed_data = {}
        
        # Safely get sequence length from available data
        seq_lengths = []
        for d in data:
            if d is None:
                seq_lengths.append(0)
                continue
                
            # Try different keys to get sequence length
            if "wrist_pos" in d and d["wrist_pos"] is not None:
                seq_lengths.append(len(d["wrist_pos"]))
            elif "motion_length" in d:
                seq_lengths.append(d["motion_length"])
            elif "mano_joints" in d and isinstance(d["mano_joints"], dict):
                # Get length from first available mano joint
                for joint_key, joint_data in d["mano_joints"].items():
                    if joint_data is not None:
                        seq_lengths.append(len(joint_data))
                        break
                else:
                    seq_lengths.append(self.max_episode_length)  # fallback
            else:
                seq_lengths.append(self.max_episode_length)  # fallback
                
        packed_data["seq_len"] = torch.tensor(seq_lengths, device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            return torch.stack(stack_data).squeeze()

        # Filter out None data entries
        valid_data = [d for d in data if d is not None]
        
        if not valid_data:
            # If no valid data, create minimal packed_data
            packed_data["seq_len"] = torch.tensor([0], device=self.device)
            return packed_data
            
        for k in valid_data[0].keys():
            if "alt" in k:
                continue
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in valid_data:
                    if k in d and d[k] is not None and isinstance(d[k], dict):
                        joint_tensors = []
                        for j_name in self.dexhand.body_names:
                            hand_joint = self.dexhand.to_hand(j_name)[0]
                            if hand_joint != "wrist" and hand_joint in d[k]:
                                joint_tensors.append(d[k][hand_joint])
                        mano_joints.append(torch.cat(joint_tensors, dim=-1))

                if mano_joints:
                    packed_data[k] = fill_data(mano_joints)
            elif k in valid_data[0] and type(valid_data[0][k]) == torch.Tensor:
                stack_data = [d[k] if k in d and d[k] is not None else torch.zeros_like(valid_data[0][k]) for d in valid_data]
                # Skip object-related tensors
                if k not in ["obj_verts", "obj_trajectory"]:
                    packed_data[k] = fill_data(stack_data)
            elif k in valid_data[0] and type(valid_data[0][k]) == dict:
                # Handle nested dictionaries (like mano_joints)
                nested_dict = {}
                for sub_key in valid_data[0][k].keys():
                    stack_data = []
                    for d in valid_data:
                        if k in d and d[k] is not None and sub_key in d[k] and d[k][sub_key] is not None:
                            stack_data.append(d[k][sub_key])
                    if stack_data:
                        nested_dict[sub_key] = fill_data(stack_data)
                if nested_dict:
                    packed_data[k] = nested_dict
            else:
                # Handle other data types safely
                if k in valid_data[0]:
                    packed_data[k] = [d[k] if k in d else None for d in valid_data]

        def to_cuda(x):
            if type(x) == torch.Tensor:
                return x.to(self.device)
            elif type(x) == list:
                return [to_cuda(xx) for xx in x]
            elif type(x) == dict:
                return {k: to_cuda(v) for k, v in x.items()}
            else:
                return x

        packed_data = to_cuda(packed_data)
        return packed_data


class DexHandImitatorHandOnlyLHEnv(DexHandImitatorHandOnlyRHEnv):
    """Left hand version of hand-only DexHandImitator"""
    side = "left"
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs) 