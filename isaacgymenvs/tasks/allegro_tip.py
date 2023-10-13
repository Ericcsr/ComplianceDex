# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, torch_rand_float, tensor_clamp, quat_apply, quat_rotate, quat_rotate_inverse
from isaacgymenvs.tasks.base.vec_task import VecTask
import open3d as o3d
from torchsdf import compute_sdf

# TODO: remove it once actual tip pose is available
TIP_POSES = [[-0.0375,0.0, 0.0325], [0.0375, -0.03, 0.0325], [0.0375, 0.0, 0.0325], [0.0375, 0.03, 0.0325]]
TAR_POSES = [[-0.02,0.0, 0.0325], [0.02, -0.03, 0.0325], [0.02, 0.0, 0.0325], [0.02, 0.03, 0.0325]]
TIP_Z_OFFSET = 0.1
DEFAULT_COMPLIANCE = 50.0
IDLE_POSES = [[-0.15, 0.15, 0.15], [0.15, -0.15, 0.15], [0.15, 0.0, 0.15], [0.15, 0.15, 0.15]]

class AllegroTip(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, 
                 force_render, tip_poses = TIP_POSES, idle_poses = IDLE_POSES, tar_poses = TAR_POSES, debug=False):

        self.cfg = cfg
        self.debug = debug

        self.aggregate_mode = self.cfg["env"]["aggregateMode"] # TODO: We may need to keep inter fingertip

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.target_penalty_scale = self.cfg["env"]["targetPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.num_fingers = self.cfg["env"]["numFingers"]
        self.init_tips = torch.tensor(tip_poses, device=sim_device)
        self.init_tips[:,2] += TIP_Z_OFFSET
        self.idle_tips = torch.tensor(idle_poses, device=sim_device)
        self.idle_tips[:,2] += TIP_Z_OFFSET
        self.tar_poses = torch.tensor(tar_poses, device=sim_device)
        self.tar_poses[:,2] += TIP_Z_OFFSET

        self.idle_tip_pose= self.idle_tips.repeat(self.cfg["env"]["numEnvs"],1,1)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)
        self.max_substeps = self.cfg["env"].get("maxSubsteps", 100)
        self.contact_force_thresh = self.cfg["env"].get("contactForceThresh", 0.5) # At least 0.5 N force?
        self.contact_dist_thresh = self.cfg["env"].get("contactDistThresh", 0.015)
        self.friction_mu = self.cfg["env"].get("frictionMu", 0.5)
        self.preparation_steps = self.cfg["env"].get("preparationStep", 10)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])
        # Load 3D objects
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        object_mesh_path = self.cfg["env"].get("objectMeshPath", "urdf/objects/meshes/cube_multicolor.obj")
        object_mesh = o3d.io.read_triangle_mesh(asset_root+"/"+object_mesh_path).scale(0.065,center=[0,0,0])
        triangles = np.asarray(object_mesh.triangles)
        vertices = np.asarray(object_mesh.vertices)
        self.face_vertices = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles), 3, 3)).to(sim_device).float()
        print(self.face_vertices)

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "full_no_vel": 62, # TODO: Should revise
            "full": 80,
            "full_state": 92
        }

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 92

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 32

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(-0.5, -0.5, 0.3)
            cam_target = gymapi.Vec3(1.0, 1.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

             dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
             self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_allegro_tip_dofs * self.num_fingers)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.allegro_hand_default_dof_pos = torch.tensor(self.init_tips, dtype=torch.float, device=self.device).flatten()
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.compliance = torch.ones(self.num_envs, self.num_fingers, dtype=torch.float, device = self.device) * DEFAULT_COMPLIANCE

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0
        self.substep_cnt = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.prev_detach_flag = torch.zeros((self.num_envs, self.num_fingers), dtype=torch.bool, device = self.device)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        allegro_tip_asset_file = "urdf/tip.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            allegro_tip_asset_file = self.cfg["env"]["asset"].get("assetFileName", allegro_tip_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        allegro_tip_asset = self.gym.load_asset(self.sim, asset_root, allegro_tip_asset_file, asset_options)

        #self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        #self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_allegro_tip_dofs = self.gym.get_asset_dof_count(allegro_tip_asset) # Should be 3
        self.num_allegro_hand_dofs = self.num_allegro_tip_dofs * self.num_fingers
        print("Num dofs: ", self.num_allegro_tip_dofs)
        self.num_shadow_hand_actuators = self.num_allegro_hand_dofs

        self.actuated_dof_indices = [i for i in range(self.num_allegro_hand_dofs)]

        # set shadow_hand dof properties
        allegro_tip_dof_props = self.gym.get_asset_dof_properties(allegro_tip_asset)

        self.allegro_tip_dof_lower_limits = []
        self.allegro_tip_dof_upper_limits = []
        self.allegro_tip_dof_default_pos = []
        self.allegro_tip_dof_default_vel = []
        self.sensors = []

        for i in range(self.num_allegro_tip_dofs):
            self.allegro_tip_dof_lower_limits.append(allegro_tip_dof_props['lower'][i])
            self.allegro_tip_dof_upper_limits.append(allegro_tip_dof_props['upper'][i])
            self.allegro_tip_dof_default_pos.append(0.0)
            self.allegro_tip_dof_default_vel.append(0.0)

            print("Max effort: ", allegro_tip_dof_props['effort'][i])
            allegro_tip_dof_props['stiffness'][i] = 3
            allegro_tip_dof_props['damping'][i] = 0.3
            allegro_tip_dof_props['armature'][i] = 0.001

        # limits and default pos for allegro hand
        self.allegro_hand_dof_lower_limits = self.allegro_tip_dof_lower_limits * self.num_fingers
        self.allegro_hand_dof_upper_limits = self.allegro_tip_dof_upper_limits * self.num_fingers
        self.allegro_hand_dof_default_pos = self.allegro_tip_dof_default_pos * self.num_fingers
        self.allegro_hand_dof_default_vel = self.allegro_tip_dof_default_vel* self.num_fingers

        # limits for compliance
        self.compliance_lb = torch.ones(self.num_fingers, device=self.device) * 10.0
        self.compliance_ub = torch.ones(self.num_fingers, device=self.device) * 100.0

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device) # For each tip
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)
        self.allegro_hand_dof_default_pos = to_torch(self.allegro_hand_dof_default_pos, device=self.device)
        self.allegro_hand_dof_default_vel = to_torch(self.allegro_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        allegro_tip_start_poses = []
        for i in range(self.num_fingers):
            allegro_tip_start_pose = gymapi.Transform()
            allegro_tip_start_pose.p = gymapi.Vec3(*self.init_tips[i])
            allegro_tip_start_pose.r = gymapi.Quat(1, 0, 0, 0)
            allegro_tip_start_poses.append(allegro_tip_start_pose)

        # Object should be placed at the frame center
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0, 0, 0.0325+TIP_Z_OFFSET)

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        self.prev_object_pose  = torch.zeros(num_envs, 7, device=self.device)

        self.init_target_poses = self.tar_poses.repeat(num_envs,1,1)
        self.prev_target_poses = self.init_target_poses.clone()

        
        self.allegro_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(allegro_tip_asset) * self.num_fingers
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(allegro_hand_rb_count, allegro_hand_rb_count + object_rb_count)) # TODO: May be wrong

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            allegro_tip_actors = []
            hand_idx = []
            hand_start_state = []
            for fid in range(self.num_fingers):
                allegro_tip_start_pose = allegro_tip_start_poses[fid]
                # TODO: May be segmentation id is necessary
                default_tip_tf = gymapi.Transform()
                default_tip_tf.p = gymapi.Vec3(0, 0, 0)
                default_tip_tf.r = gymapi.Quat(1, 0, 0, 0)
                allegro_tip_actor = self.gym.create_actor(env_ptr, allegro_tip_asset, default_tip_tf, f"tip:{fid}", group=i, filter=0)
                hand_state = [allegro_tip_start_pose.p.x, allegro_tip_start_pose.p.y, allegro_tip_start_pose.p.z,
                              allegro_tip_start_pose.r.x, allegro_tip_start_pose.r.y, allegro_tip_start_pose.r.z, allegro_tip_start_pose.r.w,
                              0, 0, 0, 0, 0, 0]
                self.gym.set_actor_dof_states(env_ptr, allegro_tip_actor, hand_state[0:3] ,gymapi.STATE_POS)
                hand_start_state.append(hand_state)
                #self.prev_target_poses[i,fid] = torch.tensor(hand_state[0:3], dtype=torch.float, device=self.device)
                self.gym.set_actor_dof_properties(env_ptr, allegro_tip_actor, allegro_tip_dof_props)
                allegro_tip_actors.append(allegro_tip_actor)
                tip_idx = self.gym.get_actor_index(env_ptr, allegro_tip_actor, gymapi.DOMAIN_SIM)
                hand_idx.append(tip_idx)
            self.hand_indices.append(hand_idx)
            self.hand_start_states.append(hand_start_state)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            shape_props[0].friction = 0.5
            shape_props[0].rolling_friction = 0.5
            shape_props[0].torsion_friction = 0.5
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, shape_props)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            # NOTE: We shouldn't allow inter fingertip collision.

            self.envs.append(env_ptr)
            self.allegro_hands = self.allegro_hands + allegro_tip_actors

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 4, 13)

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )
        compliance_penalty = torch.sum(actions[:,28:] ** 2, dim=-1) # Penalize being stiff
        pd_target = actions[:,12:24].reshape(-1,3)
        pd_target_local = quat_rotate_inverse(self.object_rot.repeat(1,4).view(-1,4), pd_target - self.object_pos.repeat(1,4).view(-1,3))
        dist, sign, _, _ = compute_sdf(pd_target_local * 10.0, self.face_vertices * 10.0)
        sdist = torch.sqrt(dist)/10.0 * sign
        pd_target_penalty = torch.sum(sdist.view(-1,4), dim=-1) # Encourage PD target stay inside/close to the object
        self.rew_buf += compliance_penalty * self.action_penalty_scale + pd_target_penalty * self.target_penalty_scale

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
             self.compute_full_state()
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_observations(self, no_vel=False):
        action_dim = self.num_fingers * 8 # ACTION: [init_pos, target_pos, detach_flag, k_p]
        if no_vel: # 62
            self.obs_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos, # TODO: change shadow_hand
                                                                   self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
            offset = self.num_allegro_hand_dofs
            self.obs_buf[:, offset:offset+7] = self.object_pose
            offset = offset + 7
            self.obs_buf[:, offset:offset+7] = self.goal_pose
            offset = offset + 7
            self.obs_buf[:, offset:offset+4] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            offset = offset + 4
            self.obs_buf[:, offset:offset+action_dim] = self.actions
        else: # 80
            self.obs_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos, # TODO: change shadow_hand
                                                                   self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
            self.obs_buf[:, self.num_allegro_hand_dofs:2*self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel # TODO: change shadow_hand

            offset = 2*self.num_allegro_hand_dofs
            self.obs_buf[:, offset:offset+7] = self.object_pose
            offset += 7
            self.obs_buf[:, offset:offset+3] = self.object_linvel
            offset += 3
            self.obs_buf[:, offset:offset+3] = self.vel_obs_scale * self.object_angvel
            offset += 3

            self.obs_buf[:, offset:offset+7] = self.goal_pose
            offset += 7
            self.obs_buf[:, offset:offset+4] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            offset += 4

            self.obs_buf[:, offset:offset+action_dim] = self.actions

    def compute_full_state(self, asymm_obs=False):
        action_dim = self.num_fingers * 8
        if asymm_obs:
            self.states_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos, # TODO: change shadow_hand
                                                                      self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
            self.states_buf[:, self.num_allegro_hand_dofs:2*self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel # TODO: change shadow_hand
            self.states_buf[:, 2*self.num_allegro_hand_dofs:3*self.num_allegro_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor # TODO: Is it necessary, or use PD error

            obj_obs_start = 3*self.num_allegro_hand_dofs  # 36
            self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 49
            self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            fingertip_obs_start = goal_obs_start + 11  # 60

            # obs_total = obs_end + num_actions = 60 + 32 = 92
            obs_end = fingertip_obs_start
            self.states_buf[:, obs_end:obs_end + action_dim] = self.actions
        else:
            self.obs_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos, # TODO: change shadow_hand
                                                                      self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
            self.obs_buf[:, self.num_allegro_hand_dofs:2*self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel # TODO: change shadow_hand
            self.obs_buf[:, 2*self.num_allegro_hand_dofs:3*self.num_allegro_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor # TODO: change shadow_hand

            obj_obs_start = 3*self.num_allegro_hand_dofs  # 36
            self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 49
            self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            fingertip_obs_start = goal_obs_start + 11  # 60

            # obs_total = obs_end + num_actions = 60 + 32 = 92
            obs_end = fingertip_obs_start #+ num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end:obs_end + action_dim] = self.actions

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])
        if self.debug:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Target pose reset!")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        #self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] #+ \
            #self.reset_position_noise * rand_floats[:, 0:2]
        #self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] #+ \
            #self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        #new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        #new_object_rot = randomize_rotation(rand_floats[:,3]*0.0, rand_floats[:,4]*0.0, self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            #new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
            #                                        self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        self.prev_object_pose[env_ids, 3:7] = self.root_state_tensor[self.object_indices[env_ids],3:7]
        self.prev_object_pose[env_ids, 0:3] = self.root_state_tensor[self.object_indices[env_ids],0:3]
        self.prev_target_poses[env_ids] = self.init_target_poses[env_ids]
        self.compliance[env_ids] = torch.ones(len(env_ids), self.num_fingers, dtype=torch.float, device = self.device) * DEFAULT_COMPLIANCE
        if self.debug:
            print("Compliance:", self.compliance)

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        # TODO: Some bug hidden here
        delta_max = self.allegro_hand_dof_upper_limits - self.allegro_hand_dof_default_pos
        delta_min = self.allegro_hand_dof_lower_limits - self.allegro_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5+self.num_allegro_hand_dofs] + 1)

        pos = self.allegro_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta * 0.0
        self.allegro_hand_dof_pos[env_ids, :] = pos
        #self.prev_target_poses[env_ids] = pos.view(len(env_ids), self.num_fingers, 3) # Should be with all fingers
        self.allegro_hand_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        # NOTE: dof_state cannot be changed
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.prev_detach_flag[env_ids] = self.prev_detach_flag[env_ids].fill_(False)
        if self.debug:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("System reset!")
            #print("Hand indices:",hand_indices, self.dof_state, pos)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # TODO: Clarify the finite state machine
    # TODO: Define gapping mechanism
    # update action: 0: intermediate, 1: received action, preparing, 2: execute
    # action could be non if update action = 0, 2
    def pre_physics_step(self, actions=None, substep_id=0):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            if self.debug:
                print("Env ids:", env_ids)
            self.reset_idx(env_ids, goal_env_ids)

        object_pose = self.root_state_tensor[self.object_indices, 0:7]
        # things needed regardless of stage
        # New action start
        # Action: [[init_poses, target_poses, detach_flags, compliances],..] between -1 to 1
        # Alternating stepping:
        # 1). First step determine which finger to detach;
        # 2). Second step determine which finger to compliantly attach to where
        if actions is not None: # INSTANT
            self.update_action=False
            self.actions = actions.clone().to(self.device)
            offset = 2 * self.num_dofs + self.num_fingers
            self.init_poses = scale(self.actions[:,:self.num_dofs], 
                            self.allegro_hand_dof_lower_limits[self.actuated_dof_indices],
                            self.allegro_hand_dof_upper_limits[self.actuated_dof_indices]).view(-1, self.num_fingers, 3)

            self.target_poses = scale(self.actions[:,self.num_dofs:2*self.num_dofs], 
                            self.allegro_hand_dof_lower_limits[self.actuated_dof_indices],
                            self.allegro_hand_dof_upper_limits[self.actuated_dof_indices]).view(-1, self.num_fingers, 3)
            total_prev_detach = self.prev_detach_flag.sum(dim=-1).bool() # whether there are detached fingers previously [num_envs,]
            # If total_prev_detach, current must all attach
            # If not total_prev_detach, current can have at most one detach
            # GPT_WARN: may be buggy or slow
            detach_signal = self.actions[:,2 * self.num_dofs:2 * self.num_dofs+self.num_fingers]
            if self.debug:
                print("Raw detach:", detach_signal)
            max_vals, max_indices = torch.max(detach_signal, dim=1, keepdim=True)
            mask_greater_than_zero = max_vals > 0.0
            self.detach_flag = torch.zeros_like(detach_signal, dtype=torch.bool, device=self.device) # [num_envs, num_fingers]
            self.detach_flag.scatter_(1, max_indices, mask_greater_than_zero)
            self.detach_flag[total_prev_detach] = self.detach_flag[total_prev_detach].fill_(False)

            # If this is new detachment, check whether it will violate friction cone constraint in equilibrium
            mask = (~self.prev_detach_flag) * self.detach_flag
            total_new_detach = mask.sum(dim=-1).bool()
            active_envs = total_new_detach.sum()
            if active_envs != 0:
                # Tip target of relavant fingers
                tip_in_contact_if_detach = ~mask[total_new_detach]
                prev_tip_tar = self.prev_target_poses[total_new_detach][tip_in_contact_if_detach] # should have 3 contact points
                cur_tip_pose = self.allegro_hand_dof_pos.view(self.num_envs, self.num_fingers, 3)[total_new_detach][tip_in_contact_if_detach] # should have 3 contact points
                cur_tip_norm = self.get_surface_normal(cur_tip_pose, object_pose[total_new_detach])
                infeasible_grasps = force_eq_check(cur_tip_pose.view(active_envs,-1,3), 
                                                prev_tip_tar.view(active_envs,-1,3), 
                                                self.compliance[total_new_detach][tip_in_contact_if_detach].view(-1,3),
                                                self.friction_mu, 
                                                cur_tip_norm.view(active_envs,-1,3))
                
                merged_mask = torch.zeros_like(total_new_detach, dtype=torch.bool)
                merged_mask[total_new_detach] = infeasible_grasps
                self.detach_flag[merged_mask] = self.detach_flag[merged_mask].fill_(False)
            self.detach_flag[env_ids] = self.detach_flag[env_ids].fill_(False) # Just resetted environment cannot detach
            if self.debug:
                    print("Infeasible, detachment:",infeasible_grasps)

            self.actual_target_pose = self.idle_tip_pose.clone() # default pose
            
            
            # previously detached, currenly undetached
            mask = self.prev_detach_flag * (~self.detach_flag)
            if self.debug:
                print("Prev and current:",self.prev_detach_flag, self.detach_flag, env_ids)
                #self.print_state(mask, self.init_poses, self.target_poses)
            self.actual_target_pose[mask] = self.init_poses[mask]
            self.undetached_mask = (~self.prev_detach_flag) * (~self.detach_flag)
            self.compliance[(~self.prev_detach_flag) * self.detach_flag] = 1.0
            self.target_start = None
            
        elif self.update_action: # INSTANT: Action execution, assume self.action unchanged
            offset = 2 * self.num_dofs + self.num_fingers
            # Using compliance from current effective action.
            self.new_detach_mask = self.prev_detach_flag * (~self.detach_flag) # TODO: Should include all fingers
            self.compliance[self.new_detach_mask] = scale(self.actions[:, offset:offset+self.num_fingers], self.compliance_lb, self.compliance_ub)[self.new_detach_mask]
            self.actual_target_pose = self.idle_tip_pose.clone()

            
            # previously detached, currently attached
            self.target_start = self.prev_target_poses[self.new_detach_mask]
            # update prev flag
            # self.prev_detach_flag = self.detach_flag # Action driven, not result driven


        # Previously undetached target_pose currently undetached # Should always moving
        if self.target_start is not None:
            interp_len = self.max_substeps//2
            self.actual_target_pose[self.new_detach_mask] = torch.lerp(self.target_start, 
                                                                       self.target_poses[self.new_detach_mask], 
                                                                       float(substep_id-interp_len)/float(interp_len))
        self.actual_target_pose[self.undetached_mask] = self.prev_target_poses[self.undetached_mask]
        # object_pose_extended = object_pose.repeat(1,self.num_fingers).view(self.num_envs, self.num_fingers, -1)[self.undetached_mask]
        # object_pose_prev_extended = self.prev_object_pose.repeat(1,self.num_fingers).view(self.num_envs, self.num_fingers, -1)[self.undetached_mask]
        # self.actual_target_pose[self.undetached_mask] = 0.5 * apply_tf(self.prev_target_poses[self.undetached_mask],
        #                                                                object_pose_extended, 
        #                                                                object_pose_prev_extended) + 0.5 * self.prev_target_poses[self.undetached_mask]
        self.cur_targets[:, self.actuated_dof_indices] = scale(self.actual_target_pose.view(self.num_envs, -1),
                                                               self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], 
                                                               self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
        self.prev_target_poses = self.actual_target_pose # Should be initialized as current target pose
        #self.prev_detach_flag = self.detach_flag 
        self.prev_object_pose = object_pose
        
        # Should use force control instead of position control
        tau = self.compute_torque(self.allegro_hand_dof_pos, 
                                  self.allegro_hand_dof_vel, 
                                  self.actual_target_pose.view(self.num_envs, -1), 
                                  self.compliance.repeat_interleave(3,dim=1))
        if self.debug and ((actions is not None) or self.update_action):
            print("Actual targets:",self.actual_target_pose.view(self.num_envs,-1))
            print("Current joint q:", self.allegro_hand_dof_pos)
            print("tau:", tau)
            print("Compliance:",self.compliance)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(tau))
        
        if self.update_action:
            self.update_action = False
        # Apply force on the object as perturbation
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    # Simulation reset should be done after a bigger step
    def post_physics_step(self, get_reward=False):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.substep_cnt += 1

        self.compute_observations()
        if self.substep_cnt % self.max_substeps == self.max_substeps / 2 and self.substep_cnt != 0:
            self.update_action = True
        if get_reward: # the action would be obsolete
            self.compute_reward(self.actions)
            self.substep_cnt = 0

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
    # Redefine stepping here
    def step(self, actions: torch.Tensor):
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions # For simplicity use fixed interval always max steps
        # Should place reset in post physics step to prevent asynchronous resetting
        # TODO: Add preparation steps
        for s in range(self.max_substeps):
            if s == 0:
                self.pre_physics_step(action_tensor,s)
            else:
                self.pre_physics_step(None,s)

            # step physics and render each frame
            for i in range(self.control_freq_inv):
                if self.force_render:
                    self.render()
                self.gym.simulate(self.sim)

            # to fix!
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

            # compute observations, rewards, resets, ...
            if s == self.max_substeps - 1:
                self.post_physics_step(get_reward=True)
                self.prev_detach_flag = self.detect_detachment(self.allegro_hand_dof_pos.view(self.num_envs,self.num_fingers,3), 
                                                               self.object_pose)
            else:
                self.post_physics_step(get_reward=False)

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def compute_torque(self,cur_pos, cur_vel, tar_pos, compliance_xyz):
        """
        cur_pos: [num_envs, num_fingers * 3]
        tar_pos: [num_envs, num_fingers * 3]
        compliance_xyz: [num_envs, num_fingers * 3]
        """
        tau = compliance_xyz * (tar_pos - cur_pos) - 0.2 * compliance_xyz * cur_vel
        return tau

    # TODO: Prev contact flag should be result driven
    # TODO: Need a better contact detection mechanism
    # TODO: Ideally we should use an SDF to compute shortest distance between tip_pose and object in realtime.
    def detect_detachment(self, tip_poses, object_pose):
        """
        Should check detachment during/after reward calculation
        tip_poses: [num_envs, num_fingers, 3]
        object_pose: [num_envs, 7]
        return:
        mask: [num_envs, num_fingers]
        """
        tip_pose_local = quat_rotate_inverse(object_pose[:,3:7].repeat(1,self.num_fingers).view(-1,4),
                                             (tip_poses - object_pose[:,:3].unsqueeze(1)).view(-1,3))
        dis, _,_,_ = compute_sdf(tip_pose_local * 10.0, self.face_vertices * 10.0)
        dis = torch.sqrt(dis.view(self.num_envs, self.num_fingers)) / 10.0
        mask = dis > self.contact_dist_thresh
        if self.debug:
            print("Distance:", dis, tip_pose_local)
        # compliance = compliance.repeat_interleave(3,dim=1).view(self.num_envs, self.num_fingers, 3)
        # F = torch.norm(compliance * (tip_targets - tip_poses), dim=-1)
        # if self.debug:
        #     print("Contact force:",F)
        # mask = (F < self.contact_force_thresh)
        return mask
        
    # In debug mode assume num_env = 1
    def print_state(self, mask, init_pose, target_pose):
        mask = mask.squeeze()
        init_pose = init_pose.squeeze()
        target_pose = target_pose.squeeze()
        print("===========================================================")
        for i in range(self.num_fingers):
            if mask[i]:
                print(f"Finger {i} attaching, init: {init_pose[i]}, tar: {target_pose[i]}")
        print("===========================================================")

    # get surface normal in world coordinate
    # Assume all points are making contact, if not enough fill it in with dummy variables
    # Should call get_surface before detaching fingers Use sdf
    def get_surface_normal(self, points, object_poses):
        """
        points: [N * 3, 3] , in case of allegro hand 3, 4
        object_poses: [N, 7]
        """
        object_poses_extend = object_poses.repeat_interleave(3, dim=0)
        vec = points - object_poses_extend[:,0:3]
        points_local = quat_rotate_inverse(object_poses_extend[:,3:7], vec)
        _,_,normals_local,_ = compute_sdf(points_local * 10.0, self.face_vertices * 10.0)
        return quat_rotate(object_poses_extend[:,3:7], 
                           normals_local.view(-1,3)).view(object_poses.shape[0], -1, 3)


#####################################################################
###=========================jit functions=========================###
#####################################################################

# TODO: define the grasp criteria and revise reward term
@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # action_penalty = torch.sum(actions**2, dim=-1)
    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew# + action_penalty * action_penalty_scale # Should be negative

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(object_pos[:,2] <= 0.05, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[:,2] <= 0.05, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

# Should output new pos in world coordinate
@torch.jit.script
def apply_tf(vectors, new_pose, prev_pose):
    rel_pos = quat_rotate_inverse(prev_pose[:,3:7], vectors - prev_pose[:,:3])
    delta_quat = quat_mul(new_pose[:,3:7], quat_conjugate(prev_pose[:,3:7]))
    rel_pos_new = quat_rotate_inverse(delta_quat, rel_pos)
    pos_new = quat_rotate(new_pose[:,3:7], rel_pos_new) + new_pose[:,0:3]
    return pos_new

# TODO: May need safe guard this alittle bit
@torch.jit.script
def optimal_transformation_batch(S1, S2, weight):
    """
    S1: [num_envs, num_points, 3]
    S2: [num_envs, num_points, 3]
    weight: [num_envs, num_points]
    """
    weight = weight.unsqueeze(2) # [num_envs, num_points, 1]
    c1 = (weight * S1).mean(dim=1).unsqueeze(1) # [num_envs, num_points]
    c2 = (weight * S2).mean(dim=1).unsqueeze(1)
    H = (weight * (S1 - c1)).transpose(1,2) @ (weight * (S2 - c2))
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.mH
    R = V @ U.transpose(1,2)
    t = (weight * (S2 - (R@S1.transpose(1,2)).transpose(1,2))).sum(dim=1) / weight.sum(dim=1)
    return R, t

# Assume friction is uniform
# Differentiable
def force_eq_reward(tip_pose, target_pose, compliance, friction_mu, current_normal):
    """
    Params:
    tip_pose: world frame [num_envs, num_fingers, 3]
    target_pose: world frame [num_envs, num_fingers, 3]
    compliance: [num_envs, num_fingers]
    friction_mu: scaler
    current_normal: world frame [num_envs, num_fingers, 3]
    
    Returns:
    reward: [num_envs]
    """
    R,t = optimal_transformation_batch(tip_pose, target_pose, compliance) # t: [num_envs, 3]
    tip_pose_eq = (R@tip_pose.transpose(1,2)).transpose(1,2) + t.unsqueeze(1)
    diff_vec = tip_pose_eq - target_pose # [num_envs, num_fingers, 3]
    dir_vec = diff_vec / diff_vec.norm(dim=2).unsqueeze(2)
    # Rotate local norm to final
    normal_eq = (R @ current_normal.transpose(1,2)).transpose(1,2)
    # measure cos between
    ang_diff =  torch.einsum("ijk,ijk->ij",dir_vec, normal_eq)
    margin = (ang_diff - torch.sqrt(1/(1+torch.tensor(friction_mu)**2))).clamp(min=-0.99)
    # we hope margin to be as large as possible, never below zero
    return torch.log(margin+1).sum(dim=1)

def force_eq_check(tip_pose, target_pose, compliance, friction_mu, current_normal):
    """
    Params:
    tip_pose: world frame [num_envs, num_fingers, 3]
    target_pose: world frame [num_envs, num_fingers, 3]
    compliance: [num_envs, num_fingers]
    friction_mu: scaler
    current_normal: world frame [num_envs, num_fingers, 3]
    
    Returns:
    feasible grasps: [num_envs] bool
    """
    R,t = optimal_transformation_batch(tip_pose, target_pose, compliance) # t: [num_envs, 3]
    tip_pose_eq = (R@tip_pose.transpose(1,2)).transpose(1,2) + t.unsqueeze(1)
    diff_vec = tip_pose_eq - target_pose # [num_envs, num_fingers, 3]
    dir_vec = diff_vec / (diff_vec.norm(dim=2)+1e-7).unsqueeze(2) # prevent nan
    # Rotate local norm to final
    normal_eq = (R @ current_normal.transpose(1,2)).transpose(1,2)
    # measure cos between
    ang_sim =  torch.einsum("ijk,ijk->ij",dir_vec, normal_eq)
    infeasible_flag = (ang_sim < torch.sqrt(1/(1+torch.tensor(friction_mu)**2)))
    return infeasible_flag.sum(dim=-1).bool()
