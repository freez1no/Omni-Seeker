from __future__ import annotations
from collections.abc import Sequence
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.sim.spawners import spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera 
from ultralytics import YOLO
from .yolo_env_cfg import yoloEnvCfg


class yoloEnv(DirectRLEnv):
    cfg: yoloEnvCfg

    def __init__(self, cfg: yoloEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(f"Loading YOLOv11 model 'yolo11n.pt' on device {self.device}...")
        self.yolo_model = YOLO("yolo11n.pt") 
        self.yolo_model.to(self.device)
        print("YOLO model loaded successfully.")
        self.bbox_obs = torch.zeros((self.num_envs, 4), device=self.device)
        self.prev_h_norm = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.target = RigidObject(self.cfg.target_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())
        self.cfg.camera_cfg.num_envs = self.cfg.scene.num_envs
        camera_sensor = Camera(self.cfg.camera_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target 
        self.scene.sensors["camera"] = camera_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        image_data_rgba = self.scene.sensors["camera"].data.output["rgb"]
        images_rgb_nhwc = image_data_rgba[..., :3]
        images_rgb_bchw_float = images_rgb_nhwc.permute(0, 3, 1, 2).float() / 255.0

        results = self.yolo_model(
            images_rgb_bchw_float, 
            classes=[32], # sports ball
            verbose=False, 
            device=self.device
        )

        obs_tensor = torch.zeros((self.num_envs, 4), device=self.device)
        for i, res in enumerate(results):
            if res.boxes.shape[0] > 0:
                obs_tensor[i] = res.boxes.xywhn[0]
        
        self.bbox_obs = obs_tensor
        return {"policy": self.bbox_obs}

    def _get_rewards(self) -> torch.Tensor:
        x_norm, _, w_norm, h_norm = torch.split(self.bbox_obs, 1, dim=-1)
        x_norm = x_norm.squeeze(-1) # [B]
        w_norm = w_norm.squeeze(-1) # [B]
        h_norm = h_norm.squeeze(-1) # [B]

        found_target = (w_norm > 0).float()
        center_error = torch.abs(x_norm - 0.5)
        centering_reward = (0.5 - center_error) * 2.0 * found_target
        optimal_h = 0.5
        distance_error_sq = (h_norm - optimal_h) ** 2
        distance_reward = torch.exp(-20.0 * distance_error_sq) * found_target
        delta_h = h_norm - self.prev_h_norm
        worsening_penalty = torch.clamp(delta_h, max=0.0) * -1.0 * found_target
        too_close_penalty = (h_norm > 0.8).float() * -2.0
        total_reward = (centering_reward * 1.0) + \
                       (distance_reward * 1.5) + \
                       (worsening_penalty * 0.5) + \
                       too_close_penalty
        
        self.prev_h_norm = h_norm.detach()
        return total_reward.reshape(-1, 1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        dones = torch.zeros_like(time_out)
        return dones, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids) 
        if env_ids is None:
            reset_env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            reset_env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            reset_env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            
        num_resets = len(reset_env_ids_tensor)
        robot_new_state = torch.zeros((num_resets, self.robot.data.default_root_state.shape[1]), device=self.device)
        robot_new_state[:, 2] = 0.05 # 지면에서 5cm 띄우기
        robot_new_state[:, 3] = 1.0  # 기본 방향 (W=1)
        robot_new_state[:, :3] += self.scene.env_origins[reset_env_ids_tensor]

        self.robot.write_root_state_to_sim(robot_new_state, reset_env_ids_tensor)
        default_joint_pos = self.robot.data.default_joint_pos[reset_env_ids_tensor] * 0.0
        default_joint_vel = self.robot.data.default_joint_vel[reset_env_ids_tensor] * 0.0
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, reset_env_ids_tensor)
        target_new_state = torch.zeros((num_resets, self.target.data.default_root_state.shape[1]), device=self.device)
        random_angle = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        distance = 0.20 # 20cm
        rel_x = torch.cos(random_angle) * distance
        rel_y = torch.sin(random_angle) * distance

        target_new_state[:, 0] = robot_new_state[:, 0] + rel_x
        target_new_state[:, 1] = robot_new_state[:, 1] + rel_y
        target_new_state[:, 2] = robot_new_state[:, 2] + 0.20 # Jetbot보다 20cm 위

        # 기본 방향 (W=1)
        target_new_state[:, 3] = 1.0 
        # 속도는 0
        target_new_state[:, 7:13] = 0.0 
        self.target.write_root_state_to_sim(target_new_state, reset_env_ids_tensor)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        scaled_actions = self.actions * 2.5 
        self.robot.set_joint_velocity_target(scaled_actions, joint_ids=self.dof_idx)