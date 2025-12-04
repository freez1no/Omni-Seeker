# exts/jetbot_yolo/jetbot_yolo/tasks/direct/jetbot_yolo/jetbot_yolo_env.py

import torch
import gymnasium as gym
import numpy as np
import cv2
from ultralytics import YOLO

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.sim import spawn_ground_plane, GroundPlaneCfg, DomeLightCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import wrap_to_pi
import isaaclab.utils.math as math_utils

from .jetbot_yolo_env_cfg import JetbotYoloEnvCfg

class JetbotYoloEnv(DirectRLEnv):
    cfg: JetbotYoloEnvCfg

    def __init__(self, cfg: JetbotYoloEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # YOLOv8 모델 로딩
        print("[INFO] Loading YOLOv8 model...")
        self.yolo_model = YOLO("yolov8n.pt") 
        
        # 로봇 관절 인덱스 찾기
        self.wheel_dof_idx, _ = self.robot.find_joints(".*wheel_joint")
        
        # 런타임 변수 초기화
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        
        # YOLO 결과 저장용 (obs)
        # [center_x_norm, center_y_norm, area_norm, flag] = 4
        self.yolo_obs = torch.zeros((self.num_envs, 4), device=self.device)

    def _setup_scene(self):
        # Ground Plane, light
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # create Jetbot
        robot_cfg = self.cfg.robot_cfg.copy()
        robot_cfg.prim_path = f"{self.scene.env_regex_ns}/Robot"
        
        self.robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self.robot
        
        # target object
        target_cfg = self.cfg.target_object.copy()
        target_cfg.prim_path = f"{self.scene.env_regex_ns}/Target"
        
        self.target = RigidObject(target_cfg)
        self.scene.rigid_objects["target"] = self.target

        # camera sensor
        camera_cfg = self.cfg.tiled_camera.copy()
        camera_cfg.prim_path = f"{self.scene.env_regex_ns}/Robot/chassis/rgb_camera/camera"
        
        self.camera = Camera(camera_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        velocity_command = self.actions * 10.0
        self.robot.set_joint_velocity_target(velocity_command, joint_ids=self.wheel_dof_idx)

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        # camera data load
        # shape: (num_envs, height, width, 4) - RGBA
        self.camera.update(dt=self.cfg.sim.dt)
        rgba_images = self.camera.data.output["rgb"]
        
        # YOLO model inference
        # 학습 속도를 위해 num_envs=1일 때만 카메라 창을 띄우기
        for i in range(self.num_envs):
            # Tensor -> Numpy 변환 (CPU로 이동)
            img_np = rgba_images[i, :, :, :3].cpu().numpy().astype(np.uint8)
            
            # OpenCV는 BGR, Isaac은 RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # YOLO Inference
            results = self.yolo_model(img_bgr, verbose=False, classes=[32]) # 타겟 클래스지정

            
            best_box = None
            max_area = 0
            
            # 시각화를 위한 프레임 복사
            debug_frame = img_bgr.copy()

            for result in results:
                for box in result.boxes:
                    # BBox 좌표
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # 시각화 (바운딩 박스 그리기)
                    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 가장 큰 객체를 타겟으로 선정
                    if area > max_area:
                        max_area = area
                        best_box = [cx, cy, area]

            # Observation 생성 (Normalization: -1 ~ 1)
            img_h, img_w, _ = img_np.shape
            if best_box is not None:
                norm_cx = (best_box[0] / img_w) * 2 - 1
                norm_cy = (best_box[1] / img_h) * 2 - 1
                norm_area = best_box[2] / (img_w * img_h)
                self.yolo_obs[i] = torch.tensor([norm_cx, norm_cy, norm_area, 1.0], device=self.device)
            else:
                # 타겟 없음
                self.yolo_obs[i] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)

            # Viewport 시각화
            if self.num_envs == 1: # 단일 에이전트일 때만 팝업
                cv2.imshow(f"Jetbot Camera (Env {i})", debug_frame)
                cv2.waitKey(1)

        # 3. Policy에 전달할 관측값 리턴
        return {"policy": self.yolo_obs}

    def _get_rewards(self) -> torch.Tensor:
        # Ground Truth 거리를 보상 계산에 사용 (학습 효율을 위해)
        robot_pos = self.robot.data.root_pos_w
        target_pos = self.target.data.root_pos_w
        
        distance = torch.norm(target_pos[:, :2] - robot_pos[:, :2], dim=1)
        
        # reward 계산부분
        reward_reach = 1.0 / (1.0 + distance) #타겟접근보상
        

        align_error = torch.abs(self.yolo_obs[:, 0]) # 중앙 정렬 오차
        reward_align = torch.where(self.yolo_obs[:, 3] > 0.5, 1.0 - align_error, torch.zeros_like(align_error))
        
        # 충돌 페널티
        penalty_collision = torch.where(distance < self.cfg.dist_threshold, -1.0, 0.0)
        
        total_reward = (
            self.cfg.rew_scale_reach * reward_reach + 
            self.cfg.rew_scale_align * reward_align + 
            self.cfg.rew_scale_collision * penalty_collision
        )
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos = self.robot.data.root_pos_w
        target_pos = self.target.data.root_pos_w
        distance = torch.norm(target_pos[:, :2] - robot_pos[:, :2], dim=1)
        
        # 종료조건
        has_reached = distance < self.cfg.dist_threshold #타겟 도달
        time_out = self.episode_length_buf >= self.max_episode_length - 1 #시간초과
        too_far = distance > 4.0 # 너무 멀리 벗어남
        
        return has_reached | too_far, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)

        # 로봇 랜덤 배치
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        robot_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 랜덤 회전 (Yaw)
        random_yaw = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        quat = math_utils.quat_from_euler_xyz(torch.zeros_like(random_yaw), torch.zeros_like(random_yaw), random_yaw)
        robot_root_state[:, 3:7] = quat
        self.robot.write_root_state_to_sim(robot_root_state, env_ids)
        
        # 타겟 랜덤 배치 (로봇 주변 1~3m)
        target_root_state = self.target.data.default_root_state[env_ids].clone()
        random_dist = torch.rand(len(env_ids), device=self.device) * 2.0 + 1.0
        random_angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        
        target_pos_x = robot_root_state[:, 0] + random_dist * torch.cos(random_angle)
        target_pos_y = robot_root_state[:, 1] + random_dist * torch.sin(random_angle)
        
        target_root_state[:, 0] = target_pos_x
        target_root_state[:, 1] = target_pos_y
        target_root_state[:, 2] = 0.1 # 높이 고정
        
        self.target.write_root_state_to_sim(target_root_state, env_ids)