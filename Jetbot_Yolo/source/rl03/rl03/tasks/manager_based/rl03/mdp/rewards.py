# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import cv2
import numpy as np

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# Global Model Loading
YOLO_MODEL = None
MODEL_PATH = "/home/freezino-inc/dev/robotrl/rl03/yolo26n.pt"


def _load_yolo():
    global YOLO_MODEL
    if YOLO_MODEL is None and YOLO is not None:
        try:
            YOLO_MODEL = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(
    env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def _get_yolo_detections(env: ManagerBasedRLEnv):
    """Run YOLO inference and cache results for the current step."""
    _load_yolo()

    # Initialize cache if needed
    if not hasattr(env, "_yolo_cache"):
        env._yolo_cache = {"step": -1, "results": None}

    # Return cached results if available for this step
    if env._yolo_cache["step"] == env.common_step_counter:
        return env._yolo_cache["results"]

    # Get camera data
    # Assuming sensor name is "camera" and it returns [N, H, W, 4]
    try:
        camera_sensor = env.scene.sensors["camera"]
        images = camera_sensor.data.output["rgb"].cpu().numpy()  # [N, H, W, 3] RGB
    except KeyError:
        # Fallback/Error if camera not found
        return {
            "detected": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "centers": torch.zeros((env.num_envs, 2), device=env.device),
        }

    detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    centers = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)

    if YOLO_MODEL:
        # Batch inference
        # Note: ultralytics might be slow on CPU loop, but standard predict handles list of arrays
        # Filter for 'sports ball' (class index 32 in COCO)
        results = YOLO_MODEL(list(images), verbose=False, classes=[32])

        for i, r in enumerate(results):
            if len(r.boxes) > 0:
                # Naive: take the first detection
                # Ideally check for class if model has multiple
                box = r.boxes[0]
                xywh = box.xywh[0].cpu().numpy()  # cx, cy, w, h

                # Normalize center to [-1, 1]
                h, w = r.orig_shape
                cx = (xywh[0] / w) * 2 - 1
                cy = (xywh[1] / h) * 2 - 1

                detected[i] = True
                centers[i, 0] = cx
                centers[i, 1] = cy

                # Visualization (Optional): Draw bbox in debug vis if enabled
                # This requires 3D projection which is complex here.
                # Just storing data.

        # Visualization part
        if env.num_envs > 0:
            # Visualize only the first environment environment 0
            img_to_show = images[0].copy()  # RGB
            # Convert to BGR for OpenCV
            if img_to_show.dtype != np.uint8:
                # If float [0,1], scale to [0,255]
                if img_to_show.max() <= 1.0:
                    img_to_show = (img_to_show * 255).astype(np.uint8)
                else:
                    img_to_show = img_to_show.astype(np.uint8)

            img_bgr = cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)

            # Draw bbox if detected in env 0
            r0 = results[0]
            if len(r0.boxes) > 0:
                box0 = r0.boxes[0]
                x1, y1, x2, y2 = box0.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            try:
                cv2.imshow("Jetbot Camera View", img_bgr)
                cv2.waitKey(1)
            except Exception:
                # Likely running headless or no display available
                pass

    cache = {"detected": detected, "centers": centers}
    env._yolo_cache = {"step": env.common_step_counter, "results": cache}
    return cache


def object_detected_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for detecting the object (1.0), Penalty for not detecting (-1.0)."""
    data = _get_yolo_detections(env)
    return torch.where(data["detected"], 1.0, -1.0)


def approach_object_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for approaching the target (only if detected)."""
    data = _get_yolo_detections(env)
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]

    # Calculate distance
    pos_robot = robot.data.root_pos_w[:, :2]
    pos_target = target.data.root_pos_w[:, :2]
    dist = torch.norm(pos_robot - pos_target, dim=1)

    # Reward only if detected
    # Use exponential gradient for sharper reward when close
    dist = torch.norm(pos_robot - pos_target, dim=1)
    # sigma = 1.0 (controlling the width of the reward)
    reward = torch.exp(-dist)
    return torch.where(data["detected"], reward, torch.tensor(0.0, device=env.device))


def approach_velocity_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for moving towards the target (dot product of velocity and direction)."""
    data = _get_yolo_detections(env)
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]

    # Directions
    pos_robot = robot.data.root_pos_w[:, :2]
    pos_target = target.data.root_pos_w[:, :2]
    target_vec = pos_target - pos_robot
    target_dir = target_vec / (torch.norm(target_vec, dim=1, keepdim=True) + 1e-6)

    # Robot velocity
    vel_robot = robot.data.root_lin_vel_w[:, :2]

    # Project velocity onto target direction
    approach_vel = torch.sum(vel_robot * target_dir, dim=1)

    # Only reward if moving towards (positive dot product) and detected
    reward = torch.clamp(approach_vel, min=0.0)
    return torch.where(data["detected"], reward, torch.tensor(0.0, device=env.device))


def approach_centered_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Largest reward for approaching the target while keeping it centered."""
    data = _get_yolo_detections(env)
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]

    # Distance
    pos_robot = robot.data.root_pos_w[:, :2]
    pos_target = target.data.root_pos_w[:, :2]
    dist = torch.norm(pos_robot - pos_target, dim=1)

    # Alignment (Center error)
    center_error = torch.norm(data["centers"], dim=1)
    alignment_factor = torch.exp(-2.0 * center_error)

    reward = (1.0 / (1.0 + dist)) * alignment_factor
    return torch.where(data["detected"], reward, torch.tensor(0.0, device=env.device))


def bbox_center_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for bounding box deviating from center."""
    data = _get_yolo_detections(env)
    dist = torch.norm(data["centers"], dim=1)
    # Only penalize if detected? Or if not detected, full penalty?
    # Requirement: "box deviates... penalty" => implies existing box.
    return torch.where(data["detected"], dist, torch.tensor(0.0, device=env.device))


def collision_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalty for collision."""
    # Check net contact forces
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Check max force over history and bodies
    # net_forces_w_history: [N, History, Body, 3]
    forces = torch.norm(contact_sensor.data.net_forces_w_history, dim=-1)  # [N, H, B]
    max_force = torch.amax(forces, dim=(1, 2))  # Max over history and bodies
    return (max_force > 1.0).float()


def object_detected_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: Object detected (1.0 or 0.0)."""
    data = _get_yolo_detections(env)
    return data["detected"].float().unsqueeze(-1)


def bbox_center_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: Bounding box center (x, y)."""
    data = _get_yolo_detections(env)
    return data["centers"]


def detection_timeout(
    env: ManagerBasedRLEnv, time_threshold: float = 3.0
) -> torch.Tensor:
    """Terminate if the object is not detected after a certain time."""
    # current time > threshold
    time_out = env.episode_length_buf * env.step_dt > time_threshold
    # not detected
    data = _get_yolo_detections(env)
    return time_out & (~data["detected"])
