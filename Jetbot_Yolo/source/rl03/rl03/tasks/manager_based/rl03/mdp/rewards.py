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

    # Make sure images have valid dimensions before passing to YOLO
    if YOLO_MODEL and images.size > 0:
        # Handle case where output is (N, num_cameras, H, W, C)
        if images.ndim == 5 and images.shape[1] == 1:
            images = np.squeeze(images, axis=1)

        if images.ndim >= 3 and images.shape[1] > 0 and images.shape[2] > 0:
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


def explore_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """When target is NOT detected, reward active exploration."""
    data = _get_yolo_detections(env)
    robot = env.scene[robot_cfg.name]

    current_time = env.episode_length_buf * env.step_dt
    is_scanning_phase = current_time < 4.0

    lin_vel = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    ang_vel_z = robot.data.root_ang_vel_w[:, 2]

    # --- Phase 1: Scan Phase (First 4 seconds) ---
    # Goal: Spin in place to find the target.
    # Reward angular velocity (up to 2.0 rad/s), penalize linear velocity.
    spin_reward = torch.clamp(torch.abs(ang_vel_z), max=2.0)
    spin_penalty = lin_vel * 2.0
    phase_1_reward = spin_reward - spin_penalty

    # --- Phase 2: Search Phase (After 4 seconds) ---
    # Goal: Move around avoiding collisions.
    # Reward moving forward, penalize standing completely still to prevent getting stuck.
    speed_bonus = torch.clamp(lin_vel, max=1.0) * 1.5
    stand_still_penalty = torch.where(
        lin_vel < 0.1,
        torch.tensor(1.0, device=env.device),
        torch.tensor(0.0, device=env.device),
    )

    phase_2_reward = speed_bonus - stand_still_penalty

    reward = torch.where(is_scanning_phase, phase_1_reward, phase_2_reward)

    return torch.where(~data["detected"], reward, torch.tensor(0.0, device=env.device))


def object_detected_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small baseline reward for successfully tracking/keeping the object in view."""
    data = _get_yolo_detections(env)
    return data["detected"].float()


def approach_target_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """When detected, strongly reward moving TOWARDS the target."""
    data = _get_yolo_detections(env)
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]

    pos_robot = robot.data.root_pos_w[:, :2]
    pos_target = target.data.root_pos_w[:, :2]
    target_dir = pos_target - pos_robot
    target_dir = target_dir / (torch.norm(target_dir, dim=1, keepdim=True) + 1e-6)

    vel_robot = robot.data.root_lin_vel_w[:, :2]
    approach_speed = torch.sum(vel_robot * target_dir, dim=1)

    # Only reward positive approach speed
    reward = torch.clamp(approach_speed, min=0.0)

    return torch.where(data["detected"], reward, torch.tensor(0.0, device=env.device))


def center_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the object not being in the center. (Only when detected)"""
    data = _get_yolo_detections(env)
    # centers are in [-1, 1], penalize distance from center
    center_error = torch.norm(data["centers"], dim=1)
    # Exponentially stronger penalty for larger deviations
    penalty = torch.square(center_error)
    return torch.where(data["detected"], penalty, torch.tensor(0.0, device=env.device))


def target_reached_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Huge reward for reaching the target."""
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]

    dist = torch.norm(
        robot.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=1
    )
    return (dist < 0.4).float()


def collision_penalty_strict(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize collisions heavily, but ignore if we are at the target (since we want to catch it)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w_history, dim=-1)
    max_force = torch.amax(forces, dim=(1, 2))

    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    dist = torch.norm(
        robot.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=1
    )

    # If distance is > 0.4 and max_force > 1.0, it's a collision with an obstacle/wall.
    collision = (max_force > 1.0) & (dist >= 0.4)
    return collision.float()


def target_reached_termination(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate the episode successfully when reaching the target."""
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    dist = torch.norm(
        robot.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=1
    )
    return dist < 0.35


def smooth_driving_penalty(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize excessive angular velocity (left/right swaying) to encourage smooth driving.
    Disabled during the initial scanning phase where turning is required."""
    robot = env.scene[robot_cfg.name]
    ang_vel = robot.data.root_ang_vel_w[:, 2]

    current_time = env.episode_length_buf * env.step_dt
    is_scanning_phase = current_time < 4.0

    data = _get_yolo_detections(env)

    # Apply penalty only if target detected OR not in scanning phase
    penalty_active = data["detected"] | (~is_scanning_phase)

    penalty = torch.square(ang_vel)

    return torch.where(penalty_active, penalty, torch.tensor(0.0, device=env.device))


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
