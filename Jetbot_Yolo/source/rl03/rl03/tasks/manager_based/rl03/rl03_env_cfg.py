# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass


from . import mdp

##
# Scene definition
##


@configclass
class Rl03SceneCfg(InteractiveSceneCfg):
    """Configuration for the Jetbot + Sphere scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0), color=(0.9, 0.9, 0.9)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.707, 0.0, 0.707, 0.0)),
    )
    # walls
    wall_n = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallN",
        spawn=sim_utils.CuboidCfg(
            size=(15.2, 0.2, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 7.5, 1.0)),
    )
    wall_s = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallS",
        spawn=sim_utils.CuboidCfg(
            size=(15.2, 0.2, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -7.5, 1.0)),
    )
    wall_e = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallE",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 15.2, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(7.5, 0.0, 1.0)),
    )
    wall_w = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallW",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 15.2, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-7.5, 0.0, 1.0)),
    )

    # roof to cast varying shadows
    roof = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Roof",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )

    # obstacles to create varied environments
    obstacle_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        spawn=sim_utils.CylinderCfg(
            radius=0.4,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.5)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, -2.5, 0.5)),
    )

    obstacle_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.2, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.5, 2.5, 0.5)),
    )

    obstacle_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle3",
        spawn=sim_utils.ConeCfg(
            radius=0.4,
            height=0.8,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.5, 0.4)),
    )

    # robot: Jetbot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/data/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0},
        ),
        actuators={
            "diff_drive": ImplicitActuatorCfg(
                joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
                effort_limit_sim=400.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )

    # target: Red Sphere
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.1)),
    )

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/camera_mount",
        update_period=0,
        height=240,
        width=240,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1), rot=(0.5, -0.5, 0.5, -0.5)
        ),  # Adjust orientation if needed (ROS convention vs Optical)
        # Using quaternion (w, x, y, z) conventions
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    body_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=10.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # YOLO findings
        detected = ObsTerm(func=mdp.object_detected_obs)
        bbox_center = ObsTerm(func=mdp.bbox_center_obs)

        # Robot state
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_sphere = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sphere"),
        },
    )

    reset_obstacle_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("obstacle_1"),
        },
    )

    reset_obstacle_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("obstacle_2"),
        },
    )

    reset_obstacle_3 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("obstacle_3"),
        },
    )

    reset_roof = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "z": (-0.5, 0.5),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("roof"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task Rewards
    object_detected = RewTerm(func=mdp.object_detected_reward, weight=2.0)

    explore = RewTerm(
        func=mdp.explore_reward,
        weight=2.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    approach = RewTerm(
        func=mdp.approach_target_reward,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("sphere"),
        },
    )

    target_reached = RewTerm(
        func=mdp.target_reached_reward,
        weight=50.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("sphere"),
        },
    )

    # Penalties
    center_focus = RewTerm(func=mdp.center_penalty, weight=-2.0)

    smooth_driving_penalty = RewTerm(
        func=mdp.smooth_driving_penalty,
        weight=-0.5,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    collision = RewTerm(
        func=mdp.collision_penalty_strict,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("sphere"),
        },
    )

    # Standard
    alive = RewTerm(func=mdp.is_alive, weight=0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    detection_timeout = DoneTerm(
        func=mdp.detection_timeout,
        params={"time_threshold": 40.0},
    )

    target_reached = DoneTerm(
        func=mdp.target_reached_termination,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("sphere"),
        },
    )


##
# Environment configuration
##


@configclass
class Rl03EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Rl03SceneCfg = Rl03SceneCfg(
        num_envs=64, env_spacing=15.0
    )  # Reduced num_envs for YOLO
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4  # 120/4 = 30Hz Control
        self.episode_length_s = 80.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.solver_velocity_iteration_count = 1
        # Enable debug visualization for potential bbox drawing (implemented in rewards if needed or manual)
