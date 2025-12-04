from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class JetbotYoloEnvCfg(DirectRLEnvCfg):
    decimation = 2  # policy 제어 주기
    episode_length_s = 20.0 # 1에피소드당 길이(초)
    
    # Action Space [왼쪽 바퀴 속도, 오른쪽 바퀴 속도] -> 2개
    action_space = 2
    
    # [YOLO BBox Center X, YOLO BBox Center Y, YOLO BBox Area, Target Detected Flag] -> 4개
    # 이미지 픽셀 전체가 아닌, YOLO 결과값만 RL에 넘깁니다.
    observation_space = 4
    state_space = 0

    # 시뮬레이션 설정
    sim: SimulationCfg = SimulationCfg(
        dt=1/60, 
        render_interval=decimation,
        physx=PhysxCfg(gpu_found_lost_pairs_capacity=4096) # 충돌 버퍼 확보
    )

    # Scene 설정
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # 학습 환경 수
        env_spacing=5.0, # 간격(m)
        replicate_physics=True # 물리 엔진 복제
    )

    # Jetbot 설정
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05), # 약간 띄워서 스폰
        ),
        # Jetbot은 Differential Drive 방식
        actuators={
            "velocity_ctrl": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"],
                velocity_limit=15.0,
                effort_limit=10.0,
                stiffness=0.0,
                damping=100.0,
            ),
        },
    )

    # 카메라 센서 Jetbot 전면
    tiled_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/rgb_camera/camera", # Jetbot USD 구조에 맞춤
        update_period=0.0, # 매 스텝 업데이트
        height=240, width=320, # 카메라 해상도(추론속도에 영향있음)
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.08), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # 타겟설정
    target_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # 빨간색
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.1)),
    )

    # reward function
    rew_scale_reach = 1.5     # 타겟 접근 보상
    rew_scale_align = 2.0     # YOLO 박스가 중앙에 올수록 보상
    rew_scale_collision = -10.0 # 충돌 페널티
    dist_threshold = 0.3      # 목표 도달 거리