import argparse
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Jetbot Granular Sand Simulation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


args_cli.device = "cuda:0"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
import math
import numpy as np
import torch
from pxr import UsdGeom, Gf, Sdf, PhysxSchema, UsdPhysics
from omni.physx.scripts import particleUtils, physicsUtils
import omni.usd


@configclass
class JetbotSandSceneCfg(InteractiveSceneCfg):
    
    jetbot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Jetbot",
        spawn=sim_utils.UsdFileCfg(
            # Isaac Sim Nucleus 서버의 기본 Jetbot 자산 경로 사용
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0, # 물리 안정성을 위한 관통 방지 속도 제한
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8, # 정확한 관절 제어를 위해 반복 횟수 증가
                solver_velocity_iteration_count=0,
                fix_root_link=False, # 로봇이 움직여야 하므로 고정 해제
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15), # 모래 위에 약간 떠서 스폰 (침하 고려)
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # 차동 구동을 위한 휠 액추에이터 설정
            "diff_drive": ImplicitActuatorCfg(
                joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
                effort_limit=500.0,    # 모래 저항을 이기기 위해 토크 제한을 넉넉히 설정
                velocity_limit=100.0,
                stiffness=0.0,         # 속도 제어를 위해 강성 0
                damping=1000.0,        # 목표 속도 추종을 위한 댐핑 설정
            ),
        },
    )



def create_sand_environment(stage, scene_path):

    pbd_material_path = Sdf.Path("/World/Physics_Materials/SandMaterial")
    particleUtils.add_pbd_particle_material(
        stage, 
        pbd_material_path,
        friction=0.7,        # 입자 간 및 바닥과의 마찰 계수 (건조한 모래)
        damping=0.0,         # 에너지 손실률 (흩날리는 효과를 위해 낮게 설정)
        adhesion=0.0,        # 표면 부착력 없음
        viscosity=0.0,       # 점성 없음 (유체가 아님)
        cohesion=0.02,       # 약간의 응집력 부여 (완전 산란 방지 및 모래 더미 형성용)
        gravity_scale=1.0,
        drag=0.0,
        lift=0.0
    )


    particle_system_path = Sdf.Path("/World/Physics_System/ParticleSystem")
    particleUtils.add_physx_particle_system(
        stage=stage,    
        particle_system_path=particle_system_path,
        simulation_owner=Sdf.Path(scene_path),
        contact_offset=0.02,         # 충돌 감지 거리
        rest_offset=0.015,           # 입자 간 평형 거리
        particle_contact_offset=0.02,
        solid_rest_offset=0.015,
        fluid_rest_offset=0.0,
        solver_position_iterations=16 # TGS 솔버 안정성을 위해 반복 횟수 증가
    )


    dim_x, dim_y, dim_z = 1.5, 1.5, 0.08  # 가로 1.5m, 세로 1.5m, 높이 8cm의 모래밭
    particle_spacing = 0.035              # 입자 간격 (약 3.5cm)
    
    # 그리드 포인트 계산
    x_range = np.arange(-dim_x/2, dim_x/2, particle_spacing)
    y_range = np.arange(-dim_y/2, dim_y/2, particle_spacing)
    z_range = np.arange(0.02, dim_z, particle_spacing)
    
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    positions = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
    
    # 자연스러운 배치를 위해 약간의 랜덤 노이즈 추가
    jitter = np.random.uniform(-0.005, 0.005, positions.shape)
    positions += jitter
    
    # USD 포맷(VtArray) 호환을 위해 리스트로 변환 (Gf.Vec3f 형태)
    positions_gf = [Gf.Vec3f(float(p), float(p), float(p)) for p in positions]
    num_particles = len(positions_gf)
    
    print(f"[INFO] Spawning {num_particles} sand particles...")

    particle_set_path = Sdf.Path("/World/Sand/ParticleSet")
    
    # 입자 세트 생성 명령
    particleUtils.add_physx_particleset_points(
        stage=stage,
        path=particle_set_path,
        positions_list=positions_gf,
        velocities_list=[Gf.Vec3f(0,0,0)] * num_particles,
        widths_list=[0.02] * num_particles, # 시각적 입자 크기
        particle_system_path=particle_system_path,
        self_collision=True, # 입자끼리 충돌 허용
        fluid=False,         # False = Granular(모래), True = Fluid(물)
        particle_group=0,
        particle_mass=0.01,  # 입자 개당 질량
        density=0.0          # 질량을 직접 지정하므로 밀도는 0
    )
    
    # 생성된 입자 세트에 물성 재질 적용
    prim = stage.GetPrimAtPath(particle_set_path)
    physicsUtils.add_physics_material_to_prim(stage, prim, pbd_material_path)

# ---------------------------------------------------------
# 5. 메인 실행 루프
# ---------------------------------------------------------

def main():
    # A. 시뮬레이션 컨텍스트 설정
    # 입자 시뮬레이션의 안정성을 위해 TGS (Temporal Gauss-Seidel) 솔버 사용 권장
    sim_cfg = SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 60.0, # 60Hz 업데이트
        physx=PhysxCfg(
            solver_type=1, # 1 = TGS Solver (입자 적층 안정성 우수)
            gpu_max_particle_contacts=2*1024*1024, # 입자 충돌 버퍼 크기 대폭 증가
            enable_ccd=True # 고속 충돌 감지를 위한 연속 충돌 감지 활성화
        )
    )
    sim = SimulationContext(sim_cfg)

    # B. 기본 씬 구성 (바닥 및 조명)
    # 입자가 빠지지 않도록 바닥 평면 생성
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)
    
    # 조명 추가
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.9, 0.85, 0.8))
    cfg_light.func("/World/Sunlight", cfg_light)
    
    # C. 모래 환경 절차적 생성
    # USD 스테이지에 직접 접근하여 입자 시스템을 구축합니다.
    # 안전한 수정을 위해 시뮬레이션 타임라인을 일시 정지 상태로 둡니다.
    sim.pause()
    stage = omni.usd.get_context().get_stage()
    
    # 현재 PhysicsScene 경로 획득
    scene_path = sim.get_physics_context().prim_path
    create_sand_environment(stage, scene_path)

    # D. Isaac Lab 씬 초기화 (Jetbot 로드)
    # 위에서 정의한 설정 클래스를 사용하여 로봇을 스폰합니다.
    scene_cfg = JetbotSandSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # E. 시뮬레이션 시작 및 리셋
    sim.reset()
    print("[INFO] Simulation initialized. Starting control loop...")
    print("[INFO] Jetbot will move forward. Observe the sand scattering.")

    # 제어 명령: 전진 (양쪽 바퀴에 양의 속도 인가)
    cmd_vel = torch.tensor([[15.0, 15.0]], device=sim.device)

    # F. 런타임 루프
    while simulation_app.is_running():
        # 1. 로봇 제어 입력 (Action Application)
        # Jetbot 관절체에 속도 명령 전달
        scene.articulations["jetbot"].set_joint_velocity_target(cmd_vel)
        
        # 2. 시뮬레이션 데이터 쓰기 (USD/PhysX 버퍼 업데이트)
        scene.write_data_to_sim()
        
        # 3. 물리 스텝 진행 및 렌더링
        sim.step()
        
        # 4. 시뮬레이션 데이터 읽기 (상태 업데이트)
        scene.update(dt=sim.get_physics_dt())

    # 종료 처리
    simulation_app.close()

if __name__ == "__main__":
    main()