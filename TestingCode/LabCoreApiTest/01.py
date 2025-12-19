"""

앱 런처, 파서 설정

"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Lab Test"
    )
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""

주요기능 import

"""
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


"""

씬 구성&설정

"""
def design_scene():
    # 평지 생성
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 광원 생성
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0, # 조도
        color=(0.75, 0.75, 0.75), # 색상
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # 오브젝트 그룹 생성
    prim_utils.create_prim("/World/Objects", "Xform")
    
    # 빨간 원뿔 설정
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    # 설정기반으로, 각각 생성
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # rigid body 적용된 초록 원뿔 설정
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    # 설정기반으로 생성
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # deformable body 적용된 파란색 직육면체 설정
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    # 설정기반으로 생성
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    # usd 파일 불러오기
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))



"""

메인함수

"""
def main():

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    design_scene() # 설정된 씬 불러오기
    # Play the simulator
    sim.reset()
    print("Simulation started.")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()