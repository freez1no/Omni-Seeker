# =========================================================
# OpenVLA + Isaac Sim 5.0.0 + Franka Panda stacking example
# =========================================================

import os
import sys
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np
import time
import re

# ===============================
# 1️⃣ SimulationApp 초기화 (함수 밖에서)
# ===============================
try:
    if "simulation_app" not in globals():
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({"headless": False})
    else:
        print("[INFO] simulation_app already exists, skipping new initialization.")
except Exception as e:
    print(f"[WARN] omni.isaac initialization failed: {e}")

# ===============================
# 2️⃣ Isaac Sim Core Imports (함수 밖에서)
# ===============================
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane, VisualCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import is_prim_path_valid
from pxr import UsdPhysics, UsdGeom, Gf

# ===============================
# 3️⃣ OpenVLA 모델 로드 (함수 밖에서)
# ===============================
GLOBAL_VLA_MODEL = "my_openvla_model_instance"
GLOBAL_VLA_PROCESSOR = "my_openvla_processor_instance"
device = "cuda" if torch.cuda.is_available() else "cpu"

if GLOBAL_VLA_MODEL not in globals() or globals()[GLOBAL_VLA_MODEL] is None:
    print("[INFO] OpenVLA 모델이 VRAM에 없습니다. 새로 로드합니다...")
    model_id = "openvla/openvla-7b"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
        trust_remote_code=True        
    ).to(device)
    model.eval()

    globals()[GLOBAL_VLA_MODEL] = model
    globals()[GLOBAL_VLA_PROCESSOR] = processor
    print("[INFO] OpenVLA 모델 로드 완료 및 전역 변수에 저장됨.")
else:
    print("[INFO] VRAM에 이미 로드된 OpenVLA 모델을 재사용합니다.")
    model = globals()[GLOBAL_VLA_MODEL]
    processor = globals()[GLOBAL_VLA_PROCESSOR]


# ===============================
# 🔥 메인 함수 정의
# ===============================
def run_openvla_demo():
    """OpenVLA 데모 실행 함수 - Stop/Play 버튼을 눌러도 재실행 가능"""
    
    print("\n" + "="*60)
    print("Starting OpenVLA Demo")
    print("="*60 + "\n")
    
    # ===============================
    # 기존 World 완전히 정리
    # ===============================
    from omni.isaac.core import World
    
    # 기존 World 인스턴스가 있으면 완전히 제거
    try:
        existing_world = World.instance()
        if existing_world is not None:
            print("[INFO] Clearing existing world...")
            existing_world.stop()  # 시뮬레이션 중지
            existing_world.clear_all_callbacks()
            existing_world.clear()
            World._instance = None  # 강제로 인스턴스 제거
            print("[INFO] Existing world cleared.")
    except Exception as e:
        print(f"[INFO] No existing world or error clearing: {e}")
    
    # Stage에서 직접 객체들 제거
    from omni import usd
    context = usd.get_context()
    stage = context.get_stage()
    
    if stage:
        paths_to_remove = [
            "/World/Franka", 
            "/World/Camera", 
            "/World/RedCube", 
            "/World/BlueCube", 
            "/World/ground_plane"
        ]
        for path in paths_to_remove:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                stage.RemovePrim(path)
                print(f"[INFO] Removed {path}")
    
    # 약간의 대기 시간 (정리 완료를 위해)
    import time
    time.sleep(0.5)
    
    # ===============================
    # 4️⃣ World 설정
    # ===============================
    print("[INFO] Creating new world...")
    world = World(stage_units_in_meters=1.0)
    stage = world.stage

    # Physics Scene 생성 - 조건 제거
    physics_scene_path = "/physicsScene"
    physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
    print("[INFO] Physics scene created.")

    # 바닥 - 조건 제거
    add_reference_to_stage(
        usd_path="/home/ubuntu/isaacsim_assets/Assets/Isaac/5.0/Isaac/Environments/Grid/default_environment.usd",
        prim_path="/World/ground_plane"
    )
    print("[INFO] Wood floor created.")

    # ===============================
    # 5️⃣ Franka 로봇 로드 - 조건 제거
    # ===============================
    franka_usd_path = "/home/ubuntu/isaacsim_assets/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FactoryFranka/factory_franka.usd"
    print(f"[INFO] Loading Franka from {franka_usd_path}")
    add_reference_to_stage(usd_path=franka_usd_path, prim_path="/World/Franka")
    
    franka_prim = stage.GetPrimAtPath("/World/Franka")
    if not franka_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print("[INFO] Adding ArticulationRootAPI to Franka...")
        UsdPhysics.ArticulationRootAPI.Apply(franka_prim)

    # ===============================
    # 6️⃣ 카메라 설정
    # ===============================
    camera = world.scene.add(
        Camera(
            prim_path="/World/Camera",
            position=np.array([3.67634, -0.24096, 1.4259]),
            frequency=10,
            resolution=(224, 224),
            name="camera"
        )
    )

    # 카메라 방향 설정 (기존 orient_op 확인 후 설정)
    camera_prim = stage.GetPrimAtPath("/World/Camera")
    xform = UsdGeom.Xformable(camera_prim)

    # 기존 orient_op 찾기
    orient_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op
            break

    # orient_op가 없을 때만 추가
    if orient_op is None:
        orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

    # 값 설정
    orient_op.Set(Gf.Quatd(0.59411, 0.41798, 0.39546, 0.56209))
    print("[INFO] Camera created and oriented.")

    # ===============================
    # 7️⃣ 큐브 설정 - 조건 제거
    # ===============================
    red_cube = world.scene.add(
        VisualCuboid(
            prim_path="/World/RedCube",
            name="red_cube",
            position=np.array([0.5, -0.11, 0.04]),
            size=0.08,
            color=np.array([1, 0, 0])
        )
    )
    print("[INFO] Red cube created.")

    blue_cube = world.scene.add(
        VisualCuboid(
            prim_path="/World/BlueCube",
            name="blue_cube",
            position=np.array([0.7, 0.1, 0.04]),
            size=0.08,
            color=np.array([0, 0, 1])
        )
    )
    print("[INFO] Blue cube created.")

    # ===============================
    # 8️⃣ Franka Articulation 설정 - 조건 제거
    # ===============================
    franka = world.scene.add(
        Articulation(prim_path="/World/Franka", name="franka")
    )
    print("[INFO] Franka articulation added to scene.")

    # ===============================
    # 9️⃣ 초기화
    # ===============================
    print("[INFO] Initializing world...")
    world.reset()
    camera.initialize()

    # Stabilize
    for _ in range(50):
        world.step(render=True)
        time.sleep(0.02)

    # # Home position으로 이동
    # print("[INFO] Moving to home position...")
    # home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
    # franka.set_joint_positions(home_joints)

    # for _ in range(100):
    #     world.step(render=True)
    #     time.sleep(0.01)

    # print("[INFO] Home position reached.")

    # ===============================
    # 🔟 카메라 이미지 캡처
    # ===============================
    print("[INFO] Capturing camera image...")
    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError("[ERROR] Failed to capture camera image")

    rgb = rgba[:, :, :3].copy()  # 이미 uint8이므로 그냥 복사만
    img = Image.fromarray(rgb)
    img.save("/tmp/camera_capture.png")
    print("[INFO] Camera image saved to /tmp/camera_capture.png")
    

    # ===============================
    # 1️⃣1️⃣ OpenVLA 명령어 처리 및 반복 실행
    # ===============================
    # command = "Pick up the red cube and place it on the blue cube."
    command = "move gripper to red cube."
    print(f"[INFO] Processing command: {command}")

    # IK Solver는 루프 밖에서 한 번만 초기화
    print("[INFO] Initializing IK Solver...")
    from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    from omni.isaac.motion_generation import interface_config_loader
    from scipy.spatial.transform import Rotation as R

    kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula_kinematics_solver = LulaKinematicsSolver(**kinematics_config)
    art_kinematics_solver = ArticulationKinematicsSolver(franka, lula_kinematics_solver, "panda_hand")
    print("[INFO] IK Solver initialized.")

    # 10 스텝 반복 실행
    for step in range(10):
        print(f"\n{'='*60}")
        print(f"Step {step+1}/10")
        print(f"{'='*60}")
        
        # 이미지 캡처
        print("[INFO] Capturing camera image...")
        rgba = camera.get_rgba()
        if rgba is None:
            print("[ERROR] Failed to capture camera image")
            break
        
        rgb = rgba[:, :, :3].copy()
        img = Image.fromarray(rgb)
        img.save(f"/tmp/camera_capture{step}.png")
        
        # OpenVLA 프롬프트
        prompt = f"In: What action should the robot take to {command.lower()}?\nOut:"
        
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                if key == 'input_ids' or key == 'attention_mask':
                    inputs[key] = inputs[key].to(device, dtype=torch.long)
                else:
                    inputs[key] = inputs[key].to(device, dtype=torch.float16)
        
        # predict_action 사용 (nyu_franka_play로 변경)
        with torch.no_grad():
            action_vector = model.predict_action(**inputs, unnorm_key="nyu_franka_play_dataset_converted_externally_to_rlds", do_sample=False)

        # ===== 스케일링 조정 =====
        # Franka 데이터셋은 실제 로봇 스케일이므로 스케일 팩터를 더 작게
        POSITION_SCALE = 1.0  # 1/10 → 1/2로 증가 (더 큰 움직임)
        ROTATION_SCALE = 1.0  # 1/5 → 1/2로 증가

        action_vector_scaled = action_vector.copy()
        action_vector_scaled[0:3] *= POSITION_SCALE  # xyz
        action_vector_scaled[3:6] *= ROTATION_SCALE  # roll/pitch/yaw
        # gripper (index 6)는 그대로 유지
        
        print(f"[OpenVLA Action Vector (Original)] {action_vector}")
        print(f"[OpenVLA Action Vector (Scaled)] {action_vector_scaled}")
        
                
        print(f"[OpenVLA Action Vector] {action_vector}")
        print(f"[INFO] Target position delta: [{action_vector[0]:.3f}, {action_vector[1]:.3f}, {action_vector[2]:.3f}]")
        print(f"[INFO] Target orientation delta: [{action_vector[3]:.3f}, {action_vector[4]:.3f}, {action_vector[5]:.3f}]")
        print(f"[INFO] Gripper command: {action_vector[6]:.3f}")
        
        # ===============================
        # 1️⃣3️⃣ Action Vector를 End-Effector Pose로 변환
        # ===============================
        # 현재 EE pose 가져오기
        current_ee_position, current_ee_rotation = art_kinematics_solver.compute_end_effector_pose()
        
        # Roll, pitch, yaw를 quaternion으로 변환
        current_rotation_scipy = R.from_matrix(current_ee_rotation)
        current_euler = current_rotation_scipy.as_euler('xyz')
        
        # Delta euler angle 적용
        
        # 이후 action_vector 대신 action_vector_scaled 사용
        target_position = current_ee_position + action_vector_scaled[:3]
        target_euler = current_euler + action_vector_scaled[3:6]
        
        # target_euler = current_euler + action_vector[3:6]
        target_rotation_scipy = R.from_euler('xyz', target_euler)
        target_orientation = target_rotation_scipy.as_quat()  # [x, y, z, w]
        
        print(f"[INFO] Current EE position: {current_ee_position}")
        print(f"[INFO] Target EE position: {target_position}")
        print(f"[INFO] Target EE orientation (quat): {target_orientation}")
        
        # ===============================
        # 1️⃣4️⃣ IK 계산
        # ===============================
        action, success = art_kinematics_solver.compute_inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation
        )
        
        if not success:
            print("[WARN] IK solution not found! Skipping this step.")
            continue
        else:
            print("[INFO] IK solution found successfully.")
            
            # Gripper 값 추가
            gripper_cmd = action_vector[6]
            gripper_pos = 0.04 * (1.0 - gripper_cmd)
            
            # ArticulationAction에서 joint positions 가져오기
            target_joints = action.joint_positions
            
            # Gripper joint 추가 (마지막 2개)
            if len(target_joints) == 7:  # 팔만 7개 joint
                target_joints = np.append(target_joints, [gripper_pos, gripper_pos])
            
            print(f"[INFO] Target joints: {target_joints}")
            
            # ===============================
            # 1️⃣5️⃣ 로봇 제어
            # ===============================
            print("[INFO] Executing action...")
            
            # 현재 joint positions
            current_joints = franka.get_joint_positions()
            
            # 부드러운 이동
            steps = 100  # 스텝당 이동 시간 단축
            for i in range(steps):
                alpha = (i + 1) / steps
                interpolated = current_joints + alpha * (target_joints - current_joints)
                franka.set_joint_positions(interpolated)
                world.step(render=True)
                time.sleep(0.01)
            
            print("[INFO] Action execution complete.")
            print(f"[INFO] Gripper: {'CLOSED' if gripper_cmd > 0.5 else 'OPEN'}")

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60 + "\n")
        
        
# ===============================
# ✅ 실행
# ===============================
# 이 부분만 실행하면 됩니다!
run_openvla_demo()