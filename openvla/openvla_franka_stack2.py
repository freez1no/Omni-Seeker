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
    # 4️⃣ World 설정 (수정)
    # ===============================
    # 항상 새로운 World 생성
    print("[INFO] Creating new world...")
    world = World(stage_units_in_meters=1.0)
    stage = world.stage

    # Physics Scene 생성 (이미 있으면 스킵)
    physics_scene_path = "/physicsScene"
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
        physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        print("[INFO] Physics scene created.")
    else:
        print("[INFO] Physics scene already exists.")

    # 바닥
    if not is_prim_path_valid("/World/ground_plane"):
        world.scene.add(GroundPlane(prim_path="/World/ground_plane"))
        print("[INFO] Ground plane created.")
    else:
        print("[INFO] Ground plane already exists.")

    # ===============================
    # 5️⃣ Franka 로봇 로드
    # ===============================
    if not is_prim_path_valid("/World/Franka"):
        franka_usd_path = "/home/ubuntu/isaacsim_assets/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FactoryFranka/factory_franka.usd"
        print(f"[INFO] Loading Franka from {franka_usd_path}")
        add_reference_to_stage(usd_path=franka_usd_path, prim_path="/World/Franka")
        
        # articulation 확인
        franka_prim = stage.GetPrimAtPath("/World/Franka")
        if not franka_prim.IsValid():
            raise RuntimeError("Franka articulation prim not found.")
        
        if not franka_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print("[INFO] Adding ArticulationRootAPI to Franka...")
            UsdPhysics.ArticulationRootAPI.Apply(franka_prim)
    else:
        print("[INFO] Franka already loaded.")

    # ===============================
    # 6️⃣ 카메라 설정
    # ===============================
    if not world.scene.object_exists("camera"):
        camera = world.scene.add(
            Camera(
                prim_path="/World/Camera",
                position=np.array([-3.51154, -3.38932, 2.22184]),
                frequency=10,
                resolution=(224, 224),  # 해상도 추가
                name="camera"
            )
        )
        
        # 카메라 방향 설정
        camera_prim = stage.GetPrimAtPath("/World/Camera")
        xform = UsdGeom.Xformable(camera_prim)
        
        orient_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
                break
        
        if orient_op:
            orient_op.Set(Gf.Quatd(0.74619, 0.53125, -0.23269, -0.32684))
        else:
            orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
            orient_op.Set(Gf.Quatd(0.74619, 0.53125, -0.23269, -0.32684))
        
        print("[INFO] Camera created and oriented.")
    else:
        camera = world.scene.get_object("camera")
        print("[INFO] Using existing camera.")

    # ===============================
    # 7️⃣ 큐브 설정
    # ===============================
    if not world.scene.object_exists("red_cube"):
        red_cube = world.scene.add(
            VisualCuboid(
                prim_path="/World/RedCube",
                name="red_cube",
                position=np.array([0.45, 0.0, 0.02]),
                size=0.04,
                color=np.array([1, 0, 0])
            )
        )
        print("[INFO] Red cube created.")
    else:
        print("[INFO] Red cube already exists.")

    if not world.scene.object_exists("blue_cube"):
        blue_cube = world.scene.add(
            VisualCuboid(
                prim_path="/World/BlueCube",
                name="blue_cube",
                position=np.array([0.55, 0.0, 0.02]),
                size=0.04,
                color=np.array([0, 0, 1])
            )
        )
        print("[INFO] Blue cube created.")
    else:
        print("[INFO] Blue cube already exists.")

    # ===============================
    # 8️⃣ Franka Articulation 설정
    # ===============================
    if not world.scene.object_exists("franka"):
        franka = world.scene.add(
            Articulation(prim_path="/World/Franka", name="franka")
        )
        print("[INFO] Franka articulation added to scene.")
    else:
        franka = world.scene.get_object("franka")
        print("[INFO] Using existing Franka articulation.")

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

    # Home position으로 이동
    print("[INFO] Moving to home position...")
    home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
    franka.set_joint_positions(home_joints)

    for _ in range(100):
        world.step(render=True)
        time.sleep(0.01)

    print("[INFO] Home position reached.")

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
    # 1️⃣1️⃣ OpenVLA 명령어 처리
    # ===============================
    command = "Pick up the red cube and place it on the blue cube."
    print(f"[INFO] Processing command: {command}")

    # OpenVLA 프롬프트
    prompt = f"In: What action should the robot take to {command.lower()}?\nOut:"

    inputs = processor(text=prompt, images=img, return_tensors="pt")
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            if key == 'input_ids' or key == 'attention_mask':
                inputs[key] = inputs[key].to(device, dtype=torch.long)
            else:
                inputs[key] = inputs[key].to(device, dtype=torch.float16)

    # predict_action 사용
    with torch.no_grad():
        action_vector = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    print(f"[OpenVLA Action Vector] {action_vector}")
    print(f"[INFO] Target position: [{action_vector[0]:.3f}, {action_vector[1]:.3f}, {action_vector[2]:.3f}]")
    print(f"[INFO] Target orientation: [{action_vector[3]:.3f}, {action_vector[4]:.3f}, {action_vector[5]:.3f}]")
    print(f"[INFO] Gripper command: {action_vector[6]:.3f} (0=open, 1=close)")

    # ===============================
    # 1️⃣2️⃣ 로봇 제어
    # ===============================
    print("[INFO] Executing action...")

    target_position = np.array(action_vector[:3])
    gripper_cmd = action_vector[6]

    # 현재 joint positions 가져오기
    current_joints = franka.get_joint_positions()
    target_joints = current_joints.copy()

    # End-effector 위치를 joint space로 매핑
    target_joints[0] = np.clip(target_position[1] * 2.0, -2.8973, 2.8973)
    target_joints[1] = np.clip(-0.785 - (target_position[2] - 0.5) * 2.0, -1.7628, 1.7628)
    target_joints[3] = np.clip(-2.356 + (target_position[0] - 0.307) * 3.0, -3.0718, -0.0698)

    # Gripper
    gripper_pos = 0.04 * (1.0 - gripper_cmd)
    target_joints[7] = gripper_pos
    target_joints[8] = gripper_pos

    print(f"[INFO] Target joints: {target_joints}")

    # 부드러운 이동
    steps = 200
    for i in range(steps):
        alpha = (i + 1) / steps
        interpolated = current_joints + alpha * (target_joints - current_joints)
        franka.set_joint_positions(interpolated)
        world.step(render=True)
        time.sleep(0.01)
        
        if (i + 1) % 50 == 0:
            print(f"[INFO] Step {i+1}/{steps}")

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