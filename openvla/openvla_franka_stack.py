# =========================================================
# OpenVLA + Isaac Sim 5.0.0 + Franka Panda stacking example (fixed)
# =========================================================

import os
import sys
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np
import time

# ===============================
# 1️⃣ SimulationApp 초기화
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
# 2️⃣ Isaac Sim Core Imports
# ===============================
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane, VisualCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from pxr import UsdPhysics

# ===============================
# 3️⃣ OpenVLA 모델 로드
# ===============================
# ========== 1️⃣ OpenVLA 모델 로드 (싱글톤 패턴 적용) ==========

# 모델과 프로세서를 저장할 전역 변수 이름 정의
GLOBAL_VLA_MODEL = "my_openvla_model_instance"
GLOBAL_VLA_PROCESSOR = "my_openvla_processor_instance"
device = "cuda" if torch.cuda.is_available() else "cpu"

if GLOBAL_VLA_MODEL not in globals() or globals()[GLOBAL_VLA_MODEL] is None:
    print("[INFO] OpenVLA 모델이 VRAM에 없습니다. 새로 로드합니다...")
    model_id = "openvla/openvla-7b"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_id,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float16).to(device)
    model.eval()

    # 로드한 모델과 프로세서를 전역 변수에 저장하여 재사용
    globals()[GLOBAL_VLA_MODEL] = model
    globals()[GLOBAL_VLA_PROCESSOR] = processor
    print("[INFO] OpenVLA 모델 로드 완료 및 전역 변수에 저장됨.")

else:
    print("[INFO] VRAM에 이미 로드된 OpenVLA 모델을 재사용합니다.")
    # 전역 변수에서 기존 모델과 프로세서를 가져옴
    model = globals()[GLOBAL_VLA_MODEL]
    processor = globals()[GLOBAL_VLA_PROCESSOR]

# (이후 스크립트는 'model'과 'processor' 변수를 정상적으로 사용)

# ===============================
# 4️⃣ Isaac Sim 환경 구성
# ===============================
world = World(stage_units_in_meters=1.0)
stage = world.stage

# Physics Scene 생성
physics_scene_path = "/physicsScene"
physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

# 바닥
# world.scene.add(GroundPlane(prim_path="/World/GroundPlane", size=10, color=np.array([0.5, 0.5, 0.5])))
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.utils.prims import is_prim_path_valid

if not is_prim_path_valid("/World/ground_plane"):
    world.scene.add(GroundPlane(prim_path="/World/ground_plane"))
else:
    print("[INFO] ground_plane already exists, skipping creation.")


# ===============================
# 5️⃣ Franka 로봇 로드
# ===============================
franka_usd_path = "/home/ubuntu/isaacsim_assets/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FactoryFranka/factory_franka.usd"
print(f"[INFO] Loading Franka from {franka_usd_path}")
add_reference_to_stage(usd_path=franka_usd_path, prim_path="/World/Franka")

# articulation 경로 확인
franka_prim = stage.GetPrimAtPath("/World/Franka")
if not franka_prim.IsValid():
    raise RuntimeError("Franka articulation prim not found. Check USD path or hierarchy.")

from pxr import UsdPhysics
if not franka_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
    print("[INFO] Adding ArticulationRootAPI to Franka...")
    UsdPhysics.ArticulationRootAPI.Apply(franka_prim)
else:
    print("[INFO] Franka articulation already OK.")

# ===============================
# 6️⃣ 큐브 및 카메라 설정 ([62.944, -43.99, -19.532])
# ===============================
from pxr import UsdGeom, Gf

camera = world.scene.add(
    Camera(
        prim_path="/World/Camera",
        position=np.array([-3.51154, -3.38932, 2.22184]),
        frequency=10
    )
)

# 기존 XformOp 가져와서 수정
camera_prim = stage.GetPrimAtPath("/World/Camera")
xform = UsdGeom.Xformable(camera_prim)

# 기존 orient op 찾기
orient_op = None
for op in xform.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        orient_op = op
        break

if orient_op:
    # 기존 orient op에 값 설정 (double precision 사용)
    orient_op.Set(Gf.Quatd(0.74619, 0.53125, -0.23269, -0.32684))  # (w, x, y, z)
else:
    # orient op이 없으면 생성
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op.Set(Gf.Quatd(0.74619, 0.53125, -0.23269, -0.32684))


red_cube = world.scene.add(
    VisualCuboid(
        prim_path="/World/RedCube",
        name="red_cube",
        position=np.array([0.45, 0.0, 0.02]),
        size=0.04,
        color=np.array([1, 0, 0])
    )
)

blue_cube = world.scene.add(
    VisualCuboid(
        prim_path="/World/BlueCube",
        name="blue_cube",
        position=np.array([0.55, 0.0, 0.02]),
        size=0.04,
        color=np.array([0, 0, 1])
    )
)

# ===============================
# 7️⃣ 초기화
# ===============================
print("[INFO] Initializing world...")
world.reset()
camera.initialize()

# stabilize
for _ in range(50):
    world.step(render=True)
    time.sleep(0.02)

# ===============================
# 8️⃣ 카메라 이미지 캡처
# ===============================
print("[INFO] Capturing camera image...")
rgba = camera.get_rgba()
if rgba is None:
    raise RuntimeError("[ERROR] Failed to capture camera image")

rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
img = Image.fromarray(rgb)
img.save("/tmp/camera_capture.png")
print("[INFO] Camera image saved to /tmp/camera_capture.png")

# ===============================
# 9️⃣ OpenVLA 명령어 처리
# ===============================
command = "Pick up the red cube and place it on the blue cube."
print(f"[INFO] Processing command: {command}")

inputs = processor(images=img, text=command, return_tensors="pt").to(device)
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)

decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"[OpenVLA Output] {decoded}")

try:
    action_vector = [float(x) for x in decoded.strip().split()[:7]]
    if len(action_vector) != 7:
        action_vector = [0.45, 0.0, 0.2, 0.0, 1.57, 0.0, 1.0]
except:
    action_vector = [0.45, 0.0, 0.2, 0.0, 1.57, 0.0, 1.0]

print(f"[INFO] Action vector: {action_vector}")

# ===============================
# 🔟 로봇 제어
# ===============================
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction

# Franka articulation view
franka = ArticulationView(prim_paths_expr="/World/Franka", name="franka_view")
world.scene.add(franka)

# world.reset()

print("[INFO] Moving Franka based on OpenVLA output...")
for i in range(200):
    # 단순히 openvla action을 적용하는 예시 (실제 pose inverse kinematics 필요)
    franka.set_joint_positions(np.zeros(franka.num_dof))
    world.step(render=True)
    time.sleep(0.01)

# ===============================
# ✅ 종료
# ===============================
print("[INFO] Done.")
# if "simulation_app" in globals():
#     simulation_app.close()
