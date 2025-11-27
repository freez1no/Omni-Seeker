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

    
print("\n" + "="*60)
print("Starting OpenVLA Demo")
print("="*60 + "\n")


# ===============================
# 함수 정의
# ===============================


import numpy as np
from scipy.spatial.transform import Rotation as R

def Add_translate(current, delta):
    """
    현재 좌표 + 더해줄 좌표를 더해서 (x, y, z) 반환.
    current, delta: 튜플, list, numpy array, 등 다양한 형태 허용
    """
    # numpy array로 바꾸기
    curr_arr = np.array(current, dtype=float)
    delta_arr = np.array(delta, dtype=float)

    # 길이 맞추기: 둘 다 길이 3이어야 함
    if curr_arr.shape not in [(3,), (3,1), (1,3)] or delta_arr.shape not in [(3,), (3,1), (1,3)]:
        raise ValueError(f"current {current} 또는 delta {delta}의 형식이 잘못됨")

    result = curr_arr + delta_arr
    # 1차원 형태로 풀어서 반환
    return float(result[0]), float(result[1]), float(result[2])


def Add_orient(current_quat, add):
    """
    현재 쿼터니언(rotation) + 더해줄 값(add)을 합친 쿼터니언(w, x, y, z)을 반환.
    - current_quat: (w, x, y, z) 형태 쿼터니언
    - add: 
        * 길이 4라면 쿼터니언으로 판단하여 quaternion 곱 (회전 합성)
        * 길이 3이라면 (roll, pitch, yaw) 변화량(euler delta)으로 판단
    """
    # numpy 배열로 변환
    curr = np.array(current_quat, dtype=float)
    add_arr = np.array(add, dtype=float)

    if curr.shape != (4,):
        raise ValueError(f"current_quat {current_quat}는 (w, x, y, z) 형식이어야 함")

    w0, x0, y0, z0 = curr

    # case 1: add가 쿼터니언 (길이 4)
    if add_arr.shape == (4,):
        wa, xa, ya, za = add_arr
        # 쿼터니언 곱 (Hamilton product): q_new = q_add * q_curr (순서는 목적에 따라 바꿔도 됨)
        # 여기서는 “현재 회전에 추가 회전을 덧붙인다”는 의미로 add * current
        w_new = wa*w0 - xa*x0 - ya*y0 - za*z0
        x_new = wa*x0 + xa*w0 + ya*z0 - za*y0
        y_new = wa*y0 - xa*z0 + ya*w0 + za*x0
        z_new = wa*z0 + xa*y0 - ya*x0 + za*w0

    # case 2: add가 (roll, pitch, yaw) 변화량
    elif add_arr.shape == (3,):
        dr, dp, dy = add_arr  # 라디안으로 가정
        # 현재 쿼터니언을 Rotation 객체로 만들기
        curr_rot = R.from_quat([x0, y0, z0, w0])  # scipy는 (x, y, z, w) 순서 사용
        # delta 회전을 Euler로부터 quaternion으로 만들기
        delta_rot = R.from_euler('xyz', [dr, dp, dy])
        # 회전 합성
        new_rot = delta_rot * curr_rot
        # 다시 쿼터니언으로 변환
        x_new, y_new, z_new, w_new = new_rot.as_quat()
    else:
        raise ValueError(f"add {add}의 형식이 (4,) 또는 (3,)이어야 함")

    # 보통 쿼터니언은 정규화해서 사용하는 것이 안전함
    norm = np.linalg.norm([w_new, x_new, y_new, z_new])
    w_new /= norm
    x_new /= norm
    y_new /= norm
    z_new /= norm

    return w_new, x_new, y_new, z_new


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
        # position=np.array([3.67634, -0.24096, 1.4259]),
        position=np.array([0.13308905, -0.00436904, 0.93973273]),
        frequency=10,
        resolution=(224, 224),
        name="camera"
    )
)
camera2 = world.scene.add(
    Camera(
        prim_path="/World/Camera2",
        position=np.array([3.67634, -0.24096, 1.4259]),
        frequency=10,
        resolution=(224, 224),
        name="camera2"
    )
)

# 카메라 방향 설정 (기존 orient_op 확인 후 설정)
camera_prim = stage.GetPrimAtPath("/World/Camera")
camera2_prim = stage.GetPrimAtPath("/World/Camera2")
xform = UsdGeom.Xformable(camera_prim)
xform2 = UsdGeom.Xformable(camera2_prim)

# 기존 orient_op 찾기
orient_op = None
for op in xform.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        orient_op = op
        break
orient_op2 = None
for op in xform2.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        orient_op2 = op
        break

# orient_op가 없을 때만 추가
if orient_op is None:
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
if orient_op2 is None:
    orient_op2 = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
# # 값 설정
# orient_op.Set(Gf.Quatd(0.59411, 0.41798, 0.39546, 0.56209))
# print("[INFO] Camera created and oriented.")


# o_ori = Gf.Quatd(0.59411, 0.41798, 0.39546, 0.56209)
o_ori = Gf.Quatd(0.0159026, 0.9697918, 0.2388076, 0.0472021)


w = o_ori.GetReal()     # w
x = o_ori.GetImaginary()[0]  # x
y = o_ori.GetImaginary()[1]  # y
z = o_ori.GetImaginary()[2]  # z

print("origial_orient: ", (w, x, y, z))

delta_orient = (0.03394982, -0.11835314, -0.00514122)

target_orient = Add_orient((w, x, y, z), delta_orient)
print("target_orient: ", target_orient)

orient_op.Set(Gf.Quatd(w, x, y, z))
orient_op2.Set(Gf.Quatd(target_orient[0], target_orient[1], target_orient[2], target_orient[3]))
print("[INFO] Camera created and oriented.")

# print("orient_op: ", orient_op.Get())


# quaternion = [w, x, y, z]

# # SciPy는 [x, y, z, w] 순서를 사용하므로 변환
# quat_xyzw = [x, y, z, w]

# # Rotation 객체 생성
# rot = R.from_quat(quat_xyzw)

# # XYZ(roll, pitch, yaw)로 변환
# roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

# print("roll:", roll)
# print("pitch:", pitch)
# print("yaw:", yaw)


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
camera2.initialize()

# Stabilize
for _ in range(50):
    world.step(render=True)
    time.sleep(0.02)



# IK Solver는 루프 밖에서 한 번만 초기화
print("[INFO] Initializing IK Solver...")
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from scipy.spatial.transform import Rotation as R

kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
lula_kinematics_solver = LulaKinematicsSolver(**kinematics_config)
art_kinematics_solver = ArticulationKinematicsSolver(franka, lula_kinematics_solver, "panda_hand")
print("[INFO] IK Solver initialized.")







# 현재 EE pose 가져오기
current_ee_position, current_ee_rotation = art_kinematics_solver.compute_end_effector_pose()
print("current_ee_position : ", current_ee_position)
print("current_ee_rotation : ", current_ee_rotation)





# 회전 행렬 → 쿼터니언 변환
rotation_matrix = current_ee_rotation

quat_xyzw = R.from_matrix(rotation_matrix).as_quat()  # scipy 기본 출력
w = quat_xyzw[3]
x = quat_xyzw[0]
y = quat_xyzw[1]
z = quat_xyzw[2]

quat_xyzw = np.array([x, y, z, w])
quat_wxyz = np.array([w, x, y, z])

camera2_translate = np.array([0.22938, -0.24096, 0.24258])  # ee의 x,y,z 좌표로 이동
camera2_quat_wxyz = np.array([0.5, 0.5, 0.5, 0.5])  # ee의 쿼터니언방향으로 gripper가 향하는것을 GUI로 확인했음!!!

print()
print()
print("quat_wxyz: ", quat_wxyz)




action, success = art_kinematics_solver.compute_inverse_kinematics(
    target_position=camera2_translate,
    target_orientation=camera2_quat_wxyz
)

print("action: ", action)
print("success: ", success)


target_joints = action.joint_positions
print("target_joints: ", target_joints)






# Home position으로 이동
print("[INFO] Moving to home position...")
home_joints = np.concatenate([target_joints, [0.04, 0.04]])
franka.set_joint_positions(home_joints)

for _ in range(100):
    world.step(render=True)
    time.sleep(0.01)

print("[INFO] Home position reached.")




# from scipy.spatial.transform import Rotation as R
# import numpy as np

# # EE rotation matrix from solver
# R_ee = R.from_matrix(current_ee_rotation)

# # EE는 +X forward, Camera는 +Z forward
# # 따라서 camera frame을 EE forward로 맞추기 위해 -90deg about Y 적용
# R_offset = R.from_euler('y', -90, degrees=True)

# R_cam = R_ee * R_offset

# # to quat (w, x, y, z)
# x, y, z, w = R_cam.as_quat()
# quat_wxyz = (w, x, y, z)

# print("Camera orientation aligned to EE forward:", quat_wxyz)





        
# # Roll, pitch, yaw를 quaternion으로 변환
# current_rotation_scipy = R.from_matrix(current_ee_rotation)
# current_euler = current_rotation_scipy.as_euler('xyz')





# action, success = art_kinematics_solver.compute_inverse_kinematics(
#     target_position=target_position,
#     target_orientation=target_orientation
# )

# if not success:
#     print("[WARN] IK solution not found! Skipping this step.")
#     continue
# else:
#     print("[INFO] IK solution found successfully.")
    
#     # Gripper 값 추가
#     gripper_cmd = action_vector[6]
#     gripper_pos = 0.04 * (1.0 - gripper_cmd)
    
#     # ArticulationAction에서 joint positions 가져오기
#     target_joints = action.joint_positions
    
#     # Gripper joint 추가 (마지막 2개)
#     if len(target_joints) == 7:  # 팔만 7개 joint
#         target_joints = np.append(target_joints, [gripper_pos, gripper_pos])
    
#     print(f"[INFO] Target joints: {target_joints}")





# # Home position으로 이동
# print("[INFO] Moving to home position...")
# home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
# franka.set_joint_positions(home_joints)

# for _ in range(100):
#     world.step(render=True)
#     time.sleep(0.01)

# print("[INFO] Home position reached.")


# # 10 스텝 반복 실행
# for step in range(10):
#     print(f"\n{'='*60}")
#     print(f"Step {step+1}/10")
#     print(f"{'='*60}")
    
#     # 이미지 캡처
#     print("[INFO] Capturing camera image...")
#     rgba = camera.get_rgba()
#     if rgba is None:
#         print("[ERROR] Failed to capture camera image")
#         break
    
#     rgb = rgba[:, :, :3].copy()
#     img = Image.fromarray(rgb)
#     img.save(f"/tmp/camera_capture{step}.png")
    
#     # OpenVLA 프롬프트
#     prompt = f"In: What action should the robot take to {command.lower()}?\nOut:"
    
#     inputs = processor(text=prompt, images=img, return_tensors="pt")
#     for key in inputs:
#         if torch.is_tensor(inputs[key]):
#             if key == 'input_ids' or key == 'attention_mask':
#                 inputs[key] = inputs[key].to(device, dtype=torch.long)
#             else:
#                 inputs[key] = inputs[key].to(device, dtype=torch.float16)
    
#     # predict_action 사용 (nyu_franka_play로 변경)
#     with torch.no_grad():
#         action_vector = model.predict_action(**inputs, unnorm_key="nyu_franka_play_dataset_converted_externally_to_rlds", do_sample=False)

#     # ===== 스케일링 조정 =====
#     # Franka 데이터셋은 실제 로봇 스케일이므로 스케일 팩터를 더 작게
#     POSITION_SCALE = 1.0  # 1/10 → 1/2로 증가 (더 큰 움직임)
#     ROTATION_SCALE = 1.0  # 1/5 → 1/2로 증가

#     action_vector_scaled = action_vector.copy()
#     action_vector_scaled[0:3] *= POSITION_SCALE  # xyz
#     action_vector_scaled[3:6] *= ROTATION_SCALE  # roll/pitch/yaw
#     # gripper (index 6)는 그대로 유지
    
#     print(f"[OpenVLA Action Vector (Original)] {action_vector}")
#     print(f"[OpenVLA Action Vector (Scaled)] {action_vector_scaled}")
    
            
#     print(f"[OpenVLA Action Vector] {action_vector}")
#     print(f"[INFO] Target position delta: [{action_vector[0]:.3f}, {action_vector[1]:.3f}, {action_vector[2]:.3f}]")
#     print(f"[INFO] Target orientation delta: [{action_vector[3]:.3f}, {action_vector[4]:.3f}, {action_vector[5]:.3f}]")
#     print(f"[INFO] Gripper command: {action_vector[6]:.3f}")
    
#     # ===============================
#     # 1️⃣3️⃣ Action Vector를 End-Effector Pose로 변환
#     # ===============================
#     # 현재 EE pose 가져오기
#     current_ee_position, current_ee_rotation = art_kinematics_solver.compute_end_effector_pose()
    
#     # Roll, pitch, yaw를 quaternion으로 변환
#     current_rotation_scipy = R.from_matrix(current_ee_rotation)
#     current_euler = current_rotation_scipy.as_euler('xyz')
    
#     # Delta euler angle 적용
    
#     # 이후 action_vector 대신 action_vector_scaled 사용
#     target_position = current_ee_position + action_vector_scaled[:3]
#     target_euler = current_euler + action_vector_scaled[3:6]
    
#     # target_euler = current_euler + action_vector[3:6]
#     target_rotation_scipy = R.from_euler('xyz', target_euler)
#     target_orientation = target_rotation_scipy.as_quat()  # [x, y, z, w]
    
#     print(f"[INFO] Current EE position: {current_ee_position}")
#     print(f"[INFO] Target EE position: {target_position}")
#     print(f"[INFO] Target EE orientation (quat): {target_orientation}")
    
#     # ===============================
#     # 1️⃣4️⃣ IK 계산
#     # ===============================
#     action, success = art_kinematics_solver.compute_inverse_kinematics(
#         target_position=target_position,
#         target_orientation=target_orientation
#     )
    
#     if not success:
#         print("[WARN] IK solution not found! Skipping this step.")
#         continue
#     else:
#         print("[INFO] IK solution found successfully.")
        
#         # Gripper 값 추가
#         gripper_cmd = action_vector[6]
#         gripper_pos = 0.04 * (1.0 - gripper_cmd)
        
#         # ArticulationAction에서 joint positions 가져오기
#         target_joints = action.joint_positions
        
#         # Gripper joint 추가 (마지막 2개)
#         if len(target_joints) == 7:  # 팔만 7개 joint
#             target_joints = np.append(target_joints, [gripper_pos, gripper_pos])
        
#         print(f"[INFO] Target joints: {target_joints}")
        
#         # ===============================
#         # 1️⃣5️⃣ 로봇 제어
#         # ===============================
#         print("[INFO] Executing action...")
        
#         # 현재 joint positions
#         current_joints = franka.get_joint_positions()
        
#         # 부드러운 이동
#         steps = 100  # 스텝당 이동 시간 단축
#         for i in range(steps):
#             alpha = (i + 1) / steps
#             interpolated = current_joints + alpha * (target_joints - current_joints)
#             franka.set_joint_positions(interpolated)
#             world.step(render=True)
#             time.sleep(0.01)
        
#         print("[INFO] Action execution complete.")
#         print(f"[INFO] Gripper: {'CLOSED' if gripper_cmd > 0.5 else 'OPEN'}")

# print("\n" + "="*60)
# print("Demo Complete!")
# print("="*60 + "\n")