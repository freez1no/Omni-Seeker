import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from ultralytics import YOLO


class AIBrainNode(Node):
    def __init__(self):
        super().__init__("ai_brain_node")

        self.get_logger().info("AI 뇌(Brain) 파이프라인 초기화 중...")

        # 1. 모델 로드 (YOLO 및 JIT 변환된 RL 모델)
        self.yolo_model = YOLO("/home/freezino-inc/dev/robotrl/rl03/yolo26n.pt")
        self.rl_model = torch.jit.load("isaac_rl_policy.pt")
        self.rl_model.eval()

        self.bridge = CvBridge()

        # 2. ROS 2 통신 설정
        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )
        self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)

        # 3. 로봇 상태 메모리 (RL Observation 용)
        self.last_vx = 0.0
        self.last_wz = 0.0

        # 4. 물리 스펙
        self.WHEEL_RADIUS = 0.03
        self.TRACK_WIDTH = 0.11

        self.get_logger().info("두뇌 가동 완료! 시각 데이터를 기다립니다.")

    def image_callback(self, msg):
        # ROS 이미지 -> OpenCV 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = cv_image.shape

        # [단계 1: YOLO 객체 인식]
        results = self.yolo_model(cv_image, verbose=False, classes=[32])
        detected, cx, cy = 0.0, 0.0, 0.0

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xywh = box.xywh[0].cpu().numpy()

            # 중심 좌표 정규화 (-1.0 ~ 1.0)
            cx = (xywh[0] / w) * 2.0 - 1.0
            cy = (xywh[1] / h) * 2.0 - 1.0
            detected = 1.0

            # 디버깅용 화면 출력
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Jetbot AI Vision", cv_image)
        cv2.waitKey(1)

        # [단계 2: RL 상태 구성 및 행동 추론]
        # 9차원 상태: [detected, cx, cy, vx, vy, vz, wx, wy, wz]
        obs_array = [detected, cx, cy, self.last_vx, 0.0, 0.0, 0.0, 0.0, self.last_wz]
        obs_tensor = torch.tensor([obs_array], dtype=torch.float32)

        cmd = Twist()
        with torch.no_grad():
            raw_action = self.rl_model(obs_tensor).squeeze().numpy()

            # 시뮬레이션 환경 동일 조건 (안전 클리핑)
            clamped_action = np.clip(raw_action, -1.0, 1.0)

            # Action Space(scale=10.0) -> 조인트 각속도(rad/s)
            left_joint_vel = clamped_action[0] * 10.0
            right_joint_vel = clamped_action[1] * 10.0

            # 정운동학(Forward Kinematics) -> 몸체 속도(m/s, rad/s)
            raw_vx = (right_joint_vel + left_joint_vel) * self.WHEEL_RADIUS / 2.0
            raw_wz = (
                (right_joint_vel - left_joint_vel)
                * self.WHEEL_RADIUS
                / self.TRACK_WIDTH
            )

            # [단계 3: Sim-to-Real 증폭 및 하달]
            # 현실의 마찰력을 극복하기 위한 물리적 명령 증폭
            SIM_TO_REAL_SCALE = 0.7

            cmd.linear.x = float(raw_vx * SIM_TO_REAL_SCALE)
            cmd.angular.z = float(raw_wz * SIM_TO_REAL_SCALE)

            # RL 모델의 혼동을 막기 위해, 피드백은 '증폭 전 원본'으로 기록
            self.last_vx = float(raw_vx)
            self.last_wz = float(raw_wz)

        self.publisher_.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = AIBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
