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
        super().__init__('ai_brain_node')

        self.get_logger().info("AI 뇌(Brain) 초기화 중...")
        
        self.yolo_model = YOLO('/home/freezino-inc/dev/robotrl/rl03/yolo26n.pt') 
        self.rl_model = torch.jit.load('isaac_rl_policy.pt')
        self.rl_model.eval()

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.last_vx = 0.0
        self.last_wz = 0.0

        self.WHEEL_RADIUS = 0.03
        self.TRACK_WIDTH = 0.11
        
        self.get_logger().info("Model Ready.")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = cv_image.shape

        results = self.yolo_model(cv_image, verbose=False, classes=[32])
        
        detected = 0.0
        cx, cy = 0.0, 0.0

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xywh = box.xywh[0].cpu().numpy()
            
            # 중심점 정규화 (-1.0 ~ 1.0)
            cx = (xywh[0] / w) * 2.0 - 1.0
            cy = (xywh[1] / h) * 2.0 - 1.0
            detected = 1.0

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Jetbot Detection view", cv_image)
        cv2.waitKey(1)
        obs_array = [detected, cx, cy, self.last_vx, 0.0, 0.0, 0.0, 0.0, self.last_wz]
        obs_tensor = torch.tensor([obs_array], dtype=torch.float32)

        cmd = Twist()
        with torch.no_grad():
            raw_action = self.rl_model(obs_tensor).squeeze().numpy()
            
            # 🚨 [가장 핵심적인 보호 장치] Isaac Lab처럼 행동을 -1.0 ~ 1.0으로 강제 제한
            clamped_action = np.clip(raw_action, -1.0, 1.0)
            
            # 학습 코드의 ActionsCfg 기준 (scale=10.0) -> Joint Velocity (rad/s)
            left_joint_vel = clamped_action[0] * 10.0
            right_joint_vel = clamped_action[1] * 10.0

            # Forward Kinematics (Twist로 변환)
            vx = (right_joint_vel + left_joint_vel) * self.WHEEL_RADIUS / 2.0
            wz = (right_joint_vel - left_joint_vel) * self.WHEEL_RADIUS / self.TRACK_WIDTH

            # 젯봇 모터 노드로 전송할 최종 명령 (motor_node.py에서 마찰력을 스스로 이겨냄)
            cmd.linear.x = float(vx)
            cmd.angular.z = float(wz)

            print(f"Move  V: {vx:.3f} m/s, W: {wz:.3f} rad/s | Detect: {detected==1.0}")

            # 피드백 업데이트 (절대 폭발하지 않는 정상적인 수치)
            self.last_vx = float(vx)
            self.last_wz = float(wz)

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

if __name__ == '__main__':
    main()
