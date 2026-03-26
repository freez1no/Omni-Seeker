<div align="center" dir="auto"><img width="1051" height="527" alt="image" src="https://github.com/user-attachments/assets/3aa88360-2756-4d5d-876e-5177cec76a96" />
<p dir="auto"><a href="https://github.com/freez1no/Omni-Seeker/blob/master/README.md" rel="nofollow">ENG</a><br></p></div>

# Nvidia Omniverse Isaac Sim & Lab - Jetbot & YOLO 모델을 이용한 객체 인식 강화학습 주행
이 프로젝트는 **NVIDIA Isaac Lab** 시뮬레이션 환경에서 **Jetbot** 로봇이 자율적으로 객체를 탐색하고 접근하도록 훈련하는 것을 목표로 한다.

## 소개
본 프로젝트는 로봇의 '인식'과 '행동'을 결합하는 강화학습 파이프라인을 구축한다.
시뮬레이션 환경(Isaac Lab)에서 `Jetbot.usd` 에셋을 불러와 로봇에 탑재된 **카메라 센서**를 통해 실시간으로 주변 환경 데이터를 수집하고, 이 비전 데이터는 최신 객체 인식 모델인 **YOLO v11**로 전송되어 '사람' 또는 사전에 정의된 '사물'의 위치를 바운딩박스로 식별한다.
강화학습 에이전트는 이 YOLO 모델의 객체 인식 결과를 Reward 및 State 정보로 활용하여, 탐지된 객체에게 안전하고 효율적으로 접근하는(적당히 가까이 다가가는) 최적의 주행 policy를 학습한다.

### 목표
1. Core API 기반 Isaac Lab Code로 Jetbot, Target prim 생성 ✓
2. Jetbot 카메라 인식 확인, RL 환경에서, 여러대의 Jetbot 및 Camera 인식 확인 ✓
3. Isaac Lab 내 Jetbot의 카메라 센서로부터 이미지 데이터를 실시간으로 수집 ✓
4. 수집된 비전 데이터를 YOLO v11 모델이 인식하도록 변환 ✓
5. Isaac Lab RL 테스트 ✓
6. 원하는 객체(사람, 사물 등)의 바운딩 박스 정보를 획득, 바운딩 박스 실시간 모니터링 ✓
7. 객체 탐지 정보를 기반으로, 로봇이 목표물에 성공적으로 접근했을 때 보상을 제공하는 강화학습 환경 설계 ✓
8. 최종 Isaac Lab RL, 모델 생성
9. 실제 Jetbot에 모델 적용 및 테스트

###  기술 스택
- 환경 : NVIDIA Isaac Lab
- 객체인식 : YOLO 26n
- 알고리즘 : RL Games, skrl, PPO
- 데이터 파이프라인 : Isaac Sim Camera Sensor

## 강화학습 환경 상세 (Environment Details)
Jetbot의 객체 인식 기반 강화학습 환경(`rl03`)에 대한 상세 정의는 다음과 같습니다.

### 1. Task (목표)
*   **Goal**: Jetbot이 환경 내에 있는 빨간색 구체(Target Sphere)를 인식하고, 그 물체에 가까이 다가가도록 학습합니다.
*   로봇은 물체를 카메라의 중심에 두면서 접근하는 최적의 경로를 스스로 학습하게 됩니다.

### 2. Action (행동)
*   **Type**: `JointVelocityAction` (바퀴 속도 제어)
*   **Details**: 로봇의 왼쪽 바퀴(`left_wheel_joint`)와 오른쪽 바퀴(`right_wheel_joint`)의 속도를 제어합니다.
*   **Scale**: 정책(Policy)이 출력한 행동 값에 **10.0**을 곱하여 실제 속도 명령으로 변환합니다.

### 3. Observation (관측 정보)
로봇은 결정을 내리기 위해 총 **9차원**의 관측 정보를 사용합니다.
1.  **`detected` (1-dim)**: 현재 카메라 화면에 목표 물체가 인식되었는지 여부 (인식됨=1.0, 인식안됨=0.0).
2.  **`bbox_center` (2-dims)**: 인식된 물체의 Bounding Box 중심 좌표 (x, y). 이미지 중심을 기준으로 정규화된 값입니다.
3.  **`base_lin_vel` (3-dims)**: 로봇 본체(Base)의 선형 속도.
4.  **`base_ang_vel` (3-dims)**: 로봇 본체(Base)의 각속도.

### 4. Reward (보상 함수)
로봇이 올바른 행동을 학습하도록 유도하는 보상(+)과 벌칙(-)입니다.

| 이름                      | 타입  | 가중치 | 설명                                                                            |
| :------------------------ | :---: | :----: | :------------------------------------------------------------------------------ |
| **`object_detected`**     | 보상  |  +2.0  | 물체가 인식되면 매 스텝 보상을 받습니다. (인식 실패 시 -1.0)                    |
| **`approach_object`**     | 보상  |  +2.0  | 물체가 인식된 상태에서, 물체와의 거리가 가까울수록 보상 증가 (`exp(-distance)`) |
| **`approach_velocity`**   | 보상  |  +5.0  | 로봇이 물체 방향으로 움직일 때, 그 속도 성분에 비례하여 보상                    |
| **`approach_centered`**   | 보상  |  +5.0  | **[핵심]** 물체에 가까우면서 동시에 물체가 화면 중앙에 위치할수록 큰 보상       |
| **`alive`**               | 보상  |  +0.1  | 에피소드 생존 보상                                                              |
| **`bbox_center_penalty`** | 벌칙  |  -2.0  | 인식된 물체가 화면 중심에서 벗어날수록 벌점                                     |
| **`collision`**           | 벌칙  | -10.0  | 벽이나 다른 물체와 충돌 시 큰 벌점                                              |

### 5. Termination (종료 조건)
*   **`time_out`**: 최대 에피소드 길이(40초) 도달 시 종료.
*   **`detection_timeout`**: 6초 이상 물체를 인식하지 못하면 에피소드 조기 종료 (학습 효율화).

## 설치 및 실행 방법
### 필요 라이브러리 설치
> 기본적으로, Isaac Sim 및 Isaac Lab이 설치되어있음을 가정합니다.
```python
pip install ultralytics opencv-python
```

```
cd Ommi-Seeker
python -m pip install -e source/yolo/
pip install ultralytics opencv-python skrl
```
