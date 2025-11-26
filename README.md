<div align="center" dir="auto"><img width="1051" height="527" alt="image" src="https://github.com/user-attachments/assets/3aa88360-2756-4d5d-876e-5177cec76a96" />
<p dir="auto"><a href="https://github.com/freez1no/Omni-Seeker/blob/master/README.md" rel="nofollow">ENG</a><br></p></div>

# Nvidia Omniverse Isaac Sim & Lab RL Project
**NVIDIA Isaac Lab** 시뮬레이션 환경에서 **Jetbot** 로봇이 다양한 방법으로 학습, 행동하는 것을 목표로 각 프로젝트를 진행합니다.

## 프로젝트 소개
| ProjectName | Description | Version |
| --- | --- | --- |
| Jetbot_YOLO | Jetbot과 YOLO v8(or 11)모델을 결합하여, 객체인식 및 추적 강화학습 | 0.0.3 |
| Jetbot_FrozenLake | Isaac Sim 환경에 FrozenLake 환경을 생성하고, 특정 구역을 지나가지 않는 강화학습 | 0.0.0 |

## Environment & Installation
이 프로젝트는 **NVIDIA Isaac Sim** 및 **Isaac Lab** 환경에서 구동됨. 원활한 시뮬레이션을 위해 다음의 하드웨어 및 소프트웨어 환경을 권장함

### Prerequisites
* **OS**: Ubuntu 22.04 +
* **GPU**: RTX Series (VRAM 16GB+) / 연구실에서는 L40S 를 사용하였음.
* **Software**:
    * [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) (Version 5.0.0+)
    * [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

### Full Project Installation
```bash
# Clone this repository
git clone [https://github.com/freez1no/Omni-Seeker.git](https://github.com/freez1no/Omni-Seeker.git)
cd Omni-Seeker