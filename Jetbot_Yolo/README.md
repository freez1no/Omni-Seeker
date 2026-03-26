<div align="center" dir="auto">
<img width="1051" height="527" alt="image" src="https://github.com/user-attachments/assets/3aa88360-2756-4d5d-876e-5177cec76a96" />
<p dir="auto"><a href="https://github.com/freez1no/Omni-Seeker/blob/master/Jetbot_Yolo/README-KOR.md" rel="nofollow">KOR</a><br></p></div>

# Nvidia Omniverse Isaac Sim & Lab - Learning to Enhance Object Awareness Using Jetbot & YOLO Model
The project aims to train **Jetbot** robots to autonomously explore and access objects in a **NVIDIA Isaac Lab** simulation environment.

## Introduction
This project establishes a reinforcement learning pipeline that combines 'recognition' and 'action' of robots.
The 'Jetbot.usd' asset is called from the simulation environment (Isaac Lab) to collect the surrounding environment data in real time through the **camera sensor** mounted on the robot, and this vision data is transmitted to the latest object recognition model **YOLO v11** to identify the location of 'people' or pre-defined 'things' as a bounding box.
The reinforcement learning agent utilizes the object recognition results of this YOLO model as Reward and State information to learn an optimal driving policy that approaches the detected object safely and efficiently (properly close).

### Goals
1. Core API based Isaac Lab Code Jetbot, Target prim generation ✓
2. Check Jetbot camera recognition, RL environment, multiple Jetbot and Camera recognition ✓
3. Real-time collection of image data from Jetbot's camera sensors within the Isaac Lab ✓
4. Transform the collected vision data to be recognized by the YOLO v11 model ✓
5. Isaac Lab RL test ✓
6. Obtain bounding box information for desired objects (people, objects, etc.) and monitor bounding them in real time ✓
7. Based on object detection information, the reinforcement learning environment is designed to reward the robot when it successfully approaches the target ✓
8. Final Isaac Lab RL, Model Creation
9. Apply and test models to real Jetbot

###  Tech Stack
- enviroments : NVIDIA Isaac Lab
- Object recognition: YOLO 26n
- Algorithm: RL Games, skrl, PPO
- Data Pipeline: Isaac Sim Camera Sensor
 
## Environment Details (RL Task)
Details of the Reinforcement Learning environment (`rl03`) for Jetbot object detection are as follows:

### 1. Task
*   **Goal**: The Jetbot learns to detect the red sphere (Target Sphere) in the environment and approach it.
*   The robot learns to approach the object while keeping it in the center of the camera view.

### 2. Action
*   **Type**: `JointVelocityAction` (Wheel Velocity Control)
*   **Details**: Controls the velocity of the left wheel (`left_wheel_joint`) and right wheel (`right_wheel_joint`).
*   **Scale**: The action value output by the policy is multiplied by **10.0** to convert it into actual velocity commands.

### 3. Observation
The robot uses a total of **9 dimensions** of observation data to make decisions.
1.  **`detected` (1-dim)**: Whether the target object is currently detected in the camera view (Detected=1.0, Not detected=0.0).
2.  **`bbox_center` (2-dims)**: The center coordinates (x, y) of the detected object's bounding box. Normalized relative to the image center.
3.  **`base_lin_vel` (3-dims)**: Linear velocity of the robot base.
4.  **`base_ang_vel` (3-dims)**: Angular velocity of the robot base.

### 4. Reward
Rewards (+) and penalties (-) to guide the robot's learning.

| Name                      |  Type   | Weight | Description                                                                           |
| :------------------------ | :-----: | :----: | :------------------------------------------------------------------------------------ |
| **`object_detected`**     | Reward  |  +2.0  | Reward per step if object is detected (Penalty -1.0 if not)                           |
| **`approach_object`**     | Reward  |  +2.0  | Increased reward as distance to object decreases (`exp(-distance)`), only if detected |
| **`approach_velocity`**   | Reward  |  +5.0  | Reward proportional to velocity towards the object                                    |
| **`approach_centered`**   | Reward  |  +5.0  | **[Core]** High reward for being close to object while keeping it centered            |
| **`alive`**               | Reward  |  +0.1  | Survival reward per step                                                              |
| **`bbox_center_penalty`** | Penalty |  -2.0  | Penalty as the bounding box deviates from the center                                  |
| **`collision`**           | Penalty | -10.0  | Large penalty for collision with walls or other objects                               |

### 5. Termination
*   **`time_out`**: Episode terminates after maximum length (40 seconds).
*   **`detection_timeout`**: Episode terminates early if object is not detected for more than 6 seconds.

## Install and Try
### Install Librarys
> By default, assume that Isaac Sim and Isaac Lab are installed.
```python
cd Ommi-Seeker
python -m pip install -e source/yolo/
pip install ultralytics opencv-python skrl
```
