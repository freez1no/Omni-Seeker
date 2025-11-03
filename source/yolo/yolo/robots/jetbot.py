import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Isaac Sim에 내장된 Jetbot 에셋 USD 파일
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"
    ),
    actuators={
        # Jetbot의 모든 관절에 기본 액추에이터 설정
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"], damping=None, stiffness=None
        ),
    },
)