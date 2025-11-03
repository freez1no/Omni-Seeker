import gymnasium as gym

# 👇 1. 'agents' 모듈을 임포트하는 코드를 추가합니다.
from . import agents 

# (사용자님의 파일 이름에 맞게 수정)
from .yolo_env import yoloEnv 
from .yolo_env_cfg import yoloEnvCfg

# Gymnasium 레지스트리에 우리 환경을 정식으로 등록합니다.
gym.register(
    id="jbtestv0", 
    
    # (사용자님의 클래스 이름에 맞게 수정)
    entry_point=f"{__name__}.yolo_env:yoloEnv",
    
    kwargs={
        # (사용자님의 클래스 이름에 맞게 수정)
        "env_cfg_entry_point": f"{__name__}.yolo_env_cfg:yoloEnvCfg",
        
        # 👇 3. (가장 중요) 이 라인을 추가합니다!
        # 'agents' 폴더 안의 기본 skrl 설정 파일을 사용하라고 알려줍니다.
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml", 
        # [cite: 2369]
    },
    disable_env_checker=True,
)