import torch
import torch.nn as nn

# 1. 학습된 체크포인트 파일 불러오기
ckpt_path = '/home/freezino-inc/dev/Omni-Seeker/Jetbot_Yolo/logs/skrl/jetbot_yolo/2026-03-31_08-34-10_ppo_torch/checkpoints/best_agent.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

# skrl은 'policy' 키 안에 모델 가중치를 저장합니다. (키가 없으면 전체 로드)
if 'policy' in ckpt:
    state_dict = ckpt['policy']
else:
    state_dict = ckpt

# 2. 동적 신경망 클래스 (출력해주신 키 목록에 맞게 완벽 설계)
class CleanInferencePolicy(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        layers = []
        
        # 히든 레이어 (net_container.0, net_container.2 ...) 자동 추출
        i = 0
        while f'net_container.{i}.weight' in state_dict:
            w = state_dict[f'net_container.{i}.weight']
            b = state_dict[f'net_container.{i}.bias']
            
            linear = nn.Linear(w.shape[1], w.shape[0])
            linear.weight.data = w.clone()
            linear.bias.data = b.clone()
            
            layers.append(linear)
            layers.append(nn.ELU()) # skrl의 Isaac Lab 기본 활성화 함수
            i += 2
            
        self.net = nn.Sequential(*layers)
        
        # 출력 레이어 (policy_layer) 추출 - 로봇의 최종 행동(Action) 결정
        policy_w = state_dict['policy_layer.weight']
        policy_b = state_dict['policy_layer.bias']
        
        self.policy_layer = nn.Linear(policy_w.shape[1], policy_w.shape[0])
        self.policy_layer.weight.data = policy_w.clone()
        self.policy_layer.bias.data = policy_b.clone()

    def forward(self, x):
        x = self.net(x)
        return self.policy_layer(x)

print("--- 1. 가중치 추출 및 아키텍처 역설계 중 ---")
try:
    # 3. 모델 생성 및 가중치 덮어씌우기
    model = CleanInferencePolicy(state_dict)
    model.eval()
    
    print("--- 2. JIT (TorchScript) 컴파일 진행 중 ---")
    # 입력 차원 추론 (우리의 경우 9차원 관측값)
    input_dim = state_dict['net_container.0.weight'].shape[1]
    dummy_input = torch.randn(1, input_dim)
    
    # 4. 모델 궤적 추적(Tracing) 및 C++ 호환 JIT 파일로 저장
    traced_model = torch.jit.trace(model, dummy_input)
    save_path = 'isaac_rl_policy.pt'
    traced_model.save(save_path)
    
    print(f"--- 3. 완료! 독립 실행형 모델 '{save_path}' 생성 성공! ---")
    print(f"-> 모델 입력 차원(Observation): {input_dim}")
    print(f"-> 모델 출력 차원(Action): {state_dict['policy_layer.weight'].shape[0]}")
    
except Exception as e:
    print(f"변환 중 에러 발생: {e}")

