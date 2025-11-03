import torch

torch.cuda.init()  # CUDA 시스템 초기화
print(torch.cuda.is_available())  # CUDA 디바이스 사용 가능 여부 확인
torch.cuda.reset_peak_memory_stats(device=None)