from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openvla/openvla-7b"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    trust_remote_code=True
).to(device)

def predict_action(image: Image.Image, instruction: str):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to(device)
    # predict_action은 OpenVLA 모델에서 제공되는 메서드
    action = model.predict_action(**inputs, do_sample=False)
    return action.detach().cpu().numpy()

# 테스트 샘플
# 이미지와 지시사항 준비
img = Image.open("pic_yellow_cup_original2.jpg")
instr = "pick up the yellow cup and place it to the left"

# processor를 이용해 inputs 변수 정의
inputs = processor(instr, img).to(device) 

# inputs 변수를 사용해 액션 예측
action = model.predict_action(
    **inputs,
    do_sample=False,
    unnorm_key="bridge_orig"
)

# 결과 출력
print("Predicted action vector:", action)
