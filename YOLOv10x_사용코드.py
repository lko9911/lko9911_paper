# 라이브러리 가져오기 
from ultralytics import YOLO
import cv2
import torch

# GPU 사용 확인 필수_1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_path = "content/test/images/val_image (251).png"

# YOLO 모델 로드 (재학습 안하고 그대로 사용 (v10 버전))
yolo_model = YOLO("yolov10x.pt")

# YOLO 모델로 객체 탐지 (신뢰도 조정)
results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=False, show=True)

# 윈도우 필수_2
cv2.waitKey(0)
cv2.destroyAllWindows()
