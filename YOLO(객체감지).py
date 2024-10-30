# 라이브러리 가져오기 
from ultralytics import YOLO
import cv2
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_path = "content/test/images/val_image (251).png"

# YOLO 모델 로드
yolo_model = YOLO("yolov10x.pt")

# YOLO 모델로 객체 탐지 (신뢰도 조정)
results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=False, show=True)

cv2.waitKey(0)
cv2.destroyAllWindows()
