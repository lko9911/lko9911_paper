# 필요한 라이브러리 임포트
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 모델과 토크나이저 불러오기
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.location_classifier = nn.Linear(self.bert.config.hidden_size, 4)
        self.object_classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        location_logits = self.location_classifier(cls_output)
        object_logits = self.object_classifier(cls_output)
        return location_logits, object_logits

# 모델 및 토크나이저 로드
model = MultiTaskModel()
model.load_state_dict(torch.load('./my_bert_model.pth'))
model.eval()  # 평가 모드로 전환
tokenizer = BertTokenizer.from_pretrained('./my_bert_tokenizer')

# 라벨 매핑
location_labels = {0: '왼쪽', 1: '오른쪽', 2: '위', 3: '없음'}
object_labels = {0: '사람', 1: '물체'}

# 예측 함수 정의
def predict(text):
    # 텍스트 토크나이징
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # 모델 예측
    with torch.no_grad():
        location_logits, object_logits = model(inputs['input_ids'], inputs['attention_mask'])
        
        # 각 태스크별 예측
        predicted_location = torch.argmax(location_logits, dim=1).item()
        predicted_object = torch.argmax(object_logits, dim=1).item()
        
    return predicted_location, predicted_object

# 예측할 텍스트 입력
test_texts = [
    "좌측의 사람 사진 분석 요청",
    "우측의 물건 분석",
    "위에 있는 사람 사진 분석",
    "왼쪽의 물건 분석"
]

# 예측 수행
for text in test_texts:
    location_pred, object_pred = predict(text)
    print(f"텍스트: {text}")
    print(f"예측된 위치: {location_labels[location_pred]}, 예측된 객체: {object_labels[object_pred]}\n")
