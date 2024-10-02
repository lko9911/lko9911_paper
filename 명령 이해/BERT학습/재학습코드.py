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

##
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':  # MacOS인 경우
    rc('font', family='AppleGothic')
else:
    print("해당 운영체제는 지원되지 않습니다.")

# 유니코드에서 마이너스 기호 깨짐 문제 해결
plt.rcParams['axes.unicode_minus'] = False
##

# 1. 데이터셋 준비 (위치 예측 / 사람/물건 예측)
data = {
    'text': [
        "좌측의 사람 사진 분석 요청",
        "우측의 사람 사진 분석 요청",
        "왼쪽의 사람 사진 분석 요청",
        "사람 사진 분석",
        "왼쪽의 물건 분석",
        "우측의 물건 분석",  
        "위에 있는 사람 사진 분석",
        "좌측의 사람 사진 분석 요청",
        "우측의 사람 사진 분석 요청",
        "왼쪽의 사람 사진 분석 요청",
        "사람 사진 분석",
        "왼쪽의 물건 분석",
        "우측의 물건 분석",  
        "위에 있는 사람 사진 분석",
        "좌측의 사람 사진 분석 요청",
        "우측의 사람 사진 분석 요청",
        "왼쪽의 사람 사진 분석 요청",
        "사람 사진 분석",
        "왼쪽의 물건 분석",
        "우측의 물건 분석",  
        "위에 있는 사람 사진 분석"
    ], 
    # 위치 예측 (왼쪽: 0, 오른쪽: 1, 위: 2, 없음: 3)
    'location_labels': [0, 1, 0, 3, 0, 1, 2,0, 1, 0, 3, 0, 1, 2,0, 1, 0, 3, 0, 1, 2],  
    # 사람/물건 예측 (사람: 0, 물건: 1)
    'object_labels': [0, 0, 0, 0, 1, 1, 0,0, 0, 0, 0, 1, 1, 0,0, 0, 0, 0, 1, 1, 0]  
}

df = pd.DataFrame(data)

# 학습 데이터와 테스트 데이터로 나누기
train_texts, test_texts, train_location_labels, test_location_labels, train_object_labels, test_object_labels = train_test_split(
    df['text'], df['location_labels'], df['object_labels'], test_size=0.2
)

# Series를 리스트로 변환
train_location_labels = train_location_labels.tolist()
test_location_labels = test_location_labels.tolist()
train_object_labels = train_object_labels.tolist()
test_object_labels = test_object_labels.tolist()

# 레이블을 tensor로 변환
train_location_labels = torch.tensor(train_location_labels, dtype=torch.long)
train_object_labels = torch.tensor(train_object_labels, dtype=torch.long)
test_location_labels = torch.tensor(test_location_labels, dtype=torch.long)
test_object_labels = torch.tensor(test_object_labels, dtype=torch.long)

# 2. 토크나이저 준비 및 토크나이징
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts):
    encoding = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    return encoding

# 데이터셋 토크나이징
train_encodings = tokenize_data(train_texts)
test_encodings = tokenize_data(test_texts)

# 3. PyTorch Dataset 클래스 정의
class MultiTaskDataset(Dataset):
    def __init__(self, encodings, location_labels, object_labels):
        self.encodings = encodings
        self.location_labels = location_labels
        self.object_labels = object_labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['location_labels'] = self.location_labels[idx]
        item['object_labels'] = self.object_labels[idx]
        return item

    def __len__(self):
        return len(self.location_labels)

train_dataset = MultiTaskDataset(train_encodings, train_location_labels, train_object_labels)
test_dataset = MultiTaskDataset(test_encodings, test_location_labels, test_object_labels)

# 4. 다중 클래스 모델 정의
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # 태스크 1: 위치 예측 (4개의 클래스)
        self.location_classifier = nn.Linear(self.bert.config.hidden_size, 4)
        # 태스크 2: 사람/물건 분류 (2개의 클래스)
        self.object_classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 출력 사용

        # 각 태스크별 출력
        location_logits = self.location_classifier(cls_output)
        object_logits = self.object_classifier(cls_output)

        return location_logits, object_logits

# 모델 준비
model = MultiTaskModel()
model.train()

# 5. 옵티마이저와 손실 함수 설정
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# 6. 모델 학습
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

for epoch in range(10):  # 3 에포크 동안 학습
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        location_labels = batch['location_labels']
        object_labels = batch['object_labels']

        # 모델 예측
        location_logits, object_logits = model(input_ids, attention_mask)

        # 각 태스크별 손실 계산
        location_loss = loss_fn(location_logits, location_labels)
        object_loss = loss_fn(object_logits, object_labels)

        # 전체 손실 계산 (가중합)
        total_loss = location_loss + object_loss
        total_loss.backward()
        optimizer.step()

        print('=========================')
        print(f"Epoch {epoch}, Total loss: {total_loss.item():.2f}")
        print(f"Location loss: {location_loss.item():.2f} || Object loss: {object_loss.item():.2f}\n")


# 7. 모델 평가
model.eval()
test_loader = DataLoader(test_dataset, batch_size=8)

correct_location = 0
correct_object = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        location_labels = batch['location_labels']
        object_labels = batch['object_labels']
        
        location_logits, object_logits = model(input_ids, attention_mask)

        # 각 태스크별 예측
        predicted_location = torch.argmax(location_logits, dim=1)
        predicted_object = torch.argmax(object_logits, dim=1)

        correct_location += (predicted_location == location_labels).sum().item()
        correct_object += (predicted_object == object_labels).sum().item()
        total += location_labels.size(0)

location_accuracy = correct_location / total
object_accuracy = correct_object / total

print(f'Location Task Accuracy: {location_accuracy:.4f}')
print(f'Object Task Accuracy: {object_accuracy:.4f}')

# 8. 모델 저장
# 모델 아키텍처 저장
torch.save(model.state_dict(), './my_bert_model.pth')
# 토크나이저 저장
tokenizer.save_pretrained('./my_bert_tokenizer')

# 9. 혼동 행렬 및 리포트 출력
def plot_confusion_matrix(cm, labels, title, cmap):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 모델 평가
model.eval()
test_loader = DataLoader(test_dataset, batch_size=8)

location_true_labels = []
location_pred_labels = []
object_true_labels = []
object_pred_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        location_labels = batch['location_labels']
        object_labels = batch['object_labels']
        
        location_logits, object_logits = model(input_ids, attention_mask)

        # 각 태스크별 예측
        predicted_location = torch.argmax(location_logits, dim=1)
        predicted_object = torch.argmax(object_logits, dim=1)

        location_true_labels.extend(location_labels.cpu().numpy())
        location_pred_labels.extend(predicted_location.cpu().numpy())
        object_true_labels.extend(object_labels.cpu().numpy())
        object_pred_labels.extend(predicted_object.cpu().numpy())

# 혼동 행렬 및 리포트 출력
location_report = classification_report(
    location_true_labels, 
    location_pred_labels, 
    labels=[0, 1, 2, 3],  # 모든 위치 클래스 레이블 명시
    target_names=['왼쪽', '오른쪽', '위', '없음'],
    zero_division=1
)

object_report = classification_report(
    object_true_labels, 
    object_pred_labels, 
    labels=[0, 1],  # 모든 객체 클래스 레이블 명시
    target_names=['사람', '물체'],
    zero_division=1
)

print("Location Task Classification Report:\n", location_report)
print("Object Task Classification Report:\n", object_report)

# 혼동 행렬 계산
location_cm = confusion_matrix(location_true_labels, location_pred_labels, labels=[0, 1, 2, 3])  # 모든 클래스 레이블 명시
object_cm = confusion_matrix(object_true_labels, object_pred_labels, labels=[0, 1])  # 모든 클래스 레이블 명시

# 혼동 행렬 시각화
plot_confusion_matrix(location_cm, labels=['왼쪽', '오른쪽', '위', '없음'], title='Location Task Confusion Matrix', cmap='Blues')
plot_confusion_matrix(object_cm, labels=['사람', '물체'], title='Object Task Confusion Matrix', cmap='Greens')
