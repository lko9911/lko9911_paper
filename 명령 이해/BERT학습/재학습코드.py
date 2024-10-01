# 필요한 라이브러리 임포트
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

# 1. 데이터셋 준비 (위치 정보_좌 우 / 분석 대상 / 분석 방법)
data = {
    'text': [
        "좌측의 사람 사진 분석 요청",
        "우측의 사람 사진 분석 요청",
        "왼쪽의 사람 사진 분석 요청",
        "사람 사진 분석",
        "왼쪽의 물건 분석",
        "우측의 물건 분석"
    ],
    'labels': [
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 1]
    ]
}


df = pd.DataFrame(data)

# 학습 데이터와 테스트 데이터로 나누기
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['labels'], test_size=0.2)

# 2. 토크나이저 준비 및 토크나이징
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts, labels):
    encoding = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    labels = torch.tensor(labels, dtype=torch.float)
    return encoding, labels

##
# 학습 데이터와 테스트 데이터로 나누기
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['labels'], test_size=0.2)

# Series를 리스트로 변환
train_labels = train_labels.tolist()
test_labels = test_labels.tolist()

# 데이터셋 토크나이징
train_encodings, train_labels = tokenize_data(train_texts, train_labels)
test_encodings, test_labels = tokenize_data(test_texts, test_labels)
##

# 3. PyTorch Dataset 클래스 정의
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# 4. 모델 준비
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # 3개의 레이블
model.train()

# 5. 옵티마이저와 손실 함수 설정
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = BCEWithLogitsLoss()

# 6. 모델 학습
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

for epoch in range(3):  # 3 에포크 동안 학습
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 7. 모델 평가
model.eval()
test_loader = DataLoader(test_dataset, batch_size=8)

correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted = torch.sigmoid(outputs.logits) > 0.5  # 0.5 기준으로 다중 레이블 분류
        correct += (predicted == labels).sum().item()
        total += labels.numel()

accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# 8. 모델 저장
model.save_pretrained('./my_bert_model')
tokenizer.save_pretrained('./my_bert_tokenizer')
