import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 모델과 토크나이저 로드
model = BertForSequenceClassification.from_pretrained('my_bert_model')
tokenizer = BertTokenizer.from_pretrained('my_bert_tokenizer')

model.eval()  # 모델을 평가 모드로 설정

# 2. 예측할 문장 정의
texts = ["왼쪽의 물건을 분석해줘", "우측의 사람의 행동 예측"]

# 3. 입력 데이터 토크나이징
encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# 4. 모델을 사용한 예측
with torch.no_grad():
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# 5. 예측 결과 처리 (sigmoid를 통해 다중 레이블 분류)
predictions = torch.sigmoid(logits) > 0.5  # 각 클래스에 대해 0.5 이상이면 1로 예측
print(predictions)
