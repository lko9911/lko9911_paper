from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 사전학습 모델가져옴 / 연구를 위한 재학습 필요
# 토큰화 : 입력된 텍스트를 토큰으로 분할 (트랜스포머 모델 사용)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 명령어 입력
input_text = "GUI를 통해 입력받은 내용"
tokens = tokenizer(input_text, return_tensors='pt')

# 결과 예측 / 명령이 사람을 찾는다 & 물체를 분석해야한다 로 구분
# 목적에 따른 재학습 필요
output = model_bert(**tokens)
pred = torch.armax(output.logits, dim=1)
