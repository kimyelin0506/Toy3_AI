from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# text = "The S was snug but in a good way. I wore a push up bra, but this dress made me look like a 34C!"
#
# # 토큰화 결과 확인
# tokens = tokenizer.tokenize(text)
# print(tokens)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer("The S was snug but in a good way. I wore a push up bra, but this dress made me look like a 34C!",
                   return_tensors="pt")
from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased")
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

cls_embedding = last_hidden_state[:, 0, :]  # [1, 768]
print(cls_embedding)