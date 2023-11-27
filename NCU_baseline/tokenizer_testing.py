from transformers import BertTokenizer, BertForTokenClassification
import torch
model_name = "bert-base-cased"
# 載入預訓練 BERT 模型和分詞器
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=2)  # 假設二元分類

# 原始文本和對應的標籤
text = "The patient has a fever."
labels = [0, 0, 0, 1, 1, 1, 0]  # 假設標籤是二元分類的結果

# 分詞和向量化
tokens = tokenizer(text, return_tensors='pt')
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# 預測
outputs = model(input_ids, attention_mask=attention_mask)
predictions = torch.argmax(outputs.logits, dim=2)

print(tokens)
# 進行 offmapping
start_positions = []
end_positions = []
for i in range(len(labels)):
    print(i)
    token = tokenizer.tokenize(text.split()[i])[0]  # 取該 token 的第一個子詞
    print(token)
    if labels[i] == 1:  # 假設 1 表示實體的開始和結束
        start_positions.append(text.find(token))
        end_positions.append(start_positions[-1] + len(token) - 1)
        print("start_positions = {}, end_positions={}".format(start_positions, end_positions))

# 顯示結果
print("原始文本:", text)
print("預測標籤:", predictions[0].tolist())
print("實際標籤:", labels)
print("起始位置:", start_positions)
print("結束位置:", end_positions)
